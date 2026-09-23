"""
Internal NUFFT imaging engine shared by all imagers.

The public functions in :mod:`snapshot_imager.imager` are thin layers over
this module. The same code runs on the CPU (NumPy + FINUFFT) and the GPU
(CuPy + cuFINUFFT); the only backend-specific step is moving arrays to and
from the device.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import tqdm

from .coordinates import compute_image_grid
from .core import prepare_weighted_visibilities
from .data_models import ImagingData

# Imaging kernel sign: exp(-2πi(ul + vm)) for visibilities that follow the
# pyuvdata/pyuvsim convention V ∝ exp(+2πi(ul + vm + wn)).
ISIGN = -1

METHODS = ("type1", "type3")


@dataclass(frozen=True)
class Backend:
    """Array and NUFFT libraries: NumPy/FINUFFT on CPU or CuPy/cuFINUFFT on GPU."""

    xp: Any
    nufft: Any
    gpu: bool

    def to_device(self, array):
        """Move a host array to the backend's device (no-op on CPU)."""
        return self.xp.asarray(array) if self.gpu else array

    def to_host(self, array):
        """Move a backend array to host memory (no-op on CPU)."""
        return self.xp.asnumpy(array) if self.gpu else array


def get_backend(use_gpu: bool = False) -> Backend:
    """
    Return the GPU backend if requested and usable, otherwise the CPU backend.

    If the GPU was requested but CuPy/cuFINUFFT cannot be imported, or no CUDA
    device is available, a RuntimeWarning is issued and the CPU is used.
    """
    if use_gpu:
        try:
            import cupy
            import cufinufft

            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device found")
            return Backend(cupy, cufinufft, gpu=True)
        except Exception as err:  # ImportError, CUDA runtime errors, ...
            warnings.warn(
                f"GPU imaging unavailable ({err}); falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )

    import finufft

    return Backend(np, finufft, gpu=False)


class GridTransform:
    """
    Evaluate ``I(l, m) = sum_j c_j exp(-2πi (u_j l + v_j m))`` on an image grid.

    The grid is ``compute_image_grid(npix, fov)`` and output images are indexed
    ``(m, l)``. Each call transforms ``n_trans`` data vectors that share the
    same uv points.

    With ``method="type1"`` one FINUFFT plan is created up front and reused
    for every :meth:`set_points` call. With ``method="type3"`` a plan is
    created per set of points; Type 3 evaluates the same sum at the grid
    points directly and is much slower for regular grids.

    Parameters
    ----------
    backend : Backend
        CPU or GPU backend.
    npix, fov : int, float
        Image size in pixels and field of view in degrees.
    n_trans : int
        Number of data vectors transformed per call.
    method : {"type1", "type3"}
        NUFFT type used to evaluate the image.
    eps : float
        NUFFT tolerance.
    complex_dtype : numpy dtype
        ``complex128`` or ``complex64``; sets the NUFFT precision.
    uv_extent : float, optional
        Largest |u| or |v| over all points (required for Type 3, which uses
        it to keep point and target coordinates well scaled).
    """

    def __init__(
        self,
        backend: Backend,
        npix: int,
        fov: float,
        n_trans: int,
        *,
        method: str = "type1",
        eps: float = 1e-13,
        complex_dtype=np.complex128,
        uv_extent: float | None = None,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")

        self.backend = backend
        self.npix = npix
        self.n_trans = n_trans
        self.method = method
        self.eps = eps
        self.complex_dtype = np.dtype(complex_dtype)
        self.real_dtype = np.float32 if self.complex_dtype == np.complex64 else np.float64

        if method == "type1":
            # Pixel k (k = -npix//2, ...) sits at l = k * 2 sin(fov/2) / npix,
            # so scaling uv by 4π sin(fov/2) / npix maps the grid onto FINUFFT modes.
            self._scale = 4 * np.pi / npix * np.sin(np.deg2rad(fov / 2))
            self._plan = backend.nufft.Plan(
                1,
                (npix, npix),
                n_trans=n_trans,
                eps=eps,
                isign=ISIGN,
                dtype=self.complex_dtype,
                modeord=0,
            )
        else:
            if uv_extent is None or not uv_extent > 0:
                raise ValueError("Type 3 imaging needs a positive uv_extent")
            # The Type 3 sum only depends on the products (u*scale)(l/scale), so
            # scale the points by 2π/uv_extent and the targets by uv_extent.
            _, _, lgrid, mgrid = compute_image_grid(npix, fov)
            self._scale = 2 * np.pi / uv_extent
            self._targets = tuple(
                backend.to_device((grid.ravel() * uv_extent).astype(self.real_dtype))
                for grid in (lgrid, mgrid)
            )
            self._plan = None

    def set_points(self, u: np.ndarray, v: np.ndarray) -> None:
        """Set the uv points (in wavelengths) for subsequent calls."""
        x = self.backend.to_device((np.asarray(u) * self._scale).astype(self.real_dtype))
        y = self.backend.to_device((np.asarray(v) * self._scale).astype(self.real_dtype))
        if self.method == "type1":
            # Pass v as FINUFFT's first coordinate so the output comes out
            # indexed (m, l) directly, without a transposed copy.
            self._plan.setpts(y, x)
        else:
            self._plan = self.backend.nufft.Plan(
                3,
                2,
                n_trans=self.n_trans,
                eps=self.eps,
                isign=ISIGN,
                dtype=self.complex_dtype,
            )
            self._plan.setpts(x, y, s=self._targets[0], t=self._targets[1])

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data of shape (n_trans, npoints) into images.

        Returns a host array of shape (n_trans, npix, npix), indexed (m, l).
        """
        data = np.ascontiguousarray(data, dtype=self.complex_dtype)
        data = data.reshape(self.n_trans, -1)
        out = self.backend.to_host(self._plan.execute(self.backend.to_device(data)))
        return out.reshape(self.n_trans, self.npix, self.npix)


# Largest block of weighted visibilities prepared at once in image_per_channel
_CHUNK_BYTES = 128 * 2**20


def _channel_chunks(data: ImagingData, itemsize: int):
    """Yield (start, stop) channel ranges whose data fits in _CHUNK_BYTES."""
    nbls, ntimes, nfreqs = data.shape
    size = max(1, min(nfreqs, _CHUNK_BYTES // max(nbls * ntimes * itemsize, 1)))
    for start in range(0, nfreqs, size):
        yield start, min(start + size, nfreqs)


def weights_constant_in_time(data: ImagingData) -> bool:
    """True if every baseline's weights are the same at all times."""
    if data.ntimes == 1:
        return True
    for start, stop in _channel_chunks(data, data.weights.itemsize):
        chunk = data.weights[:, :, start:stop]
        if not np.all(chunk == chunk[:, :1]):
            return False
    return True


def image_per_channel(
    data: ImagingData,
    transform: GridTransform,
    *,
    rm_phasor: np.ndarray | None = None,
    psf_rows: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Image every channel; each snapshot is normalized by its summed weights.

    Returns ``(images, psf)``. ``images`` has shape (ntimes, nfreqs, npix, npix),
    or (ntimes, 1, npix, npix) when ``rm_phasor`` is given (each channel is
    multiplied by ``rm_phasor[fi]`` and the channels are summed). Images of
    Hermitian data are real-valued unless ``rm_phasor`` is given.

    With ``psf_rows`` > 0, the synthesized beam (the image of unit
    visibilities, normalized the same way) is computed for the first
    ``psf_rows`` times in the same transforms as the images (the transform
    must have ``n_trans = ntimes + psf_rows``); ``psf`` then has shape
    (psf_rows, nfreqs or 1, npix, npix). Otherwise ``psf`` is None.
    """
    _, ntimes, nfreqs = data.shape
    npix = transform.npix
    if transform.n_trans != ntimes + psf_rows:
        raise ValueError("transform.n_trans must equal ntimes + psf_rows")
    real_output = data.hermitian and rm_phasor is None
    dtype = transform.real_dtype if real_output else transform.complex_dtype
    nout = nfreqs if rm_phasor is None else 1
    images = np.zeros((ntimes, nout, npix, npix), dtype=dtype)
    psf = np.zeros((psf_rows, nout, npix, npix), dtype=dtype) if psf_rows else None

    progress = tqdm.tqdm(total=nfreqs, desc="Imaging frequencies", disable=not verbose)
    for start, stop in _channel_chunks(data, transform.complex_dtype.itemsize):
        # Read the chunk of the cube once, as contiguous (channel, time, baseline)
        # blocks, instead of gathering strided slices channel by channel.
        weights = data.weights[:, :, start:stop].astype(transform.real_dtype, copy=False)
        weighted = np.ascontiguousarray(
            (data.vis[:, :, start:stop] * weights).transpose(2, 1, 0),
            dtype=transform.complex_dtype,
        )
        if psf_rows:
            # The beam is the image of unit visibilities: transform the weights
            beam_weights = np.ascontiguousarray(
                weights[:, :psf_rows].transpose(2, 1, 0), dtype=transform.complex_dtype
            )
        # Peak of each snapshot's synthesized beam, shape (channel, time). With
        # implied conjugates both the image and the beam peak double, so the
        # factor of 2 cancels in the normalization.
        sum_weights = weights.sum(axis=0).T[:, :, None, None]
        u = np.ascontiguousarray(data.u[:, start:stop].T)
        v = np.ascontiguousarray(data.v[:, start:stop].T)

        for k, fi in enumerate(range(start, stop)):
            transform.set_points(u[k], v[k])
            block = weighted[k]
            if psf_rows:
                block = np.concatenate([block, beam_weights[k]])
            out = transform(block)
            if data.hermitian:
                out = out.real
            norm = sum_weights[k]
            _store(images, fi, out[:ntimes], norm, rm_phasor)
            if psf_rows:
                _store(psf, fi, out[ntimes:], norm[:psf_rows], rm_phasor)
            progress.update()
    progress.close()

    return images, psf


def _store(cube, fi, image, norm, rm_phasor):
    """Normalize one channel's images into ``cube`` (or accumulate with rm_phasor)."""
    if rm_phasor is None:
        np.divide(image, norm, out=cube[:, fi], where=norm != 0)
    else:
        channel = np.divide(image, norm, out=np.zeros_like(image), where=norm != 0)
        cube[:, 0] += channel * rm_phasor[fi]


def image_mfs(
    data: ImagingData,
    transform: GridTransform,
    *,
    psf_rows: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Image all channels together into one (unnormalized) image per time.

    Returns ``(images, psf)``: images of shape (ntimes, 1, npix, npix),
    real-valued for Hermitian data, and, if ``psf_rows`` > 0, the synthesized
    beam for the first ``psf_rows`` times (shape (psf_rows, 1, npix, npix)),
    otherwise None.
    """
    _, ntimes, _ = data.shape
    npix = transform.npix
    dtype = transform.real_dtype if data.hermitian else transform.complex_dtype
    images = np.zeros((ntimes, 1, npix, npix), dtype=dtype)
    psf = np.zeros((psf_rows, 1, npix, npix), dtype=dtype) if psf_rows else None

    def combine(image):
        # Implied conjugates contribute the complex conjugate: 2 * Re(image)
        return 2 * image.real if data.hermitian else image

    # The uv points don't depend on time: set them once, raveled in
    # (nfreqs, nbls) order to match the weighted data below.
    transform.set_points(np.ravel(data.u.T), np.ravel(data.v.T))
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times", disable=not verbose):
        weighted = prepare_weighted_visibilities(data.vis, data.weights, time_idx=ti)
        images[ti, 0] = combine(transform(weighted.reshape(1, -1))[0])
    for ti in range(psf_rows):
        beam_weights = np.ascontiguousarray(data.weights[:, ti, :].T)
        psf[ti, 0] = combine(transform(beam_weights.reshape(1, -1))[0])

    return images, psf


# ---------------------------------------------------------------------------
# Evaluation at arbitrary sky positions
# ---------------------------------------------------------------------------

POINT_METHODS = ("auto", "direct", "type3")

# method="auto" uses the direct sum when (uv points) x (targets) is at most this
_DIRECT_MAX_TERMS = 3_000_000
# Largest (uv points x targets) phase block held in memory by the direct sum
_PHASE_BLOCK_BYTES = 64 * 2**20


class PointTransform:
    """
    Evaluate ``sum_j c_j exp(-2πi (u_j l_k + v_j m_k [+ w_j n_k]))`` at
    arbitrary directions ``(l_k, m_k[, n_k])``.

    The optional w-term follows the pyuvdata/pyuvsim convention for
    unprojected (drift-scan) data, V ∝ exp(+2πi (u l + v m + w n)).

    Parameters
    ----------
    backend : Backend
        CPU or GPU backend.
    n_trans : int
        Number of data vectors transformed per call.
    method : {"direct", "type3"}
        "direct" evaluates the sum exactly (a matrix product, best for few
        targets); "type3" uses a Type 3 NUFFT with tolerance ``eps``.
    eps : float
        NUFFT tolerance for "type3".
    complex_dtype : numpy dtype
        ``complex128`` or ``complex64``.
    include_w : bool
        Include the w-term (requires ``w`` points and ``n`` targets).
    uv_extent, w_extent : float
        Largest |u|, |v| and |w| over all points (used by "type3" to keep
        coordinates well scaled).
    """

    def __init__(
        self,
        backend: Backend,
        n_trans: int,
        *,
        method: str,
        eps: float,
        complex_dtype=np.complex128,
        include_w: bool = False,
        uv_extent: float = 1.0,
        w_extent: float = 1.0,
    ):
        if method not in ("direct", "type3"):
            raise ValueError(f"method must be 'direct' or 'type3', got {method!r}")
        self.backend = backend
        self.n_trans = n_trans
        self.method = method
        self.eps = eps
        self.complex_dtype = np.dtype(complex_dtype)
        self.real_dtype = np.float32 if self.complex_dtype == np.complex64 else np.float64
        self.include_w = include_w
        self._uv_extent = uv_extent if uv_extent > 0 else 1.0
        self._w_extent = w_extent if w_extent > 0 else 1.0
        self._points = None
        self._targets = None
        self._plan = None
        self._phase = None

    def set_points(self, u, v, w=None):
        """Set the uv(w) points in wavelengths."""
        coords = [u, v] + ([w] if self.include_w else [])
        self._points = [np.asarray(c, dtype=self.real_dtype) for c in coords]
        self._plan = self._phase = None

    def set_targets(self, l, m, n=None):
        """Set the target direction cosines."""
        coords = [l, m] + ([n] if self.include_w else [])
        self._targets = [np.asarray(c, dtype=self.real_dtype) for c in coords]
        self._plan = self._phase = None

    @property
    def ntargets(self):
        return len(self._targets[0])

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Transform data of shape (n_trans, npoints) to (n_trans, ntargets)."""
        data = np.ascontiguousarray(data, dtype=self.complex_dtype)
        data = data.reshape(self.n_trans, -1)
        if self.ntargets == 0:
            return np.zeros((self.n_trans, 0), dtype=self.complex_dtype)
        if self.method == "direct":
            return self._direct(data)
        return self._type3(data)

    def _direct(self, data):
        xp, be = self.backend.xp, self.backend
        d = be.to_device(data)
        npts = len(self._points[0])
        block = max(1, _PHASE_BLOCK_BYTES // max(npts * self.complex_dtype.itemsize, 1))
        # Cache the phase matrix when it fits in one block (e.g. MFS, where the
        # same points and targets are used for every time)
        if self._phase is None and self.ntargets <= block:
            self._phase = self._phase_block(0, self.ntargets)
        out = []
        for start in range(0, self.ntargets, block):
            stop = min(start + block, self.ntargets)
            phase = self._phase if self._phase is not None else self._phase_block(start, stop)
            out.append(d @ phase)
        result = out[0] if len(out) == 1 else xp.concatenate(out, axis=1)
        return be.to_host(result)

    def _phase_block(self, start, stop):
        xp, be = self.backend.xp, self.backend
        arg = None
        for p, t in zip(self._points, self._targets, strict=True):
            term = be.to_device(p)[:, None] * be.to_device(t[start:stop])[None, :]
            arg = term if arg is None else arg + term
        return xp.exp((-2j * np.pi) * arg).astype(self.complex_dtype, copy=False)

    def _type3(self, data):
        be = self.backend
        if self._plan is None:
            # The sum depends only on products point*target, so scale points
            # by 2π/extent and targets by extent in each dimension.
            dim = len(self._points)
            extents = [self._uv_extent, self._uv_extent, self._w_extent][:dim]
            pts = [
                be.to_device((p * (2 * np.pi / e)).astype(self.real_dtype))
                for p, e in zip(self._points, extents, strict=True)
            ]
            tgt = [
                be.to_device((t * e).astype(self.real_dtype))
                for t, e in zip(self._targets, extents, strict=True)
            ]
            self._plan = be.nufft.Plan(
                3,
                dim,
                n_trans=self.n_trans,
                eps=self.eps,
                isign=ISIGN,
                dtype=self.complex_dtype,
            )
            if dim == 2:
                self._plan.setpts(pts[0], pts[1], s=tgt[0], t=tgt[1])
            else:
                self._plan.setpts(pts[0], pts[1], pts[2], s=tgt[0], t=tgt[1], u=tgt[2])
        out = be.to_host(self._plan.execute(be.to_device(data)))
        return out.reshape(self.n_trans, -1)


def evaluate_points(
    data: ImagingData,
    transform: PointTransform,
    l: np.ndarray,
    m: np.ndarray,
    n: np.ndarray | None,
    *,
    mfs: bool = False,
    psf: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Dirty image (or synthesized beam, with ``psf``) at arbitrary directions.

    ``l``, ``m`` (and ``n``) have shape (npoints,) for the same directions at
    every time, or (ntimes, npoints) for time-dependent directions; the
    transform must have ``n_trans = ntimes`` in the first case and 1 in the
    second (or 1 for MFS). Returns (ntimes, nfreqs or 1, npoints): per-channel
    values normalized by each snapshot's summed weights, or unnormalized MFS
    sums; real-valued for Hermitian data.
    """
    _, ntimes, nfreqs = data.shape
    per_time = np.ndim(l) == 2
    npoints = np.shape(l)[-1]
    dtype = transform.real_dtype if data.hermitian else transform.complex_dtype
    nout = 1 if mfs else nfreqs
    values = np.zeros((ntimes, nout, npoints), dtype=dtype)

    def targets(ti):
        if not per_time:
            return l, m, n
        return l[ti], m[ti], None if n is None else n[ti]

    def real_part(out):
        return out.real if data.hermitian else out

    if not per_time:
        transform.set_targets(*targets(0))

    if mfs:
        transform.set_points(np.ravel(data.u.T), np.ravel(data.v.T), np.ravel(data.w.T))
        for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times", disable=not verbose):
            if psf:
                block = np.ravel(data.weights[:, ti, :].T)
            else:
                block = np.ravel(
                    prepare_weighted_visibilities(data.vis, data.weights, time_idx=ti)
                )
            if per_time:
                transform.set_targets(*targets(ti))
            out = real_part(transform(block[None])[0])
            # Implied conjugates contribute the complex conjugate: 2 * Re
            values[ti, 0] = 2 * out if data.hermitian else out
        return values

    progress = tqdm.tqdm(total=nfreqs, desc="Imaging frequencies", disable=not verbose)
    for start, stop in _channel_chunks(data, transform.complex_dtype.itemsize):
        weights = data.weights[:, :, start:stop].astype(transform.real_dtype, copy=False)
        source = weights if psf else data.vis[:, :, start:stop] * weights
        blocks = np.ascontiguousarray(
            source.transpose(2, 1, 0), dtype=transform.complex_dtype
        )
        sum_weights = weights.sum(axis=0).T[:, :, None]
        u = np.ascontiguousarray(data.u[:, start:stop].T)
        v = np.ascontiguousarray(data.v[:, start:stop].T)
        w = np.ascontiguousarray(data.w[:, start:stop].T)

        for k, fi in enumerate(range(start, stop)):
            transform.set_points(u[k], v[k], w[k])
            if per_time:
                out = np.empty((ntimes, npoints), dtype=transform.complex_dtype)
                for ti in range(ntimes):
                    transform.set_targets(*targets(ti))
                    out[ti] = transform(blocks[k, ti][None])[0]
            else:
                out = transform(blocks[k])
            norm = sum_weights[k]
            # With implied conjugates the factor of 2 cancels in the normalization
            np.divide(real_part(out), norm, out=values[:, fi], where=norm != 0)
            progress.update()
    progress.close()
    return values
