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


def image_per_channel(
    data: ImagingData,
    transform: GridTransform,
    *,
    rm_phasor: np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Image every channel; each snapshot is normalized by its summed weights.

    Returns an array of shape (ntimes, nfreqs, npix, npix), or
    (ntimes, 1, npix, npix) when ``rm_phasor`` is given (each channel is
    multiplied by ``rm_phasor[fi]`` and the channels are summed). Images of
    Hermitian data are real-valued unless ``rm_phasor`` is given.
    """
    _, ntimes, nfreqs = data.shape
    npix = transform.npix
    real_output = data.hermitian and rm_phasor is None
    dtype = transform.real_dtype if real_output else transform.complex_dtype
    nout = nfreqs if rm_phasor is None else 1
    images = np.zeros((ntimes, nout, npix, npix), dtype=dtype)

    progress = tqdm.tqdm(total=nfreqs, desc="Imaging frequencies", disable=not verbose)
    for start, stop in _channel_chunks(data, transform.complex_dtype.itemsize):
        # Read the chunk of the cube once, as contiguous (channel, time, baseline)
        # blocks, instead of gathering strided slices channel by channel.
        weights = data.weights[:, :, start:stop].astype(transform.real_dtype, copy=False)
        weighted = np.ascontiguousarray(
            (data.vis[:, :, start:stop] * weights).transpose(2, 1, 0),
            dtype=transform.complex_dtype,
        )
        # Peak of each snapshot's synthesized beam, shape (channel, time). With
        # implied conjugates both the image and the beam peak double, so the
        # factor of 2 cancels in the normalization.
        sum_weights = weights.sum(axis=0).T[:, :, None, None]
        u = np.ascontiguousarray(data.u[:, start:stop].T)
        v = np.ascontiguousarray(data.v[:, start:stop].T)

        for k, fi in enumerate(range(start, stop)):
            transform.set_points(u[k], v[k])
            image = transform(weighted[k])
            if data.hermitian:
                image = image.real
            norm = sum_weights[k]
            if rm_phasor is None:
                np.divide(image, norm, out=images[:, fi], where=norm != 0)
            else:
                channel = np.divide(
                    image, norm, out=np.zeros_like(image), where=norm != 0
                )
                images[:, 0] += channel * rm_phasor[fi]
            progress.update()
    progress.close()

    return images


def image_mfs(
    data: ImagingData, transform: GridTransform, *, verbose: bool = False
) -> np.ndarray:
    """
    Image all channels together into one (unnormalized) image per time.

    Returns an array of shape (ntimes, 1, npix, npix), real-valued for
    Hermitian data.
    """
    _, ntimes, _ = data.shape
    npix = transform.npix
    dtype = transform.real_dtype if data.hermitian else transform.complex_dtype
    images = np.zeros((ntimes, 1, npix, npix), dtype=dtype)

    # The uv points don't depend on time: set them once, raveled in
    # (nfreqs, nbls) order to match the weighted data below.
    transform.set_points(np.ravel(data.u.T), np.ravel(data.v.T))
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times", disable=not verbose):
        weighted = prepare_weighted_visibilities(data.vis, data.weights, time_idx=ti)
        image = transform(weighted.reshape(1, -1))[0]
        # Implied conjugates contribute the complex conjugate: 2 * Re(image)
        images[ti, 0] = 2 * image.real if data.hermitian else image

    return images
