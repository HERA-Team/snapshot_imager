"""
Coordinate transformations and phase tracking utilities.

This module handles coordinate system transformations and phase tracking
operations for radio interferometry imaging.
"""
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u


def radec_to_lmn(
    ra,
    dec,
    times,
    telescope_loc: EarthLocation,
):
    """
    Direction cosines (l, m, n) of sky positions at each time.

    Parameters
    ----------
    ra, dec : float or array_like
        Right ascension and declination (ICRS) in degrees, scalars or arrays
        of shape (nsrc,).
    times : np.ndarray or astropy.time.Time
        Times of the observation, Julian dates if not a ``Time``. Shape
        (ntimes,).
    telescope_loc : EarthLocation
        Location of the telescope.

    Returns
    -------
    l, m, n : np.ndarray
        Direction cosines toward East (l), North (m) and Up (n), each of shape
        (ntimes, nsrc). ``n`` is the sine of the elevation, so it is negative
        for positions below the horizon.

    Notes
    -----
    These are the coordinates used by the imagers: pass ``l`` and ``m`` (and
    ``n``) to :func:`~snapshot_imager.dirty_image_points` to evaluate the image
    at catalog positions, or compare them with ``ImageResult.l_coords`` and
    ``m_coords``.
    """
    if not isinstance(times, Time):
        times = Time(np.atleast_1d(times), format="jd")
    times = times.reshape(-1)
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    if ra.shape != dec.shape or ra.ndim != 1:
        raise ValueError(
            f"ra and dec must be scalars or 1D arrays of the same shape, "
            f"got {ra.shape} and {dec.shape}"
        )

    sources = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    frame = AltAz(obstime=times[:, None], location=telescope_loc)
    altaz = sources[None, :].transform_to(frame)

    alt, az = altaz.alt.rad, altaz.az.rad
    l = np.cos(alt) * np.sin(az)
    m = np.cos(alt) * np.cos(az)
    n = np.sin(alt)
    return l, m, n


def phase_track_to_source(
    vis: np.ndarray,
    uvw: np.ndarray,
    times: np.ndarray,
    ra_src: float,
    dec_src: float,
    telescope_loc: EarthLocation,
) -> np.ndarray:
    """
    Phase track visibility data to a specific sky position.
    
    This function applies a phase rotation to the visibility data to track
    a source at a given RA/Dec position. This is equivalent to recentering
    the image at the source position.
    
    Parameters
    ----------
    vis : np.ndarray
        The visibility data to be phased. Shape (nbls, ntimes, nfreqs)
    uvw : np.ndarray
        The uvw coordinates of the visibility data in wavelengths. 
        Shape (nbls, 3, nfreqs)
    times : np.ndarray
        The times of the visibility data (Julian dates). Shape (ntimes,)
    ra_src : float
        The right ascension of the source in degrees.
    dec_src : float
        The declination of the source in degrees.
    telescope_loc : EarthLocation
        The location of the telescope.
    
    Returns
    -------
    np.ndarray
        The phase-tracked visibility data. Shape (nbls, ntimes, nfreqs)
    
    Notes
    -----
    The phase rotation is computed using direction cosines (l, m, n) where:
    - l: East direction cosine
    - m: North direction cosine  
    - n: Up direction cosine (the sine of the elevation)
        
    The phase correction applied is:
        exp(-2j * π * (u*l + v*m + w*n))
    
    where (u, v, w) are baseline coordinates in wavelengths and (l, m, n) are
    direction cosines to the source.

    This follows the pyuvdata/pyuvsim convention used for HERA data, where
    uvw = xyz(ant2) - xyz(ant1) and a source in direction (l, m, n) has
    visibility V ∝ exp(+2πi (u*l + v*m + w*n)); the correction above therefore
    brings the source to the phase center.
    """
    # Direction cosines of the source at each time, shape (ntimes,)
    l, m, n = (x[:, 0] for x in radec_to_lmn(ra_src, dec_src, times, telescope_loc))
    
    # Extract UVW coordinates
    ucoords = uvw[:, 0, :]  # Shape: (nbls, nfreqs)
    vcoords = uvw[:, 1, :]  # Shape: (nbls, nfreqs)
    wcoords = uvw[:, 2, :]  # Shape: (nbls, nfreqs)
    
    # Compute phase correction
    phase = np.exp(
        -2j * np.pi * (
            ucoords[:, None, :] * l[None, :, None] + 
            vcoords[:, None, :] * m[None, :, None] + 
            wcoords[:, None, :] * n[None, :, None]
        )
    )
    
    return vis * phase


def compute_image_grid(npix: int, fov: float, flat_projection: bool = True):
    """
    Compute the l, m coordinate grid for an image.
    
    Parameters
    ----------
    npix : int
        Number of pixels per dimension
    fov : float
        Field of view in degrees
    flat_projection : bool, optional
        If True (default), use a flat/orthographic projection where coordinates
        are spaced uniformly in direction cosine space. If False, space angles
        uniformly and then take the sine, which more accurately represents
        the sphere for wide fields of view.
    
    Returns
    -------
    lcoords : np.ndarray
        L-coordinates (East direction cosines), shape (npix,)
    mcoords : np.ndarray
        M-coordinates (North direction cosines), shape (npix,)
    lgrid : np.ndarray
        2D L-coordinate grid, shape (npix, npix)
    mgrid : np.ndarray
        2D M-coordinate grid, shape (npix, npix)
    
    Notes
    -----
    Direction cosines are computed using:
        l = sin(θ_E)
        m = sin(θ_N)
    
    where θ_E and θ_N are angular offsets in the East and North directions.

    Pixel ``npix // 2`` is the phase center (l = m = 0) for both even and odd
    ``npix``, with pixel offsets ``-(npix // 2), ..., npix - npix // 2 - 1``.
    This matches FINUFFT's centered (``modeord=0``) mode ordering, so the
    Type 1 imagers evaluate the image exactly at these coordinates.
    """
    offsets = np.arange(npix) - npix // 2
    if flat_projection:
        extent = np.sin(np.deg2rad(fov / 2))
        lcoords = offsets * (2 * extent / npix)
    else:
        extent = np.deg2rad(fov / 2)
        lcoords = np.sin(offsets * (2 * extent / npix))
    mcoords = lcoords.copy()

    lgrid, mgrid = np.meshgrid(lcoords, mcoords)
    
    return lcoords, mcoords, lgrid, mgrid


def _below_horizon(lgrid: np.ndarray, mgrid: np.ndarray) -> np.ndarray:
    """Mask of image pixels outside the visible sky (l**2 + m**2 > 1)."""
    return lgrid**2 + mgrid**2 > 1


def compute_baseline_extent(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the maximum baseline extent in the UV plane.
    
    Parameters
    ----------
    u : np.ndarray
        U-coordinates
    v : np.ndarray
        V-coordinates
    
    Returns
    -------
    float
        Maximum baseline extent (max of |u| and |v|)
    """
    return max(np.max(np.abs(u)), np.max(np.abs(v)))
