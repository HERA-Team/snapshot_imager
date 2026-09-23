"""
Data models for snapshot imaging.

This module defines structured data containers for visibility data and imaging results.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ImagingData:
    """
    Container for visibility data prepared for imaging.
    
    Attributes
    ----------
    vis : np.ndarray
        Visibility data, shape (nbls, ntimes, nfreqs)
    weights : np.ndarray
        Visibility weights, shape (nbls, ntimes, nfreqs)
    uvw : np.ndarray
        UVW coordinates in wavelengths, shape (nbls, 3, nfreqs)
    times : np.ndarray
        Time stamps (Julian dates), shape (ntimes,)
    freqs : np.ndarray
        Frequency channels in Hz, shape (nfreqs,)
    hermitian : bool
        If True, the data holds one baseline of each conjugate pair and the
        conjugates (uvw -> -uvw, vis -> conj(vis), same weights) are implied.
        Imaging then uses half the points and returns real-valued images.
        If False (default), every baseline to be imaged is stored explicitly.
        ``unpack_data_containers`` returns Hermitian data by default.
    """
    vis: np.ndarray
    weights: np.ndarray
    uvw: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    hermitian: bool = False

    def __post_init__(self):
        """Validate shapes after initialization."""
        nbls, ntimes, nfreqs = self.vis.shape
        
        # Validate visibility and weights match
        if self.weights.shape != (nbls, ntimes, nfreqs):
            raise ValueError(
                f"weights shape {self.weights.shape} doesn't match "
                f"vis shape {self.vis.shape}"
            )
        
        # Validate UVW coordinates
        if self.uvw.shape != (nbls, 3, nfreqs):
            raise ValueError(
                f"uvw shape {self.uvw.shape} doesn't match expected "
                f"shape ({nbls}, 3, {nfreqs})"
            )
        
        # Validate time and frequency arrays
        if len(self.times) != ntimes:
            raise ValueError(
                f"times length {len(self.times)} doesn't match ntimes {ntimes}"
            )
        
        if len(self.freqs) != nfreqs:
            raise ValueError(
                f"freqs length {len(self.freqs)} doesn't match nfreqs {nfreqs}"
            )
    
    @property
    def shape(self):
        """Return (nbls, ntimes, nfreqs) shape tuple."""
        return self.vis.shape
    
    @property
    def nbls(self):
        """Number of baselines."""
        return self.vis.shape[0]
    
    @property
    def ntimes(self):
        """Number of time samples."""
        return self.vis.shape[1]
    
    @property
    def nfreqs(self):
        """Number of frequency channels."""
        return self.vis.shape[2]
    
    @property
    def u(self):
        """U coordinates, shape (nbls, nfreqs)."""
        return self.uvw[:, 0, :]
    
    @property
    def v(self):
        """V coordinates, shape (nbls, nfreqs)."""
        return self.uvw[:, 1, :]
    
    @property
    def w(self):
        """W coordinates, shape (nbls, nfreqs)."""
        return self.uvw[:, 2, :]


@dataclass
class ImageResult:
    """
    Container for imaging results.
    
    Attributes
    ----------
    images : np.ndarray
        Image cube, shape (ntimes, nfreqs, npix, npix), indexed
        ``images[time, freq, m, l]``.
    l_coords : np.ndarray
        L-coordinate values (direction cosine, East), shape (npix,). The
        legacy ``snapshot_imager_type3`` / ``snapshot_imager_mfs_type_3``
        functions instead return the flattened coordinates of every pixel,
        shape (npix*npix,).
    m_coords : np.ndarray
        M-coordinate values (direction cosine, North), shape (npix,), or
        (npix*npix,) for the legacy Type 3 functions.
    fov : float
        Field of view in degrees
    npix : int
        Number of pixels per dimension
    times : np.ndarray, optional
        Time stamps (Julian dates) of the images, shape (ntimes,).
    freqs : np.ndarray, optional
        Frequencies in Hz along the image frequency axis, shape (nfreqs,).
        For images that combine channels (MFS, or ``rm_phasor``) this is the
        mean frequency of the input channels.
    """
    images: np.ndarray
    l_coords: np.ndarray
    m_coords: np.ndarray
    fov: float
    npix: int
    times: np.ndarray | None = None
    freqs: np.ndarray | None = None

    @property
    def shape(self):
        """Return image cube shape (ntimes, nfreqs, npix, npix)."""
        return self.images.shape
    
    @property
    def ntimes(self):
        """Number of time samples."""
        return self.images.shape[0]
    
    @property
    def nfreqs(self):
        """Number of frequency channels."""
        return self.images.shape[1]