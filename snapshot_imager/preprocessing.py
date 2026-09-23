"""
Data preprocessing and unpacking utilities.

This module handles conversion of various data formats into the standard
ImagingData format used by the imaging algorithms.
"""
import warnings

import numpy as np
from astropy import constants
from hera_cal import datacontainer, utils

from .data_models import ImagingData


def unpack_data_containers(
    data: datacontainer.DataContainer, 
    flags: datacontainer.DataContainer, 
    nsamples: datacontainer.DataContainer,
    pol: str = 'ee',
    antpos: dict = None,
    freqs: np.ndarray = None,
    time_slice: slice = slice(0, None),
    freq_slice: slice = slice(0, None),
    antpairs: list = None,
    weight_by_nsamples: bool = True,
    include_conjugates: bool = False,
) -> ImagingData:
    """
    Unpack HERA data containers into imaging format.
    
    This function extracts visibility data, flags, and metadata from HERA
    DataContainer objects and formats them for use with the imaging algorithms.
    
    Parameters
    ----------
    data : datacontainer.DataContainer
        The visibility data to be unpacked.
    flags : datacontainer.DataContainer
        The flags for the visibility data.
    nsamples : datacontainer.DataContainer
        The nsamples for the visibility data.
    pol : str, optional
        The polarization to be used. Default is 'ee'.
    antpos : dict, optional
        The antenna positions. If None, will use the positions from the data container.
    freqs : np.ndarray, optional
        The frequencies to be used. If None, will use the frequencies from the data container.
    time_slice : slice, optional
        The time slice to be used. Default is slice(0, None).
    freq_slice : slice, optional
        The frequency slice to be used. Default is slice(0, None).
    antpairs : list[tuple], optional
        The antenna pairs to be used. If None, will use all cross-correlation
        antenna pairs in the data container. Autocorrelations are always
        excluded (with a warning if any are passed here).
    weight_by_nsamples : bool, optional
        Whether to weight the data by nsamples. Default is True.
    include_conjugates : bool, optional
        If False (default), return one row per baseline with
        ``hermitian=True``: the conjugate baselines are implied, which halves
        the memory and imaging work, and images are real-valued. If True,
        also store each conjugate (reversed) baseline explicitly, with
        ``hermitian=False`` (the layout used before version 0.4).

    Returns
    -------
    ImagingData
        Structured container with visibility data ready for imaging.

    Notes
    -----
    Either way the imaged visibilities are Hermitian: each baseline's
    conjugate (-uvw, conj(vis)) is either implied or stored explicitly.

    Autocorrelations have zero baseline length, so they carry no spatial
    information and would only add a constant offset to a dirty image; they
    are therefore never included.
    """
    if antpairs is None:
        antpairs = [ap for ap in data.antpairs() if ap[0] != ap[1]]
    else:
        n_autos = sum(ap[0] == ap[1] for ap in antpairs)
        if n_autos:
            warnings.warn(
                f"Dropping {n_autos} autocorrelation(s) from antpairs: "
                "autocorrelations only add a constant offset to dirty images.",
                stacklevel=2,
            )
            antpairs = [ap for ap in antpairs if ap[0] != ap[1]]

    if freqs is None:
        freqs = data.freqs
    
    if antpos is None:
        antpos = data.antpos
    
    vis_list = []
    weights_list = []
    uvw_list = []
    
    for ap in antpairs:
        blpol = ap + (pol,)
        blvec = antpos[ap[1]] - antpos[ap[0]]
        
        # Compute weights
        if weight_by_nsamples:
            weight = (
                nsamples[blpol][time_slice, freq_slice]
                * (~flags[blpol][time_slice, freq_slice]).astype(float)
            )
        else:
            weight = (
                np.ones_like(flags[blpol][time_slice, freq_slice])
                * (~flags[blpol][time_slice, freq_slice]).astype(float)
            )
        
        # UVW coordinates in wavelengths, shape (3, nfreqs)
        uvw_baseline = blvec[:, None] * freqs[freq_slice][None] / constants.c.value

        vis_list.append(data[blpol][time_slice, freq_slice])
        weights_list.append(weight)
        uvw_list.append(uvw_baseline)

        if include_conjugates:
            vis_list.append(data[utils.reverse_bl(blpol)][time_slice, freq_slice])
            weights_list.append(weight)
            uvw_list.append(-uvw_baseline)
    
    # Stack into arrays with shape (nbls, ntimes, nfreqs) for vis/weights
    # and (nbls, 3, nfreqs) for uvw
    vis = np.array(vis_list)
    weights = np.array(weights_list)
    uvw = np.array(uvw_list)
    times = data.times[time_slice]
    freqs_out = freqs[freq_slice]
    
    return ImagingData(
        vis=vis,
        weights=weights,
        uvw=uvw,
        times=times,
        freqs=freqs_out,
        hermitian=not include_conjugates,
    )


def unpack_uvdata(
    uvdata,
    pol: str = 'ee',
    time_slice: slice = slice(0, None),
    freq_slice: slice = slice(0, None),
    antpairs: list = None,
    weight_by_nsamples: bool = True,
) -> ImagingData:
    """
    Unpack UVData object into imaging format.
    
    Parameters
    ----------
    uvdata : UVData
        The UVData object to be unpacked.
    pol : str, optional
        The polarization to be used. Default is 'ee'.
    time_slice : slice, optional
        The time slice to be used. Default is slice(0, None).
    freq_slice : slice, optional
        The frequency slice to be used. Default is slice(0, None).
    antpairs : list[tuple], optional
        The antenna pairs to be used. If None, will use all antenna pairs.
    weight_by_nsamples : bool, optional
        Whether to weight the data by nsamples. Default is True.
    
    Returns
    -------
    ImagingData
        Structured container with visibility data ready for imaging.
    
    Raises
    ------
    NotImplementedError
        This function needs proper implementation to correctly unpack UVData.
    
    Notes
    -----
    TODO: This function needs to be properly implemented to handle UVData
    objects correctly. The current implementation is a placeholder and will
    not work correctly.
    """
    raise NotImplementedError(
        "unpack_uvdata needs proper implementation. "
        "The UVData format requires specific handling of baselines, "
        "polarizations, and coordinate transformations that are not yet implemented."
    )
    
    # Placeholder code (not functional):
    # vis = uvdata.data_array[time_slice, :, freq_slice, :]
    # weights = uvdata.nsample_array[time_slice, :, freq_slice, :] if weight_by_nsamples else np.ones_like(vis)
    # uvw = uvdata.uvw_array[time_slice, :]
    # times = np.unique(uvdata.time_array)[time_slice]
    # freqs = uvdata.freq_array[0, freq_slice]
    # 
    # return ImagingData(
    #     vis=vis,
    #     weights=weights,
    #     uvw=uvw,
    #     times=times,
    #     freqs=freqs
    # )
