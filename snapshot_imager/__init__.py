"""
Snapshot Imager - Radio interferometry imaging using NUFFT.

This package provides efficient snapshot imaging algorithms for radio
interferometry data using Non-Uniform Fast Fourier Transforms (NUFFT).
"""

# Data models
from .data_models import ImagingData, ImageResult

# Preprocessing
# Note: unpack_uvdata is exported but raises NotImplementedError until implemented.
from .preprocessing import unpack_data_containers, unpack_uvdata

# Coordinate transformations
from .coordinates import (
    phase_track_to_source,
    compute_image_grid,
    compute_baseline_extent,
)

# Core utilities
from .core import (
    get_nufft_library,
    estimate_memory_requirements,
)

# Imaging
from .imager import (
    dirty_image,
    snapshot_imager_type1,
    snapshot_imager_type3,
    snapshot_imager_mfs_type_1,
    snapshot_imager_mfs_type_3,
)

import importlib.metadata as _metadata

try:
    # Version is set from git tags at build time by setuptools-scm.
    __version__ = _metadata.version("snapshot_imager")
except _metadata.PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"

__all__ = [
    # Data models
    "ImagingData",
    "ImageResult",
    # Preprocessing
    "unpack_data_containers",
    "unpack_uvdata",
    # Coordinates
    "phase_track_to_source",
    "compute_image_grid",
    "compute_baseline_extent",
    # Core utilities
    "get_nufft_library",
    "estimate_memory_requirements",
    # Imaging
    "dirty_image",
    "snapshot_imager_type1",
    "snapshot_imager_type3",
    "snapshot_imager_mfs_type_1",
    "snapshot_imager_mfs_type_3",
]
