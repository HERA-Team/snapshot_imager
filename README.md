# snapshot_imager

[![CI](https://github.com/HERA-Team/snapshot_imager/actions/workflows/ci.yml/badge.svg)](https://github.com/HERA-Team/snapshot_imager/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/HERA-Team/snapshot_imager/graph/badge.svg)](https://codecov.io/gh/HERA-Team/snapshot_imager)
[![PyPI](https://img.shields.io/pypi/v/snapshot-imager.svg)](https://pypi.org/project/snapshot-imager/)

`snapshot_imager` is a Python package for radio interferometric snapshot imaging using Non-Uniform Fast Fourier Transforms (NUFFT). It is designed to efficiently produce dirty image cubes from visibility data, with support for multiple NUFFT strategies (Type 1, Type 3, and multi-frequency synthesis) and optional GPU acceleration via CuPy and cuFINUFFT.

## Installation

Install the latest release from PyPI:

```bash
pip install snapshot-imager
```

For GPU support (CUDA 12), install the `gpu` extra, which pulls in `cupy-cuda12x` and `cufinufft`:

```bash
pip install "snapshot-imager[gpu]"
```

For other CUDA versions, install the matching CuPy wheel (e.g. `cupy-cuda11x` or `cupy-cuda13x`) and `cufinufft` yourself.

> **macOS note:** on Apple-silicon Macs, use Python 3.13 or newer. The healpy wheels
> for Python 3.10–3.12 (installed via `hera_cal`) bundle their own OpenMP runtime,
> which conflicts with FINUFFT's and crashes on the first imaging call.

To install from source:

```bash
git clone https://github.com/HERA-Team/snapshot_imager.git
cd snapshot_imager
pip install .
```

## Basic Usage

The typical workflow is to unpack HERA `DataContainer` objects into an `ImagingData` container, then pass that to one of the imaging functions.

```python
import numpy as np
from snapshot_imager import unpack_data_containers, snapshot_imager_type1

# data, flags, and nsamples are hera_cal DataContainer objects
imaging_data = unpack_data_containers(
    data=data,
    flags=flags,
    nsamples=nsamples,
    pol="ee",
    antpos=antpos,
    freqs=freqs,
)

# Produce a (ntimes, nfreqs, npix, npix) image cube
result = snapshot_imager_type1(
    imaging_data,
    npix=256,
    fov=10.0,       # Field of view in degrees
    use_cupy=False, # Set to True to use GPU acceleration
)

print(result.images.shape)  # (ntimes, nfreqs, npix, npix)
```

The returned `ImageResult` contains the image cube along with the corresponding `l_coords` and `m_coords` (direction cosines) for plotting or downstream analysis.

## GPU Acceleration

`snapshot_imager` supports GPU-accelerated imaging via [CuPy](https://cupy.dev/) and [cuFINUFFT](https://github.com/flatironinstitute/finufft). Simply pass `use_cupy=True` to any imaging function:

```python
result = snapshot_imager_type1(imaging_data, npix=256, fov=10.0, use_cupy=True)
```

If CuPy or cuFINUFFT are not available, the package will automatically fall back to the CPU implementation with a warning.

## Development

Set up a development environment with the test and lint tools, and install the git hooks:

```bash
pip install -e ".[dev]"
pre-commit install
```

Run the test suite (GPU tests are skipped automatically when no GPU is available):

```bash
pytest --cov
```

Lint with [ruff](https://docs.astral.sh/ruff/) (this also runs on every commit via pre-commit, and in CI):

```bash
pre-commit run --all-files
```

## Releasing

Versions are derived from git tags by [setuptools-scm](https://setuptools-scm.readthedocs.io/), so there is no version string to bump in the code. To publish a release to PyPI:

1. Make sure CI is passing on `main`.
2. Tag the release commit and push the tag:

   ```bash
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

3. The [Publish to PyPI](.github/workflows/publish.yml) workflow builds the sdist and wheel and uploads them using PyPI trusted publishing. Optionally, create a GitHub release from the tag to record release notes.

## License

MIT
