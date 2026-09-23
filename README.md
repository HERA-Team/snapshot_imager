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

The typical workflow is to unpack HERA `DataContainer` objects into an `ImagingData` container, then pass that to `dirty_image`.

```python
from snapshot_imager import unpack_data_containers, dirty_image

# data, flags, and nsamples are hera_cal DataContainer objects
imaging_data = unpack_data_containers(
    data=data,
    flags=flags,
    nsamples=nsamples,
    pol="ee",
    antpos=antpos,
    freqs=freqs,
)

# One image per channel: a (ntimes, nfreqs, npix, npix) cube
result = dirty_image(imaging_data, npix=256, fov=10.0)

# One multi-frequency synthesis (MFS) image per time: (ntimes, 1, npix, npix)
mfs = dirty_image(imaging_data, npix=256, fov=10.0, mfs=True)

print(result.images.shape)
```

The returned `ImageResult` holds the image cube (indexed `images[time, freq, m, l]`), the pixel direction cosines `l_coords` and `m_coords`, and the `times` and `freqs` of the images. Per-channel images are normalized so a unit point source has peak 1; MFS images are the unnormalized weighted sum. Pixels below the horizon (only when `fov` > 90°) are NaN.

`unpack_data_containers` stores one row per baseline and marks the data as Hermitian (`imaging_data.hermitian`): each baseline's conjugate is implied rather than stored, which halves the memory and imaging work, and the images are real-valued. Pass `include_conjugates=True` to store the conjugate baselines explicitly instead.

The NUFFT tolerance defaults to `eps=1e-6` (the relative error of the image), far below the noise in any dirty image; pass a smaller `eps` (down to about `1e-14`) for validation.

`dirty_image` uses a Type 1 NUFFT by default; pass `method="type3"` to evaluate the same image with a Type 3 NUFFT (much slower on a regular grid, mainly useful for validation).

The earlier functions `snapshot_imager_type1`, `snapshot_imager_type3`, `snapshot_imager_mfs_type_1`, and `snapshot_imager_mfs_type_3` are still available, with their original defaults and outputs; they are thin wrappers around `dirty_image`.

## Synthesized Beam and Catalog Positions

Pass `return_psf=True` to also get the synthesized beam (point spread function) for every image, normalized like the images (a unit point source at the phase center). It is computed in the same transforms as the images, and only once when the weights don't change with time. `ImageResult.sum_weights` always holds the summed weights per time and channel.

```python
result = dirty_image(imaging_data, npix=256, fov=180.0, return_psf=True)
result.psf.shape  # same as result.images
```

`dirty_image_points` evaluates the same image, or the synthesized beam, at arbitrary directions instead of a grid, for example at catalog sources as they drift through the beam. `radec_to_lmn` gives the direction cosines of sky positions at each time:

```python
from snapshot_imager import radec_to_lmn, dirty_image_points

# l, m, n have shape (ntimes, nsources)
l, m, n = radec_to_lmn(ra_deg, dec_deg, imaging_data.times, telescope_location)

# Image values at each source, fluxes.values: (ntimes, nfreqs, nsources)
fluxes = dirty_image_points(imaging_data, l, m, n)

# Synthesized beam at the offsets of every source from source 0
beam = dirty_image_points(imaging_data, l - l[:, :1], m - m[:, :1], psf=True)
```

It evaluates the sum directly for small problems (up to about 3 million baseline–direction pairs per channel, e.g. ~150 directions for 200 antennas) and switches to a Type 3 NUFFT for larger ones (`method="direct"` or `"type3"` to choose). `w_term="unprojected"` (raw drift-scan data, where V ∝ exp(+2πi(ul + vm + wn))) or `w_term="zenith"` (data with the zenith w-phase removed, e.g. absorbed by calibration) includes the w-term for non-coplanar arrays; it is only meaningful when each row has a well-defined w, which is not exactly true for redundantly averaged data.

## GPU Acceleration

`snapshot_imager` supports GPU-accelerated imaging via [CuPy](https://cupy.dev/) and [cuFINUFFT](https://github.com/flatironinstitute/finufft). Pass `use_gpu=True`:

```python
result = dirty_image(imaging_data, npix=256, fov=10.0, use_gpu=True)
```

If CuPy or cuFINUFFT are not installed, or no CUDA device is available, it falls back to the CPU with a warning.

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

Benchmark the imagers on synthetic HERA-like data (see `--help` for sizes and options):

```bash
python benchmarks/benchmark_imagers.py
```

## Changelog

User-facing changes are recorded in [CHANGELOG.md](CHANGELOG.md). When a pull request changes behavior, adds a feature, or fixes a bug, add an entry under **Unreleased** (use a **Breaking** subsection for changes that alter results or the API).

## Releasing

Versions are derived from git tags by [setuptools-scm](https://setuptools-scm.readthedocs.io/), so there is no version string to bump in the code. To publish a release:

1. In `CHANGELOG.md`, rename **Unreleased** to the new version and date (e.g. `## [0.4.0] - 2026-10-01`), add a fresh empty **Unreleased** section above it, and update the comparison links at the bottom. Merge this to `main`.
2. Make sure CI is passing on `main`.
3. Tag the release commit and push the tag:

   ```bash
   git tag -a v0.4.0 -m "v0.4.0"
   git push origin v0.4.0
   ```

4. The [Publish to PyPI](.github/workflows/publish.yml) workflow builds the sdist and wheel, uploads them to PyPI using trusted publishing, and creates a GitHub Release whose notes are that version's section of `CHANGELOG.md`. The workflow stops before publishing if the changelog has no section for the tag.

## License

MIT
