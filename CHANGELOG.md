# Changelog

All notable changes to `snapshot_imager` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). While the major version is 0, minor
releases may include breaking changes; these are listed under **Breaking**.

## [Unreleased]

Imaging is 4–5× faster per channel and about 2.8× faster for MFS with the new
defaults. **Image values change slightly** (by a few parts per million, from the
new default tolerance) and **`unpack_data_containers` returns half as many rows**.
It also adds `dirty_image_points` and `radec_to_lmn` for evaluating images at
catalog positions or along the horizon, and synthesized-beam output.

### Breaking

- `unpack_data_containers` returns one row per baseline, with the conjugate
  baselines implied (`ImagingData.hermitian` is True), instead of storing each
  baseline and its conjugate. Pass `include_conjugates=True` for the previous
  layout.
- Images of Hermitian data are real-valued (float64, or float32 for `complex64`
  visibilities) instead of complex with a zero imaginary part. Images made
  with `rm_phasor` are still complex.
- The default NUFFT tolerance is now `eps=1e-6` (previously `1e-13`) in
  `dirty_image` and the `snapshot_imager_*` functions, so images change by a few
  parts per million (relative). Pass `eps=1e-13` for the previous precision.

### Added

- `dirty_image_points`: evaluate the dirty image, or the synthesized beam, at
  arbitrary directions (fixed, or changing with time such as catalog sources)
  instead of a grid, by direct summation or a Type 3 NUFFT. Returns a
  `PointsResult`. An optional w-term (`w_term="unprojected"` or `"zenith"`)
  handles non-coplanar arrays.
- `radec_to_lmn`: direction cosines (l, m, n) of RA/Dec positions at given
  times, for use with `dirty_image_points`.
- `dirty_image(..., return_psf=True)` returns the synthesized beam for every
  image (`ImageResult.psf`), computed in the same transforms as the images and
  only once when the weights don't change with time.
- `ImageResult.sum_weights`: summed weights per time and channel.
- `ImagingData.hermitian`, and `include_conjugates` in `unpack_data_containers`.
- This changelog. Pushing a version tag now also creates a GitHub Release whose
  notes are that version's section of this file.

### Fixed

- `phase_track_to_source` used |n| for sources below the horizon, mirroring
  them above it; it now uses n = sin(elevation).

### Performance

- With the new defaults, per-channel imaging is 4–5× faster and MFS about 2.8×
  faster (benchmark: 200 antennas, 10 times, 32 channels, 256×256 all-sky
  images). Hermitian data halves the NUFFT points; `eps=1e-6` shrinks the
  spreading kernel; visibilities are prepared in contiguous channel chunks; and
  images are written without transposed copies. With conjugates stored
  explicitly and `eps=1e-13`, the layout changes alone make per-channel
  imaging 10–15% faster.

## [0.3.0] - 2026-09-23

This release fixes several correctness bugs, so **images differ from 0.2.0**.
It also adds a single `dirty_image()` entry point; the existing
`snapshot_imager_*` functions keep working.

### Breaking

- Images now have the correct orientation for HERA data. The imagers used the
  wrong sign for the pyuvdata/pyuvsim convention (uvw = xyz(ant2) − xyz(ant1),
  V ∝ exp(+2πi(ul + vm + wn))), so images from 0.2.0 are rotated by 180°
  (East ↔ West, North ↔ South).
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- Per-channel images are normalized per snapshot, by that snapshot's summed
  weights (the peak of its synthesized beam), so a unit point source has peak 1
  in every snapshot. Previously every snapshot was divided by the beam peak over
  *all* times, so snapshots with more flagging came out too faint.
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- Pixels outside the visible sky (l² + m² > 1, only when `fov` > 90°) are NaN.
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- `unpack_data_containers` never includes autocorrelations; they only added a
  constant offset to images. Autocorrelations passed in `antpairs` are dropped
  with a warning. [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- Requesting the GPU when it is unavailable issues a `RuntimeWarning` instead of
  printing a message, and now also falls back to the CPU when CuPy is installed
  but no CUDA device is present.
  [#10](https://github.com/HERA-Team/snapshot_imager/pull/10)
- `rm_phasor` is validated: it must have shape `(nfreqs,)` and cannot be
  combined with MFS imaging.
  [#10](https://github.com/HERA-Team/snapshot_imager/pull/10)

### Added

- `dirty_image(data, npix=256, fov=180.0, *, mfs, method, eps, use_gpu,
  rm_phasor, verbose)`: one entry point for per-channel and MFS imaging with a
  Type 1 or Type 3 NUFFT. `snapshot_imager_type1`, `snapshot_imager_type3`,
  `snapshot_imager_mfs_type_1` and `snapshot_imager_mfs_type_3` remain, with
  their original defaults, as thin wrappers.
  [#10](https://github.com/HERA-Team/snapshot_imager/pull/10)
- `ImageResult.times` and `ImageResult.freqs`.
  [#10](https://github.com/HERA-Team/snapshot_imager/pull/10)
- Single-precision (`complex64`) imaging.
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- `benchmarks/benchmark_imagers.py` for timing the imagers on synthetic
  HERA-like data. [#10](https://github.com/HERA-Team/snapshot_imager/pull/10)

### Fixed

- Odd `npix`: pixel `npix // 2` is now exactly the phase center. Type 1 images
  were offset by half a pixel and Type 3 images were over-normalized.
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- `complex64` visibilities raised `TypeError` in every imager.
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- Division by zero for fully flagged snapshots in the Type 3 GPU path.
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)
- The `rm_phasor` docstring said the phasor was divided out; each channel's
  image is multiplied by it before summing.
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9)

### Performance

- Per-channel Type 1 imaging is about 2–3× faster: the extra NUFFT that only
  computed the normalization and a full-cube allocation per channel are gone,
  and one FINUFFT plan is reused for all channels. MFS imaging sets its uv
  points once (about 1.5× faster).
  [#9](https://github.com/HERA-Team/snapshot_imager/pull/9),
  [#10](https://github.com/HERA-Team/snapshot_imager/pull/10)

## [0.2.0] - 2026-09-23

First release on PyPI, as `snapshot-imager`.
[#8](https://github.com/HERA-Team/snapshot_imager/pull/8)

### Added

- Packaging for PyPI: versions from git tags (setuptools-scm), a `gpu` extra
  (`cupy-cuda12x`, `cufinufft`), and a `test` extra.
- Continuous integration: tests on Ubuntu and macOS for Python 3.10–3.14 and at
  the minimum supported dependency versions, linting with ruff, and build
  checks. Pushing a version tag publishes to PyPI.

### Fixed

- The MFS imagers paired each visibility with the wrong uv point whenever there
  was more than one channel (uv coordinates and visibilities were flattened in
  different orders).
- FINUFFT copied the data, and warned, on every call because the input was not
  contiguous.

### Known issues

- On Apple-silicon Macs with Python 3.10–3.12, the healpy wheels installed via
  `hera_cal` bundle their own OpenMP runtime, which conflicts with FINUFFT's and
  crashes on the first imaging call. Use Python 3.13 or newer on macOS.

[Unreleased]: https://github.com/HERA-Team/snapshot_imager/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/HERA-Team/snapshot_imager/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/HERA-Team/snapshot_imager/releases/tag/v0.2.0
