"""
Benchmark dirty_image on synthetic HERA-like data.

Examples
--------
Default size (200 antennas, 10 times, 32 channels, 256x256 all-sky images)::

    python benchmarks/benchmark_imagers.py

Smaller run, including Type 3 and the GPU::

    python benchmarks/benchmark_imagers.py --nants 60 --nfreqs 8 --type3 --gpu

Timings are the best of ``--repeat`` runs. Compare numbers only between runs
on the same machine.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

import snapshot_imager
from snapshot_imager import ImagingData, dirty_image, get_nufft_library

C = 299792458.0  # speed of light [m/s]


def synthetic_data(nants, ntimes, nfreqs, seed=0, dtype=np.complex128):
    """
    Random coplanar array of ``nants`` antennas within 150 m, with every
    cross-correlation and its conjugate (as unpack_data_containers returns).
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-150.0, 150.0, (nants, 3))
    pos[:, 2] = 0.0
    a1, a2 = np.triu_indices(nants, 1)
    freqs = np.linspace(100e6, 200e6, nfreqs)
    uvw = (pos[a2] - pos[a1])[:, :, None] * freqs[None, None, :] / C
    uvw = np.concatenate([uvw, -uvw])

    nbls = len(a1)
    vis = rng.standard_normal((nbls, ntimes, nfreqs)) + 1j * rng.standard_normal(
        (nbls, ntimes, nfreqs)
    )
    vis = np.concatenate([vis, np.conj(vis)]).astype(dtype)
    weights = np.ones(vis.shape)
    times = 2459000.0 + np.arange(ntimes) / 1440.0
    return ImagingData(vis, weights, uvw, times, freqs)


def best_time(func, repeat):
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nants", type=int, default=200)
    parser.add_argument("--ntimes", type=int, default=10)
    parser.add_argument("--nfreqs", type=int, default=32)
    parser.add_argument("--npix", type=int, default=256)
    parser.add_argument("--fov", type=float, default=180.0)
    parser.add_argument("--eps", type=float, default=1e-13)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--single", action="store_true", help="complex64 data")
    parser.add_argument("--type3", action="store_true", help="also time Type 3")
    parser.add_argument("--gpu", action="store_true", help="also time the GPU")
    args = parser.parse_args(argv)

    dtype = np.complex64 if args.single else np.complex128
    data = synthetic_data(args.nants, args.ntimes, args.nfreqs, dtype=dtype)
    print(
        f"snapshot_imager {snapshot_imager.__version__}: {data.nbls} baselines "
        f"(incl. conjugates), {data.ntimes} times, {data.nfreqs} channels, "
        f"npix={args.npix}, fov={args.fov}, eps={args.eps:g}, {np.dtype(dtype).name}, "
        f"visibilities {data.vis.nbytes / 1e6:.0f} MB"
    )

    methods = ["type1", "type3"] if args.type3 else ["type1"]
    devices = [False]
    if args.gpu:
        # Only time the GPU if it is really used (dirty_image would otherwise
        # fall back to the CPU and the "gpu" rows would be mislabeled).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, gpu_available = get_nufft_library(use_cupy=True)
        if gpu_available:
            devices.append(True)
        else:
            reason = str(caught[0].message) if caught else "GPU unavailable"
            print(f"Skipping GPU timings: {reason}")
    print(f"{'config':32s} {'total [s]':>10s} {'per channel [ms]':>17s}")
    for use_gpu in devices:
        for method in methods:
            for mfs in (False, True):
                name = f"{method} {'mfs' if mfs else 'per-channel'} {'gpu' if use_gpu else 'cpu'}"

                def run(method=method, mfs=mfs, use_gpu=use_gpu):
                    dirty_image(
                        data,
                        args.npix,
                        args.fov,
                        mfs=mfs,
                        method=method,
                        eps=args.eps,
                        use_gpu=use_gpu,
                    )

                seconds = best_time(run, args.repeat)
                print(f"{name:32s} {seconds:10.3f} {seconds / data.nfreqs * 1e3:17.1f}")


if __name__ == "__main__":
    main()
