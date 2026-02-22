"""Profile the QDMpy pipeline from data loading to fitting.

Usage:
    uv run python scripts/profile_pipeline.py [data_folder] [--bin FACTOR] [--mode {cprofile,line,time,guess}]

Modes:
    time     (default) Simple wall-clock timing per stage
    cprofile  cProfile + pstats, prints top hotspots
    line      line_profiler on key functions (requires `uv add line-profiler`)
    guess     Micro-benchmark each guess function after JIT warm-up
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def timed(label: str, timings: dict[str, float]) -> Generator[None]:
    t0 = time.perf_counter()
    yield
    timings[label] = time.perf_counter() - t0


def print_timings(timings: dict[str, float]) -> None:
    total = sum(timings.values())
    for _label, elapsed in timings.items():
        100 * elapsed / total if total > 0 else 0


# ---------------------------------------------------------------------------
# Pipeline stages (each returns what the next stage needs)
# ---------------------------------------------------------------------------


def stage_load(data_folder: Path) -> tuple:
    """Load raw ODMR data from disk."""
    from qdmpy_core.odmr.data import ODMRData
    from qdmpy_core.odmr.io import MatlabLoader

    loader = MatlabLoader(data_folder=str(data_folder))
    return ODMRData.from_loader(loader=loader)


def stage_build_odmr(odmr_data, bin_factor: int):
    """Construct the ODMR manager (no processing yet)."""
    from qdmpy_core.odmr.manager import ODMR
    from qdmpy_core.odmr.processors import BinningProcessor, NormalizationProcessor

    odmr = ODMR(odmr_data)
    if bin_factor > 1:
        odmr.processor_manager.add_processor(BinningProcessor(bin_factor=bin_factor))
    odmr.processor_manager.add_processor(NormalizationProcessor())
    return odmr


def stage_process(odmr) -> None:
    """Run the processor chain (binning + normalisation)."""
    odmr.process_data()


def stage_fit(odmr, model_name: str = "auto"):
    """Build FitManager and run GPU fitting."""
    from qdmpy_core.fitting.manager import FitManager

    processed = odmr.processed_data
    fm = FitManager(
        data=processed.data,
        frequencies=processed.frequencies,
        model_name=model_name,
    )
    fm.fit_odmr()
    return fm


# ---------------------------------------------------------------------------
# Profiling modes
# ---------------------------------------------------------------------------


def run_timed(data_folder: Path, bin_factor: int, model_name: str) -> None:
    timings: dict[str, float] = {}

    with timed("1. load (MatlabLoader → ODMRData)", timings):
        odmr_data = stage_load(data_folder)

    with timed("2. build ODMR + add processors", timings):
        odmr = stage_build_odmr(odmr_data, bin_factor)

    with timed("3. process (bin + normalise)", timings):
        stage_process(odmr)

    with timed("4. fit (FitManager.fit_odmr)", timings):
        stage_fit(odmr, model_name)

    print_timings(timings)


def run_cprofile(data_folder: Path, bin_factor: int, model_name: str) -> None:
    pr = cProfile.Profile()
    pr.enable()

    odmr_data = stage_load(data_folder)
    odmr = stage_build_odmr(odmr_data, bin_factor)
    stage_process(odmr)
    stage_fit(odmr, model_name)

    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)


def run_guess_bench(data_folder: Path, bin_factor: int) -> None:
    """Micro-benchmark each guess function with proper JIT warm-up."""
    from qdmpy_core.constants import DEFAULT_VMAX, DEFAULT_VMIN
    from qdmpy_core.fitting import guess as guess_mod

    odmr_data = stage_load(data_folder)
    odmr = stage_build_odmr(odmr_data, bin_factor)
    stage_process(odmr)

    processed = odmr.processed_data
    data = processed.data.values  # (pol, frange, y, x, freq)
    n_pol, n_frange, h, w, n_freq = data.shape
    flat = data.reshape(n_pol, n_frange, h * w, n_freq)
    freq = processed.frequencies  # (n_frange, n_freq)

    for _ in range(3):
        _ = guess_mod.cumsum_contrast(flat)
        _ = guess_mod.cumsum_center(flat, freq)
        _ = guess_mod.cumsum_width(flat, freq, DEFAULT_VMIN, DEFAULT_VMAX)

    N_RUNS = 10
    timings: dict[str, list[float]] = {"contrast": [], "center": [], "width": []}

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        guess_mod.cumsum_contrast(flat)
        timings["contrast"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        guess_mod.cumsum_center(flat, freq)
        timings["center"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        guess_mod.cumsum_width(flat, freq, DEFAULT_VMIN, DEFAULT_VMAX)
        timings["width"].append(time.perf_counter() - t0)

    for _name, _times in timings.items():
        pass

    sum(np.mean(v) for v in timings.values())
    max(timings, key=lambda k: np.mean(timings[k]))


def run_line_profiler(data_folder: Path, bin_factor: int, model_name: str) -> None:
    try:
        from line_profiler import LineProfiler
    except ImportError:
        sys.exit(1)

    from qdmpy_core.fitting import guess
    from qdmpy_core.fitting.manager import FitManager
    from qdmpy_core.odmr.data import ODMRData
    from qdmpy_core.odmr.io import MatlabLoader
    from qdmpy_core.odmr.manager import ODMR

    lp = LineProfiler()

    # Profile key internals
    lp.add_function(MatlabLoader.load)
    lp.add_function(ODMRData.from_loader)
    lp.add_function(ODMR.process_data)
    lp.add_function(FitManager.fit_odmr)
    lp.add_function(FitManager.fit_frange)
    lp.add_function(guess.cumsum_center)
    lp.add_function(guess.cumsum_contrast)
    lp.add_function(guess.cumsum_width)

    @lp
    def full_pipeline() -> None:
        odmr_data = stage_load(data_folder)
        odmr = stage_build_odmr(odmr_data, bin_factor)
        stage_process(odmr)
        stage_fit(odmr, model_name)

    full_pipeline()
    lp.print_stats()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile QDMpy read-to-fit pipeline")
    p.add_argument(
        "data_folder",
        nargs="?",
        default=str(Path.home() / "Documents" / "FOV18x"),
        help="Path to folder containing run_*.mat files (default: ~/Documents/FOV18x)",
    )
    p.add_argument(
        "--bin", type=int, default=6, metavar="FACTOR", help="Binning factor (default: 6)"
    )
    p.add_argument("--model", default="auto", help="Model name (default: auto)")
    p.add_argument(
        "--mode",
        choices=["time", "cprofile", "line", "guess"],
        default="time",
        help="Profiling mode (default: time)",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress loguru output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quiet:
        logger.disable("QDMpy")
    else:
        logger.enable("QDMpy")

    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        sys.exit(1)

    if args.mode == "time":
        run_timed(data_folder, args.bin, args.model)
    elif args.mode == "cprofile":
        run_cprofile(data_folder, args.bin, args.model)
    elif args.mode == "line":
        run_line_profiler(data_folder, args.bin, args.model)
    elif args.mode == "guess":
        run_guess_bench(data_folder, args.bin)


if __name__ == "__main__":
    main()
