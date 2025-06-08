#!/usr/bin/env python3
"""
Focused Performance Analysis of QDMpy Width Calculation Functions.

This script provides a quick analysis of the key performance differences.
"""

import sys
import time
import warnings

sys.path.insert(0, '/home/mike/git/QDMpy/src')

import numpy as np
import numba
from numpy.typing import NDArray

# Import new implementation
from QDMpy.guess import guess_width as new_guess_width, normalize_pixel
from QDMpy.constants import DEFAULT_VMIN, DEFAULT_VMAX

# Old implementation (parallel)
@numba.njit(parallel=True, fastmath=True)
def old_guess_width(data: NDArray, f_ghz: NDArray, vmin: float, vmax: float) -> NDArray:
    width = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        freq = f_ghz[f]
        for px in numba.prange(data.shape[2]):  # PARALLEL
            width[p, f, px] = old_guess_width_pixel(data[p, f, px], freq, vmin, vmax)
    return width

@numba.njit(fastmath=True)
def old_guess_width_pixel(pixel: NDArray, freq: NDArray, vmin: float, vmax: float) -> float:
    pixel = old_normalized_cumsum_pixel(pixel)
    lidx = np.argmin(np.abs(pixel - vmin))
    ridx = np.argmin(np.abs(pixel - vmax))
    return freq[lidx] - freq[ridx]

@numba.njit
def old_normalized_cumsum_pixel(pixel):
    pixel = np.cumsum(pixel - 1)
    pixel -= np.min(pixel)
    pixel /= np.max(pixel)
    return pixel

# Optimized new implementation (with parallel=True added back)
@numba.njit(parallel=True, fastmath=True)
def optimized_guess_width(data: NDArray, freq: NDArray, vmin: float, vmax: float) -> NDArray:
    widths = np.zeros((data.shape[0], data.shape[1], data.shape[3]))
    for p in range(data.shape[0]):
        for r in range(data.shape[1]):
            for px in numba.prange(data.shape[3]):  # PARALLEL
                widths[p, r, px] = optimized_guess_width_pixel(data[p, r, :, px], freq, vmin, vmax)
    return widths

@numba.njit(fastmath=True)
def optimized_guess_width_pixel(pixel: NDArray, freq: NDArray, vmin: float, vmax: float) -> float:
    normalized = normalize_pixel(pixel)
    lidx = np.argmin(np.abs(normalized - vmin))
    ridx = np.argmin(np.abs(normalized - vmax))
    return abs(freq[ridx] - freq[lidx])


def create_test_data(n_pixels: int) -> tuple:
    """Create test data with specified number of pixels."""
    np.random.seed(42)
    n_pol, n_frange, n_freqs = 2, 2, 50
    freqs = np.linspace(2.84, 2.90, n_freqs)
    
    data = np.ones((n_pol, n_frange, n_freqs, n_pixels))
    
    # Add simple resonance dips
    for pol in range(n_pol):
        for frange in range(n_frange):
            center_freq = 2.87 + 0.01 * frange
            for pixel in range(n_pixels):
                # Simple Lorentzian
                lorentzian = 1 - 0.05 / (1 + ((freqs - center_freq) / 0.005)**2)
                data[pol, frange, :, pixel] = lorentzian
    
    return data, freqs


def adapt_data_for_old(data, freqs):
    """Adapt data format for old functions."""
    old_data = np.transpose(data, (0, 1, 3, 2))
    old_freqs = np.tile(freqs, (data.shape[1], 1))
    return old_data, old_freqs


def benchmark(func, args, name, n_runs=3):
    """Quick benchmark."""
    # Warm up
    func(*args)
    
    times = []
    for _ in range(n_runs):
        start = time.time()
        func(*args)
        times.append(time.time() - start)
    
    avg = np.mean(times)
    print(f"   {name}: {avg:.4f} seconds")
    return avg


def main():
    """Run focused analysis."""
    warnings.filterwarnings('ignore')
    
    print("QDMpy Width Calculation - Focused Performance Analysis")
    print("="*70)
    
    print("\nKEY DIFFERENCES:")
    print("1. Old: @numba.njit(parallel=True, fastmath=True) + numba.prange()")
    print("2. New: @numba.njit(fastmath=True) + regular range()")
    print("3. Optimized: @numba.njit(parallel=True, fastmath=True) + numba.prange()")
    
    # Test different sizes
    test_sizes = [
        ("Small", 1000),
        ("Medium", 10000),
        ("Large", 100000),
    ]
    
    for size_name, n_pixels in test_sizes:
        print(f"\n{size_name} Dataset ({n_pixels:,} pixels):")
        
        data, freqs = create_test_data(n_pixels)
        old_data, old_freqs = adapt_data_for_old(data, freqs)
        
        old_time = benchmark(old_guess_width, (old_data, old_freqs, DEFAULT_VMIN, DEFAULT_VMAX), "Old (parallel)")
        new_time = benchmark(new_guess_width, (data, freqs, DEFAULT_VMIN, DEFAULT_VMAX), "New (sequential)")
        opt_time = benchmark(optimized_guess_width, (data, freqs, DEFAULT_VMIN, DEFAULT_VMAX), "Optimized (parallel)")
        
        print(f"   Old vs New speedup: {old_time/new_time:.1f}x")
        print(f"   New vs Optimized speedup: {new_time/opt_time:.1f}x")
    
    print("\n" + "="*70)
    print("SCALING ESTIMATE FOR 2000x2000 PIXELS (4M pixels):")
    
    # Extrapolate to 4M pixels based on 100k results
    if len(test_sizes) >= 3:
        data, freqs = create_test_data(100000)
        old_data, old_freqs = adapt_data_for_old(data, freqs)
        
        old_100k = benchmark(old_guess_width, (old_data, old_freqs, DEFAULT_VMIN, DEFAULT_VMAX), "Old baseline", 1)
        new_100k = benchmark(new_guess_width, (data, freqs, DEFAULT_VMIN, DEFAULT_VMAX), "New baseline", 1)
        
        # Linear scaling assumption (reasonable for this type of operation)
        scale_factor = 4000000 / 100000  # 40x more pixels
        
        old_4m = old_100k * scale_factor
        new_4m = new_100k * scale_factor
        
        print(f"   Estimated Old (4M pixels): {old_4m:.1f} seconds")
        print(f"   Estimated New (4M pixels): {new_4m:.1f} seconds")
        print(f"   Performance difference: {new_4m/old_4m:.1f}x slower")
        print(f"   Time difference: +{new_4m-old_4m:.1f} seconds per calculation")
    
    print("\n" + "="*70)
    print("RECOMMENDED FIXES:")
    print("1. Add 'parallel=True' to @numba.njit decorators in guess.py")
    print("2. Change range() to numba.prange() for pixel loops")
    print("3. Apply same fixes to guess_center and guess_contrast functions")
    print("\nFILES TO MODIFY:")
    print("- /home/mike/git/QDMpy/src/QDMpy/guess.py (lines 275, 236, 196)")


if __name__ == "__main__":
    main()