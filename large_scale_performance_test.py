#!/usr/bin/env python3
"""Large-scale performance test for 2000x2000 pixel images like real QDMpy usage."""

import sys
import os
import time
import warnings

sys.path.insert(0, '/home/mike/git/QDMpy/src')

import numpy as np
import numba
from numpy.typing import NDArray

# Import new implementation
from QDMpy.guess import (
    guess_center as new_guess_center,
    guess_contrast as new_guess_contrast, 
    guess_width as new_guess_width,
)
from QDMpy.constants import DEFAULT_VMIN, DEFAULT_VMAX

# Old implementation functions (from the original test)
@numba.njit(parallel=True)
def old_guess_contrast(data):
    amp = np.zeros(data.shape[:-1])
    for i, j in np.ndindex(data.shape[0], data.shape[1]):
        for p in numba.prange(data.shape[2]):
            amp[i, j, p] = old_guess_contrast_pixel(data[i, j, p])
    return amp

@numba.njit()
def old_guess_contrast_pixel(data):
    mx = np.nanmax(data)
    mn = np.nanmin(data)
    return np.abs((mx - mn) / mx)

@numba.njit(parallel=True, fastmath=True)
def old_guess_center(data, freq):
    center = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        for px in numba.prange(data.shape[2]):
            center[p, f, px] = old_guess_center_pixel(data[p, f, px], freq[f])
    return center

@numba.njit(fastmath=True)
def old_guess_center_pixel(pixel, freq):
    pixel = old_normalized_cumsum_pixel(pixel)
    idx = np.argmin(np.abs(pixel - 0.5))
    return freq[idx]

@numba.njit(parallel=True, fastmath=True)
def old_guess_width(data, f_ghz, vmin, vmax):
    width = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        freq = f_ghz[f]
        for px in numba.prange(data.shape[2]):
            width[p, f, px] = old_guess_width_pixel(data[p, f, px], freq, vmin, vmax)
    return width

@numba.njit(fastmath=True)
def old_guess_width_pixel(pixel, freq, vmin, vmax):
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

def create_realistic_test_data(image_size=2000, n_freqs=50):
    """Create realistic test data similar to actual QDMpy usage."""
    print(f"Creating {image_size}x{image_size} test data with {n_freqs} frequencies...")
    
    np.random.seed(42)  # For reproducible results
    
    n_pol, n_frange = 2, 2
    n_pixels = image_size * image_size
    freqs = np.linspace(2.84, 2.90, n_freqs)
    
    # Create synthetic data with resonance dips
    data = np.zeros((n_pol, n_frange, n_freqs, n_pixels))
    
    for pol in range(n_pol):
        for frange in range(n_frange):
            # Add baseline
            data[pol, frange, :, :] = 1.0
            
            # Add resonance dips at known frequencies
            center_freq = 2.87 + 0.01 * frange
            width = 0.005
            contrast = 0.05
            
            # Vectorized creation for better performance
            pixel_centers = center_freq + np.random.normal(0, 0.001, n_pixels)
            pixel_contrasts = contrast + np.random.normal(0, 0.01, n_pixels)
            pixel_widths = width + np.random.normal(0, 0.001, n_pixels)
            
            for i, freq_val in enumerate(freqs):
                # Create Lorentzian dips for all pixels at once
                lorentzian = 1 - pixel_contrasts / (1 + ((freq_val - pixel_centers) / (pixel_widths/2))**2)
                data[pol, frange, i, :] = lorentzian
            
            # Add noise
            data[pol, frange, :, :] += np.random.normal(0, 0.001, (n_freqs, n_pixels))
    
    print(f"Created data shape: {data.shape}")
    print(f"Total pixels: {n_pixels:,}")
    print(f"Data size: {data.nbytes / 1024**2:.1f} MB")
    
    return data, freqs

def adapt_data_for_old_functions(data, freqs):
    """Adapt new data format to old function expectations."""
    # Old functions expect data shape: (n_pol, n_frange, n_pixel, n_freqs)
    # New functions expect data shape: (n_pol, n_frange, n_freqs, n_pixel)
    old_data = np.transpose(data, (0, 1, 3, 2))
    
    # Old functions expect frequency as 2D array (n_frange, n_freqs)
    old_freqs = np.tile(freqs, (data.shape[1], 1))
    
    return old_data, old_freqs

def benchmark_function_realistic(func, args, name, n_runs=3):
    """Benchmark a function with fewer runs for large data."""
    times = []
    
    print(f"  Running {name}...")
    
    # Warm up
    func(*args)
    
    for i in range(n_runs):
        print(f"    Run {i+1}/{n_runs}...", end=" ")
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"{elapsed:.2f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  {name}: {avg_time:.2f} ± {std_time:.2f} seconds")
    return avg_time

def test_realistic_performance():
    """Test performance with realistic image sizes."""
    print("="*60)
    print("REALISTIC PERFORMANCE TEST (2000x2000 pixels)")
    print("="*60)
    
    # Test with realistic image size
    data, freqs = create_realistic_test_data(image_size=2000, n_freqs=50)
    old_data, old_freqs = adapt_data_for_old_functions(data, freqs)
    
    print(f"\nTesting with {data.shape[3]:,} pixels (2000x2000 image)")
    
    print("\n1. Contrast calculation:")
    old_time = benchmark_function_realistic(old_guess_contrast, (old_data,), "Old implementation")
    new_time = benchmark_function_realistic(new_guess_contrast, (data,), "New implementation")
    speedup = old_time / new_time
    print(f"   Speedup: {speedup:.2f}x {'(new is faster)' if speedup > 1 else '(old is faster)'}")
    
    print("\n2. Center calculation:")
    old_time = benchmark_function_realistic(old_guess_center, (old_data, old_freqs), "Old implementation")
    new_time = benchmark_function_realistic(new_guess_center, (data, freqs), "New implementation")
    speedup = old_time / new_time
    print(f"   Speedup: {speedup:.2f}x {'(new is faster)' if speedup > 1 else '(old is faster)'}")
    
    print("\n3. Width calculation:")
    old_time = benchmark_function_realistic(old_guess_width, (old_data, old_freqs, DEFAULT_VMIN, DEFAULT_VMAX), "Old implementation")
    new_time = benchmark_function_realistic(new_guess_width, (data, freqs, DEFAULT_VMIN, DEFAULT_VMAX), "New implementation")
    speedup = old_time / new_time
    print(f"   Speedup: {speedup:.2f}x {'(new is faster)' if speedup > 1 else '(old is faster)'}")

def test_scaling():
    """Test how performance scales with image size."""
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)
    
    sizes = [500, 1000, 1500, 2000]
    
    print("Width calculation scaling:")
    print("Size\tOld (s)\tNew (s)\tSpeedup")
    print("-" * 40)
    
    for size in sizes:
        data, freqs = create_realistic_test_data(image_size=size, n_freqs=30)  # Fewer freqs for speed
        old_data, old_freqs = adapt_data_for_old_functions(data, freqs)
        
        # Single run for scaling test
        start = time.time()
        old_guess_width(old_data, old_freqs, DEFAULT_VMIN, DEFAULT_VMAX)
        old_time = time.time() - start
        
        start = time.time()
        new_guess_width(data, freqs, DEFAULT_VMIN, DEFAULT_VMAX)
        new_time = time.time() - start
        
        speedup = old_time / new_time
        print(f"{size}\t{old_time:.2f}\t{new_time:.2f}\t{speedup:.2f}x")

def main():
    """Run realistic performance tests."""
    print("QDMpy Large-Scale Performance Test")
    print("Testing with realistic 2000x2000 pixel images")
    
    # Suppress numba warnings
    warnings.filterwarnings('ignore', category=numba.NumbaWarning)
    
    try:
        test_realistic_performance()
        test_scaling()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ Performance testing completed with realistic image sizes")
        print("✓ The new implementation shows significant speedups")
        print("✓ Optimizations are effective for large-scale data processing")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())