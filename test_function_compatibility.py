#!/usr/bin/env python3
"""Test compatibility between old and new QDMpy fit implementations.

This test compares the calculation functions from the old fit module with the new
implementation to ensure they produce the same results. It also includes basic
performance benchmarks.
"""

import sys
import os
import time
import warnings
from typing import Tuple, Any, Dict, List, Union, Optional, Callable

sys.path.insert(0, '/home/mike/git/QDMpy/src')

import numpy as np
import numba
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import NDArray

# Import new implementation
from QDMpy.guess import (
    guess_center as new_guess_center,
    guess_contrast as new_guess_contrast, 
    guess_width as new_guess_width,
    normalize_pixel as new_normalize_pixel,
    guess_center_pixel, guess_contrast_pixel, guess_width_pixel
)
from QDMpy.constants import DEFAULT_VMIN, DEFAULT_VMAX
from QDMpy.models import esr14n, esr15n, esrsingle, ModelRegistry

print("Testing compatibility between old and new fitting functions...")

# ===== OLD IMPLEMENTATION FUNCTIONS (copied from your provided code) =====

@numba.njit(parallel=True)
def old_guess_contrast(data):
    """
    Guess the contrast of a ODMR data.

    :param data: np.array
        data to guess the contrast from
    :return: np.array
        contrast of the data
    """
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
    """
    Guess the center frequency of ODMR data.

    :param data: np.array
        data to guess the center frequency from
    :param freq: np.array
        frequency range of the data

    :return: np.array
        center frequency of the data
    """
    # center frequency
    center = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        for px in numba.prange(data.shape[2]):
            center[p, f, px] = old_guess_center_pixel(data[p, f, px], freq[f])
    return center


@numba.njit(fastmath=True)
def old_guess_center_pixel(pixel, freq):
    """
    Guess the center frequency of a single frequency range.

    :param data: np.array
        data to guess the center frequency from
    :param freq: np.array
        frequency range of the data
    :return: np.array
        center frequency of the data
    """
    pixel = old_normalized_cumsum_pixel(pixel)
    idx = np.argmin(np.abs(pixel - 0.5))
    return freq[idx]


@numba.njit(parallel=True, fastmath=True)
def old_guess_width(data: NDArray, f_ghz: NDArray, vmin: float, vmax: float) -> NDArray:
    """
    Guess the width of a ODMR resonance peaks.

    :param data: np.array
        data to guess the width from
    :param f_ghz: np.array
        frequency range of the data
    :param vmin: float
        minimum value of normalized cumsum to be considered
    :param vmax: float
        maximum value of normalized cumsum to be considered

    :return: np.array
        width of the data
    """
    # width
    width = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        freq = f_ghz[f]
        for px in numba.prange(data.shape[2]):
            width[p, f, px] = old_guess_width_pixel(data[p, f, px], freq, vmin, vmax)

    return width


@numba.njit(fastmath=True)
def old_guess_width_pixel(
    pixel: NDArray, freq: NDArray, vmin: float, vmax: float
) -> NDArray:
    """
    Guess the width of a single frequency range.

    :param data: np.array
        data to guess the width from
    :param freq: np.array
        frequency range of the data

    :return: np.array
        width of the data

    Raises ValueError if the number of peaks is not 1, 2 or 3.
    """
    pixel = old_normalized_cumsum_pixel(pixel)
    lidx = np.argmin(np.abs(pixel - vmin))
    ridx = np.argmin(np.abs(pixel - vmax))
    return freq[lidx] - freq[ridx]


@numba.njit
def old_normalized_cumsum_pixel(pixel):
    """Calculate the normalized cumulative sum of the data.

    Parameters
    ----------
    data : NDArray
        Data to calculate the normalized cumulative sum of.


    Returns
    -------
    NDArray
        Normalized cumulative sum of the data.
    """
    pixel = np.cumsum(pixel - 1)
    pixel -= np.min(pixel)
    pixel /= np.max(pixel)
    return pixel

# ===== END OLD IMPLEMENTATION =====

def create_test_data() -> Tuple[NDArray, NDArray]:
    """Create synthetic test data similar to ODMR data."""
    # Create synthetic ODMR data with known parameters
    np.random.seed(42)  # For reproducible results
    
    n_pol, n_frange, n_freqs, n_pixel = 2, 2, 50, 100
    freqs = np.linspace(2.84, 2.90, n_freqs)
    
    # Create synthetic data with resonance dips
    data = np.zeros((n_pol, n_frange, n_freqs, n_pixel))
    
    for pol in range(n_pol):
        for frange in range(n_frange):
            # Add baseline
            data[pol, frange, :, :] = 1.0
            
            # Add resonance dips at known frequencies
            center_freq = 2.87 + 0.01 * frange
            width = 0.005
            contrast = 0.05
            
            for pixel in range(n_pixel):
                # Add slight variations per pixel
                pixel_center = center_freq + np.random.normal(0, 0.001)
                pixel_contrast = contrast + np.random.normal(0, 0.01)
                pixel_width = width + np.random.normal(0, 0.001)
                
                # Create Lorentzian dip
                lorentzian = 1 - pixel_contrast / (1 + ((freqs - pixel_center) / (pixel_width/2))**2)
                data[pol, frange, :, pixel] = lorentzian
            
            # Add noise
            data[pol, frange, :, :] += np.random.normal(0, 0.001, (n_freqs, n_pixel))
    
    return data, freqs


def adapt_data_for_old_functions(data: NDArray, freqs: NDArray) -> Tuple[NDArray, NDArray]:
    """Adapt new data format to old function expectations."""
    # Old functions expect data shape: (n_pol, n_frange, n_pixel, n_freqs)
    # New functions expect data shape: (n_pol, n_frange, n_freqs, n_pixel)
    old_data = np.transpose(data, (0, 1, 3, 2))
    
    # Old functions expect frequency as 2D array (n_frange, n_freqs)
    old_freqs = np.tile(freqs, (data.shape[1], 1))
    
    return old_data, old_freqs


def test_normalize_pixel_compatibility():
    """Test that old and new normalize pixel functions give the same results."""
    print("\nTesting normalize pixel compatibility...")
    
    # Create a single pixel test
    np.random.seed(42)
    pixel = np.random.randn(50) + 1.0
    
    # Run old function
    old_result = old_normalized_cumsum_pixel(pixel)
    
    # Run new function
    new_result = new_normalize_pixel(pixel)
    
    # Compare results
    are_close = np.allclose(old_result, new_result, rtol=1e-10, atol=1e-12)
    print(f"   Results are close: {are_close}")
    
    if not are_close:
        diff = np.abs(old_result - new_result)
        print(f"   Max difference: {np.max(diff)}")
        print(f"   Mean difference: {np.mean(diff)}")
        print(f"   Old result sample: {old_result[:5]}")
        print(f"   New result sample: {new_result[:5]}")
    
    return are_close


def test_contrast_compatibility():
    """Test that old and new contrast functions give the same results."""
    print("\nTesting contrast calculation compatibility...")
    
    data, freqs = create_test_data()
    old_data, old_freqs = adapt_data_for_old_functions(data, freqs)
    
    # Run old function
    old_result = old_guess_contrast(old_data)
    
    # Run new function  
    new_result = new_guess_contrast(data)
    
    # Compare results
    print(f"   Old result shape: {old_result.shape}")
    print(f"   New result shape: {new_result.shape}")
    
    # Check if results are close (within numerical precision)
    are_close = np.allclose(old_result, new_result, rtol=1e-10, atol=1e-12)
    print(f"   Results are close: {are_close}")
    
    if not are_close:
        diff = np.abs(old_result - new_result)
        print(f"   Max difference: {np.max(diff)}")
        print(f"   Mean difference: {np.mean(diff)}")
        print(f"   Old result sample: {old_result[0, 0, :5]}")
        print(f"   New result sample: {new_result[0, 0, :5]}")
    
    return are_close


def test_center_compatibility():
    """Test that old and new center functions give the same results."""
    print("\nTesting center calculation compatibility...")
    
    data, freqs = create_test_data()
    old_data, old_freqs = adapt_data_for_old_functions(data, freqs)
    
    # Run old function
    old_result = old_guess_center(old_data, old_freqs)
    
    # Run new function
    new_result = new_guess_center(data, freqs)
    
    # Compare results
    print(f"   Old result shape: {old_result.shape}")
    print(f"   New result shape: {new_result.shape}")
    
    # Check if results are close
    are_close = np.allclose(old_result, new_result, rtol=1e-10, atol=1e-12)
    print(f"   Results are close: {are_close}")
    
    if not are_close:
        diff = np.abs(old_result - new_result)
        print(f"   Max difference: {np.max(diff)}")
        print(f"   Mean difference: {np.mean(diff)}")
        print(f"   Old result sample: {old_result[0, 0, :5]}")
        print(f"   New result sample: {new_result[0, 0, :5]}")
    
    return are_close


def test_width_compatibility():
    """Test that old and new width functions give the same results."""
    print("\nTesting width calculation compatibility...")
    
    data, freqs = create_test_data()
    old_data, old_freqs = adapt_data_for_old_functions(data, freqs)
    
    vmin, vmax = DEFAULT_VMIN, DEFAULT_VMAX
    
    # Run old function
    old_result = old_guess_width(old_data, old_freqs, vmin, vmax)
    
    # Run new function
    new_result = new_guess_width(data, freqs, vmin, vmax)
    
    # Compare results
    print(f"   Old result shape: {old_result.shape}")
    print(f"   New result shape: {new_result.shape}")
    
    # Check if results are close
    are_close = np.allclose(old_result, new_result, rtol=1e-10, atol=1e-12)
    print(f"   Results are close: {are_close}")
    
    if not are_close:
        diff = np.abs(old_result - new_result)
        print(f"   Max difference: {np.max(diff)}")
        print(f"   Mean difference: {np.mean(diff)}")
        print(f"   Old result sample: {old_result[0, 0, :5]}")
        print(f"   New result sample: {new_result[0, 0, :5]}")
    
    return are_close


def benchmark_function(func: Callable, args: tuple, name: str, n_runs: int = 10) -> float:
    """Benchmark a function and return average execution time."""
    times = []
    
    # Warm up
    func(*args)
    
    for _ in range(n_runs):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"   {name}: {avg_time:.6f} ± {std_time:.6f} seconds")
    return avg_time


def performance_comparison():
    """Compare performance of old vs new implementations."""
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    data, freqs = create_test_data()
    old_data, old_freqs = adapt_data_for_old_functions(data, freqs)
    
    print("\nContrast calculation:")
    old_time = benchmark_function(old_guess_contrast, (old_data,), "Old implementation")
    new_time = benchmark_function(new_guess_contrast, (data,), "New implementation")
    speedup = old_time / new_time
    print(f"   Speedup: {speedup:.2f}x {'(new is faster)' if speedup > 1 else '(old is faster)'}")
    
    print("\nCenter calculation:")
    old_time = benchmark_function(old_guess_center, (old_data, old_freqs), "Old implementation")
    new_time = benchmark_function(new_guess_center, (data, freqs), "New implementation")
    speedup = old_time / new_time
    print(f"   Speedup: {speedup:.2f}x {'(new is faster)' if speedup > 1 else '(old is faster)'}")
    
    print("\nWidth calculation:")
    old_time = benchmark_function(old_guess_width, (old_data, old_freqs, DEFAULT_VMIN, DEFAULT_VMAX), "Old implementation")
    new_time = benchmark_function(new_guess_width, (data, freqs, DEFAULT_VMIN, DEFAULT_VMAX), "New implementation")
    speedup = old_time / new_time
    print(f"   Speedup: {speedup:.2f}x {'(new is faster)' if speedup > 1 else '(old is faster)'}")


def create_old_test_data():
    """Create test data similar to old make_dummy_data function."""
    # Create frequency ranges
    f0 = np.linspace(2.84, 2.85, 50)
    f1 = np.linspace(2.89, 2.9, 50)
    frequencies = np.array([f0, f1])  # Shape: (2, 50)
    
    # Create scan dimensions
    scan_dims = np.array([12, 19])  # Small test size
    n_pixels = scan_dims[0] * scan_dims[1]
    
    # Model parameters
    center0, center1 = np.mean(f0), np.mean(f1)
    width = 0.001
    contrast = 0.05
    offset = 0.0
    
    # Create parameter arrays for 15N model (2 peaks)
    params_15n = np.array([
        center0, center1,  # centers
        contrast, contrast,  # contrasts  
        width, width,  # widths
        offset  # offset
    ])
    
    # Generate data using ESR15N model
    data = np.zeros((2, 2, 50, n_pixels))  # (n_pol, n_frange, n_freq, n_pixels)
    
    for pol in range(2):
        for frange in range(2):
            freq = frequencies[frange]
            # Add some spatial variation
            for pixel in range(n_pixels):
                # Slight variation in parameters across pixels
                noise_factor = 1 + 0.1 * np.random.normal()
                pixel_params = params_15n.copy()
                pixel_params[:2] *= noise_factor  # Vary centers slightly
                
                # Generate spectrum using ESR15N model
                spectrum = esr15n(freq, pixel_params)
                
                # Add noise
                spectrum += 0.01 * np.random.normal(size=spectrum.shape)
                
                data[pol, frange, :, pixel] = spectrum
    
    return data, frequencies, scan_dims

def test_normalize_pixel():
    """Test normalize_pixel function."""
    print("\n1. Testing normalize_pixel function...")
    
    # Create test pixel data
    pixel = np.array([1.0, 0.9, 0.8, 0.85, 0.95, 1.0])
    
    # Test new function
    normalized = normalize_pixel(pixel)
    
    print(f"   Input pixel: {pixel}")
    print(f"   Normalized: {normalized}")
    print(f"   Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Check properties
    assert normalized.min() >= 0, "Normalized data should be non-negative"
    assert normalized.max() <= 1, "Normalized data should be <= 1"
    print("   ✓ normalize_pixel passed basic tests")

def test_guess_functions():
    """Test all guess functions with synthetic data."""
    print("\n2. Testing guess functions...")
    
    # Create test data
    data, frequencies, scan_dims = create_test_data()
    print(f"   Test data shape: {data.shape}")
    print(f"   Frequencies shape: {frequencies.shape}")
    
    # Test guess_center
    print("\n   Testing guess_center...")
    centers = guess_center(data, frequencies[0])  # Use first frequency range
    print(f"   Centers shape: {centers.shape}")
    print(f"   Center range: [{centers.min():.4f}, {centers.max():.4f}] GHz")
    expected_center = np.mean(frequencies[0])
    print(f"   Expected center: {expected_center:.4f} GHz")
    print(f"   Mean guessed center: {centers.mean():.4f} GHz")
    assert abs(centers.mean() - expected_center) < 0.01, f"Center guess too far off: {centers.mean():.4f} vs {expected_center:.4f}"
    print("   ✓ guess_center passed")
    
    # Test guess_contrast
    print("\n   Testing guess_contrast...")
    contrasts = guess_contrast(data)
    print(f"   Contrasts shape: {contrasts.shape}")
    print(f"   Contrast range: [{contrasts.min():.4f}, {contrasts.max():.4f}]")
    print(f"   Mean contrast: {contrasts.mean():.4f}")
    assert contrasts.min() >= 0, "Contrasts should be non-negative"
    assert contrasts.max() <= 1, "Contrasts should be <= 1"
    print("   ✓ guess_contrast passed")
    
    # Test guess_width
    print("\n   Testing guess_width...")
    widths = guess_width(data, frequencies[0], DEFAULT_VMIN, DEFAULT_VMAX)
    print(f"   Widths shape: {widths.shape}")
    print(f"   Width range: [{widths.min():.6f}, {widths.max():.6f}] GHz")
    print(f"   Mean width: {widths.mean():.6f} GHz")
    assert widths.min() >= 0, "Widths should be non-negative"
    print("   ✓ guess_width passed")

def test_pixel_functions():
    """Test individual pixel functions."""
    print("\n3. Testing individual pixel functions...")
    
    # Create a simple test spectrum with a dip
    freq = np.linspace(2.84, 2.85, 51)
    center_freq = 2.845
    
    # Create Lorentzian-like dip
    spectrum = 1.0 - 0.1 * np.exp(-((freq - center_freq) / 0.001)**2)
    
    print(f"   Test spectrum shape: {spectrum.shape}")
    print(f"   Spectrum range: [{spectrum.min():.3f}, {spectrum.max():.3f}]")
    
    # Test guess_center_pixel
    guessed_center = guess_center_pixel(spectrum, freq)
    print(f"   True center: {center_freq:.4f} GHz")
    print(f"   Guessed center: {guessed_center:.4f} GHz")
    print(f"   Error: {abs(guessed_center - center_freq):.6f} GHz")
    assert abs(guessed_center - center_freq) < 0.002, "Center guess error too large"
    print("   ✓ guess_center_pixel passed")
    
    # Test guess_contrast_pixel
    guessed_contrast = guess_contrast_pixel(spectrum)
    expected_contrast = (spectrum.max() - spectrum.min()) / spectrum.max()
    print(f"   Expected contrast: {expected_contrast:.4f}")
    print(f"   Guessed contrast: {guessed_contrast:.4f}")
    print(f"   Error: {abs(guessed_contrast - expected_contrast):.6f}")
    assert abs(guessed_contrast - expected_contrast) < 0.01, "Contrast guess error too large"
    print("   ✓ guess_contrast_pixel passed")
    
    # Test guess_width_pixel
    guessed_width = guess_width_pixel(spectrum, freq, DEFAULT_VMIN, DEFAULT_VMAX)
    print(f"   Guessed width: {guessed_width:.6f} GHz")
    assert guessed_width > 0, "Width should be positive"
    print("   ✓ guess_width_pixel passed")

def test_model_functions():
    """Test that model functions work correctly."""
    print("\n4. Testing model functions...")
    
    freq = np.linspace(2.84, 2.90, 100)
    
    # Test ESR15N
    print("   Testing ESR15N model...")
    params_15n = np.array([2.845, 2.885, 0.05, 0.05, 0.001, 0.001, 0.0])
    result_15n = esr15n(freq, params_15n)
    print(f"   ESR15N result shape: {result_15n.shape}")
    print(f"   ESR15N range: [{result_15n.min():.3f}, {result_15n.max():.3f}]")
    assert result_15n.shape == freq.shape, "ESR15N output shape mismatch"
    print("   ✓ ESR15N model passed")
    
    # Test ESR14N
    print("   Testing ESR14N model...")
    params_14n = np.array([2.84, 2.865, 2.89, 0.03, 0.03, 0.03, 0.001, 0.001, 0.001, 0.0])
    result_14n = esr14n(freq, params_14n)
    print(f"   ESR14N result shape: {result_14n.shape}")
    print(f"   ESR14N range: [{result_14n.min():.3f}, {result_14n.max():.3f}]")
    assert result_14n.shape == freq.shape, "ESR14N output shape mismatch"
    print("   ✓ ESR14N model passed")
    
    # Test ESRSINGLE
    print("   Testing ESRSINGLE model...")
    params_single = np.array([2.865, 0.05, 0.001, 0.0])
    result_single = esrsingle(freq, params_single)
    print(f"   ESRSINGLE result shape: {result_single.shape}")
    print(f"   ESRSINGLE range: [{result_single.min():.3f}, {result_single.max():.3f}]")
    assert result_single.shape == freq.shape, "ESRSINGLE output shape mismatch"
    print("   ✓ ESRSINGLE model passed")

def test_model_registry():
    """Test ModelRegistry functionality."""
    print("\n5. Testing ModelRegistry...")
    
    registry = ModelRegistry()
    models = registry.all()
    print(f"   Available models: {list(models.keys())}")
    
    for name in models.keys():
        model = registry.get(name)
        print(f"   {name}: {model.n_peaks} peaks, {model.n_parameters} parameters")
    
    print("   ✓ ModelRegistry passed")

def create_comparison_plot():
    """Create a plot comparing old and new function results."""
    print("\n6. Creating comparison plot...")
    
    # Create test data
    data, frequencies, scan_dims = create_test_data()
    
    # Get results from current functions
    freq = frequencies[0]
    centers = guess_center(data, freq)
    contrasts = guess_contrast(data)
    widths = guess_width(data, freq, DEFAULT_VMIN, DEFAULT_VMAX)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot sample spectrum
    pixel_idx = data.shape[-1] // 2  # Middle pixel
    spectrum = data[0, 0, :, pixel_idx]
    
    axes[0, 0].plot(freq, spectrum, 'b-o', markersize=3)
    axes[0, 0].set_title('Sample ODMR Spectrum')
    axes[0, 0].set_xlabel('Frequency (GHz)')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot normalized cumsum
    normalized = normalize_pixel(spectrum)
    axes[0, 1].plot(freq, normalized, 'r-o', markersize=3)
    axes[0, 1].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=DEFAULT_VMIN, color='g', linestyle='--', alpha=0.5, label=f'vmin={DEFAULT_VMIN}')
    axes[0, 1].axhline(y=DEFAULT_VMAX, color='g', linestyle='--', alpha=0.5, label=f'vmax={DEFAULT_VMAX}')
    axes[0, 1].set_title('Normalized Cumulative Sum')
    axes[0, 1].set_xlabel('Frequency (GHz)')
    axes[0, 1].set_ylabel('Normalized Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot center distribution
    axes[1, 0].hist(centers.flatten(), bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(freq), color='r', linestyle='--', label=f'True center: {np.mean(freq):.4f}')
    axes[1, 0].set_title('Center Frequency Distribution')
    axes[1, 0].set_xlabel('Frequency (GHz)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot contrast distribution
    axes[1, 1].hist(contrasts.flatten(), bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Contrast Distribution')
    axes[1, 1].set_xlabel('Contrast')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'function_compatibility_test.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved comparison plot to {plot_path}")
    
    # Show plot
    plt.show()

def main():
    """Run all compatibility and performance tests."""
    print("QDMpy Function Compatibility Test")
    print("="*50)
    
    # Suppress numba warnings for cleaner output
    warnings.filterwarnings('ignore', category=numba.NumbaWarning)
    
    try:
        # Run compatibility tests
        results = []
        results.append(test_normalize_pixel_compatibility())
        results.append(test_contrast_compatibility())
        results.append(test_center_compatibility())
        results.append(test_width_compatibility())
        
        # Summary
        print("\n" + "="*50)
        print("COMPATIBILITY TEST SUMMARY")
        print("="*50)
        test_names = ["Normalize Pixel", "Contrast", "Center", "Width"]
        for i, (name, result) in enumerate(zip(test_names, results)):
            status = "PASS" if result else "FAIL"
            print(f"{name}: {status}")
        
        all_passed = all(results)
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        if all_passed:
            # Run performance comparison
            performance_comparison()
            
            print("\n" + "="*50)
            print("CONCLUSION")
            print("="*50)
            print("✓ The new implementation produces identical results to the old code.")
            print("✓ Performance comparison has been completed.")
            print("✓ The new code is ready for use.")
        else:
            print("\n" + "="*50)
            print("ERROR")
            print("="*50)
            print("❌ Some compatibility tests failed!")
            print("❌ The new implementation does not match the old code.")
            print("❌ Further investigation is needed.")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())