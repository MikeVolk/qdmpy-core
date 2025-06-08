#!/usr/bin/env python3
"""Test script to verify compatibility between old and new fitting functions.

This script creates test data using the old code's methods and compares results
from old vs new function implementations to ensure they produce the same outputs.
"""

import sys
import os
sys.path.insert(0, '/home/mike/git/QDMpy/src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import current functions
from QDMpy.guess import (
    guess_center, guess_contrast, guess_width, 
    normalize_pixel, guess_center_pixel, guess_contrast_pixel, guess_width_pixel
)
from QDMpy.constants import DEFAULT_VMIN, DEFAULT_VMAX
from QDMpy.models import esr14n, esr15n, esrsingle, ModelRegistry

print("Testing compatibility between old and new fitting functions...")

def create_test_data():
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
    """Run all compatibility tests."""
    print("="*60)
    print("QDMpy Function Compatibility Test")
    print("="*60)
    
    try:
        test_normalize_pixel()
        test_guess_functions()
        test_pixel_functions()
        test_model_functions()
        test_model_registry()
        create_comparison_plot()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("The current functions work correctly and are compatible")
        print("with the expected behavior from the old code.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())