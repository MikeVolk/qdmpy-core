"""Fit results management for Quantum Diamond Microscopy.

This module provides the FitResult class which encapsulates the results of ODMR spectral
fitting and provides methods for analysis, visualization, and data export. The FitResult
class separates analysis functionality from data management, allowing for clean separation
of concerns and better code organization.

The FitResult class handles:
- Access to fitted parameters (centers, widths, contrasts, etc.)
- Magnetic field calculations from fitted resonance frequencies
- Visualization of spatial parameter maps and magnetic field maps
- Data export and persistence functionality
- Quality assessment and statistical analysis of fit results
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


class FitResult:
    """Contains ODMR fitting results and provides analysis methods.
    
    This class encapsulates the results from ODMR spectral fitting using a
    lightweight, data-only approach. It stores only the essential fitted parameters
    and metadata, without maintaining references to heavy objects like FitManager
    or Measurement.
    
    This design provides:
    - Clean separation of concerns
    - Easy serialization and persistence  
    - Minimal memory footprint
    - No circular dependencies
    
    Attributes:
        parameters: Dictionary of fitted parameters (centers, widths, contrasts, etc.)
        scan_dimensions: Spatial dimensions as (height, width)
        pixel_spacing: Physical spacing between pixels in meters
        model_name: Name of the model used for fitting
        metadata: Additional fitting metadata (quality metrics, etc.)
    """
    
    def __init__(
        self, 
        parameters: dict[str, NDArray],
        scan_dimensions: tuple[int, int],
        pixel_spacing: float,
        model_name: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Initialize FitResult with extracted fit data.
        
        Args:
            parameters: Dictionary of fitted parameters with keys like 'center',
                       'width_0', 'contrast', 'offset', 'chi2', 'states'
            scan_dimensions: Spatial dimensions as (height, width)
            pixel_spacing: Physical spacing between pixels in meters  
            model_name: Name of the model used for fitting
            metadata: Optional additional metadata dictionary
        """
        self.parameters = parameters
        self.scan_dimensions = scan_dimensions
        self.pixel_spacing = pixel_spacing
        self.model_name = model_name
        self.metadata = metadata or {}
        
        # Cache for expensive calculations
        self._b_field_cache: NDArray | None = None
        
        LOG.info("FitResult initialized with model: %s", model_name)
        LOG.debug("Available parameters: %s", list(parameters.keys()))
    
    def __repr__(self) -> str:
        """Return string representation of FitResult."""
        n_pixels = self.scan_dimensions[0] * self.scan_dimensions[1]
        n_params = len(self.parameters)
        return (f"FitResult(model='{self.model_name}', "
                f"n_pixels={n_pixels}, "
                f"parameters={n_params})")
    
    @property
    def centers(self) -> NDArray:
        """Get resonance center frequencies in Hz.
        
        Returns:
            Array of center frequencies with shape matching spatial dimensions
        """
        return self.parameters['center']
    
    @property
    def linewidths(self) -> NDArray:
        """Get ODMR linewidths in Hz.
        
        For models with multiple lines, returns the primary linewidth.
        
        Returns:
            Array of linewidths with shape matching spatial dimensions
        """
        return self.parameters.get('width_0', self.parameters.get('width'))
    
    @property
    def contrasts(self) -> NDArray:
        """Get ODMR contrasts (normalized).
        
        Returns:
            Array of contrast values with shape matching spatial dimensions
        """
        return self.parameters['contrast']
    
    @property
    def offsets(self) -> NDArray:
        """Get baseline offsets.
        
        Returns:
            Array of offset values with shape matching spatial dimensions
        """
        return self.parameters.get('offset', np.zeros_like(self.centers))
    
    @property
    def chi2(self) -> NDArray:
        """Get fit quality (chi-squared values).
        
        Returns:
            Array of chi-squared values with shape matching spatial dimensions
        """
        return self.parameters['chi2']
    
    @property
    def fit_states(self) -> NDArray:
        """Get fitting convergence states.
        
        Returns:
            Array of fit state codes with shape matching spatial dimensions
        """
        return self.parameters.get('states', np.zeros_like(self.centers, dtype=int))
    
    def get_parameter(self, param_name: str) -> NDArray:
        """Get any fitted parameter by name.
        
        Args:
            param_name: Name of the parameter to retrieve
            
        Returns:
            Array of parameter values
            
        Raises:
            KeyError: If parameter name is not found
        """
        if param_name not in self.parameters:
            available = list(self.parameters.keys())
            raise KeyError(f"Parameter '{param_name}' not found. Available: {available}")
        return self.parameters[param_name]
    
    def get_parameter_map(self, param_name: str) -> NDArray:
        """Get parameter reshaped as 2D spatial map.
        
        Args:
            param_name: Name of the parameter to retrieve
            
        Returns:
            2D array with shape (height, width) for spatial visualization
        """
        param_data = self.get_parameter(param_name)
        return param_data.reshape(self.scan_dimensions)
    
    def calculate_b_field(self, force_recalculate: bool = False) -> NDArray:
        """Calculate magnetic field map from fitted resonance frequencies.
        
        This method computes the magnetic field strength based on the Zeeman
        splitting of NV center resonances.
        
        Args:
            force_recalculate: If True, recalculate even if cached result exists
            
        Returns:
            2D array of magnetic field values in Tesla
            
        Note:
            The calculation assumes the standard NV center gyromagnetic ratio
            and appropriate model-specific conversion factors.
        """
        if self._b_field_cache is None or force_recalculate:
            LOG.info("Calculating magnetic field from %s fit results", self.model_name)
            self._b_field_cache = self._compute_b_field()
            
        return self._b_field_cache
    
    def _compute_b_field(self) -> NDArray:
        """Internal method to compute magnetic field from resonance frequencies.
        
        Returns:
            2D array of magnetic field values in Tesla
        """
        # Get center frequencies and reshape to spatial map
        centers_map = self.get_parameter_map('center')
        
        # NV center gyromagnetic ratio (Hz/T)
        # For NV centers: γ/2π ≈ 28.0 GHz/T
        gamma_nv = 28.0e9  # Hz/T
        
        # Zero-field splitting frequency (Hz)
        # D = 2.87 GHz for NV centers
        d_zfs = 2.87e9  # Hz
        
        # Calculate magnetic field: |B| = |f_center - D| / γ
        # This assumes the center frequency represents the shifted resonance
        b_field = np.abs(centers_map - d_zfs) / gamma_nv
        
        LOG.debug("B-field calculation: mean=%.2e T, std=%.2e T", 
                 b_field.mean(), b_field.std())
        
        return b_field
    
    def get_fit_quality_metrics(self) -> dict[str, float]:
        """Calculate overall fit quality metrics.
        
        Returns:
            Dictionary containing various quality metrics
        """
        chi2_values = self.chi2
        
        # Calculate basic statistics
        metrics = {
            'mean_chi2': float(np.mean(chi2_values)),
            'median_chi2': float(np.median(chi2_values)),
            'std_chi2': float(np.std(chi2_values)),
            'n_pixels': int(chi2_values.size)
        }
        
        # Add convergence rate if states are available
        if 'states' in self.parameters:
            states_values = self.fit_states
            metrics.update({
                'convergence_rate': float(np.mean(states_values == 0)),  # Assuming 0 = converged
                'n_converged': int(np.sum(states_values == 0))
            })
        
        # Add any pre-computed metrics from metadata
        if 'quality_metrics' in self.metadata:
            metrics.update(self.metadata['quality_metrics'])
        
        LOG.info("Fit quality metrics: mean_chi2=%.3f, n_pixels=%d",
                metrics['mean_chi2'], metrics['n_pixels'])
        
        return metrics
    
    
    def save_results(self, filepath: str | Path) -> None:
        """Save fit results to file.
        
        Args:
            filepath: Path where to save the results
            
        Note:
            This saves the essential fit results in a format that can be
            reloaded later. The full FitManager state may not be preserved.
        """
        filepath = Path(filepath)
        
        # Prepare data for saving
        save_data = {
            'model_name': self.model_name,
            'scan_dimensions': self.scan_dimensions,
            'pixel_spacing': self.pixel_spacing,
            'metadata': self.metadata,
            'parameters': self.parameters.copy()  # Copy all parameters
        }
        
        # Add magnetic field if calculated
        if self._b_field_cache is not None:
            save_data['b_field'] = self._b_field_cache
        
        # Save to NPZ format for efficiency
        np.savez_compressed(filepath, **save_data)
        LOG.info("Fit results saved to: %s", filepath)
    
    @classmethod
    def load_results(cls, filepath: str | Path) -> dict[str, Any]:
        """Load saved fit results from file.
        
        Args:
            filepath: Path to the saved results file
            
        Returns:
            Dictionary containing loaded results data
            
        Note:
            This returns the raw data dictionary. Creating a full FitResult
            object would require reconstructing the FitManager and Measurement,
            which may not always be possible or desired.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        # Convert back to regular dict
        result_data = {key: data[key] for key in data.files}
        
        LOG.info("Fit results loaded from: %s", filepath)
        return result_data