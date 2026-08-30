"""Bad-fit detection and neighbor-based pixel refitting for ODMR data.

After fitting, some pixels may have poor fit quality (high chi-squared or
non-convergence). Since neighboring pixels share similar magnetic environments,
their fitted parameters make excellent initial guesses for refitting the outlier
pixels.

For clusters of bad pixels, multiple passes are applied iteratively: each pass
refits the cluster border (pixels adjacent to good data), then the newly-refitted
pixels become available as neighbors for the next pass. This continues until
convergence (no more pixels can be refitted) or ``RefitSettings.max_iterations``
is reached.

Public API:
    RefitSettings: Configuration dataclass for outlier detection and refitting.
    identify_outlier_pixels: Flag pixels exceeding a chi2 percentile threshold.
    compute_neighbor_guesses: Build per-outlier initial guesses from spatial neighbors.
    refit_outliers: Orchestrate detection, guess computation, and GPU refitting.

All operations are purely functional; they do not mutate the input FitResult.
A new FitResult is returned with updated parameters and refit metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.ndimage import convolve

from qdmpy.fitting.manager import FOLDED_CONSTRAINT_OVERRIDES
from qdmpy.fitting.result import FitResult

if TYPE_CHECKING:
    import xarray as xr

    from qdmpy.fitting.manager import FitManager


class RefitSettings(BaseModel):
    """Configuration for bad-fit detection and neighbor-based refitting.

    Attributes:
        chi2_percentile: Pixels with chi2 above this percentile (per pol/frange subspace)
            are considered outliers. Range [0, 100].
        include_non_converged: When True, pixels where the fit did not converge
            (states != 0) are also marked as outliers regardless of chi2.
        window_size: Side length of the spatial neighborhood window. Must be odd and >= 3.
        min_good_neighbors: Minimum number of non-outlier neighbors required to compute
            a reliable median guess. Pixels with fewer good neighbors are skipped.
        max_iterations: Maximum number of refit passes. Each pass refits the border
            pixels of remaining outlier clusters; subsequent passes peel inward one
            window-half at a time. The loop stops early when no further pixels can
            be refitted. Default 1 (single pass, original behaviour).
    """

    model_config = ConfigDict(frozen=True)

    chi2_percentile: float = 90.0
    include_non_converged: bool = True
    window_size: int = 5
    min_good_neighbors: int = 3
    max_iterations: int = 1

    @field_validator("chi2_percentile")
    @classmethod
    def validate_percentile(cls, v: float) -> float:
        """Validate chi2_percentile is in [0, 100]."""
        if not 0.0 <= v <= 100.0:
            msg = f"chi2_percentile must be in [0, 100], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        """Validate window_size is odd and >= 3."""
        if v < 3 or v % 2 == 0:
            msg = f"window_size must be odd and >= 3, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("min_good_neighbors")
    @classmethod
    def validate_min_good_neighbors(cls, v: int) -> int:
        """Validate min_good_neighbors is >= 1."""
        if v < 1:
            msg = f"min_good_neighbors must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max_iterations is >= 1."""
        if v < 1:
            msg = f"max_iterations must be >= 1, got {v}"
            raise ValueError(msg)
        return v


def identify_outlier_pixels(
    chi2: NDArray,
    states: NDArray,
    settings: RefitSettings,
) -> NDArray:
    """Identify pixels with poor fit quality as a boolean mask.

    For each (pol, frange) subspace, a pixel is flagged as an outlier if its
    chi-squared value exceeds the threshold computed from ``settings.chi2_percentile``.
    Optionally, non-converged pixels (states != 0) are also flagged.

    Args:
        chi2: Chi-squared values with shape (n_pol, n_frange, h, w).
        states: Fit convergence state codes with shape (n_pol, n_frange, h, w).
        settings: Configuration for outlier detection.

    Returns:
        Boolean mask with shape (n_pol, n_frange, h, w). True = outlier.
    """
    n_pol, n_frange = chi2.shape[:2]
    outlier_mask = np.zeros_like(chi2, dtype=bool)

    for p in range(n_pol):
        for f in range(n_frange):
            threshold = float(np.percentile(chi2[p, f], settings.chi2_percentile))
            outlier_mask[p, f] = chi2[p, f] > threshold
            if settings.include_non_converged:
                outlier_mask[p, f] |= states[p, f] != 0

    return outlier_mask


def compute_neighbor_guesses(
    parameters: dict[str, NDArray],
    outlier_mask: NDArray,
    settings: RefitSettings,
    model_parameter_names: list[str],
) -> tuple[dict[str, NDArray], NDArray]:
    """Compute initial parameter guesses for outlier pixels from spatial neighbors.

    For each outlier pixel in each (pol, frange) subspace, the initial guess is the
    median of non-outlier neighbor values within a ``window_size x window_size`` window.
    Good-neighbor counts are computed via convolution for efficiency; the per-pixel
    median is computed in a loop over outlier pixels only.

    Args:
        parameters: Dict of fitted parameters, each with shape (n_pol, n_frange, h, w).
            Keys 'chi2' and 'states' are ignored (only model parameter names are used).
        outlier_mask: Boolean mask (n_pol, n_frange, h, w). True = outlier.
        settings: Configuration including window_size and min_good_neighbors.
        model_parameter_names: Names of model parameters (excludes 'chi2', 'states').

    Returns:
        Tuple of:
            - guess_dict: Dict mapping param_name -> (n_pol, n_frange, h, w) array
              of initial guesses. Non-outlier pixels retain their original values.
            - refittable_mask: Boolean mask (n_pol, n_frange, h, w). True = pixel
              has enough good neighbors for a reliable initial guess.
    """
    n_pol, n_frange, h, w = outlier_mask.shape
    kernel = np.ones((settings.window_size, settings.window_size), dtype=np.float32)
    half = settings.window_size // 2

    # Initialize guesses from current parameter values (mutable copies)
    guess_dict: dict[str, NDArray] = {
        name: np.array(parameters[name], dtype=np.float32)
        for name in model_parameter_names
        if name in parameters
    }
    refittable_mask = np.zeros((n_pol, n_frange, h, w), dtype=bool)

    for p in range(n_pol):
        for f in range(n_frange):
            good = ~outlier_mask[p, f]  # (h, w): True = non-outlier

            # Fast convolution-based neighbor count (out-of-bounds = 0, not good)
            good_count = convolve(good.astype(np.float32), kernel, mode="constant", cval=0.0)

            outlier_ys, outlier_xs = np.where(outlier_mask[p, f])

            for y, x in zip(outlier_ys, outlier_xs, strict=True):
                if good_count[y, x] < settings.min_good_neighbors:
                    continue

                refittable_mask[p, f, y, x] = True

                y0 = max(0, y - half)
                y1 = min(h, y + half + 1)
                x0 = max(0, x - half)
                x1 = min(w, x + half + 1)
                good_window = good[y0:y1, x0:x1]

                for name in model_parameter_names:
                    if name not in parameters:
                        continue
                    param_window = parameters[name][p, f, y0:y1, x0:x1]
                    good_values = param_window[good_window]
                    guess_dict[name][p, f, y, x] = float(np.median(good_values))

    return guess_dict, refittable_mask


def _accept_improved_refits(
    new_params: dict[str, NDArray],
    model_parameter_names: list[str],
    *,
    irange: int,
    rows: NDArray,
    cols: NDArray,
    new_fit_params: NDArray,
    new_states_arr: NDArray,
    new_chi2_arr: NDArray,
) -> int:
    """Write refit results back only where they improve on the pixel's chi2.

    A refit can land in a worse local minimum than the pixel already had;
    writing it back unconditionally would silently degrade the map. A
    non-finite old chi2 has no valid baseline, so any finite refit result is
    accepted for it. Mutates ``new_params`` in place.

    Returns:
        Number of (polarity, pixel) entries whose refit was accepted.
    """
    old_chi2 = new_params["chi2"][:, irange, rows, cols]
    improved = (new_chi2_arr < old_chi2) | ~np.isfinite(old_chi2)

    for iparam, pname in enumerate(model_parameter_names):
        old_param = new_params[pname][:, irange, rows, cols]
        new_params[pname][:, irange, rows, cols] = np.where(
            improved, new_fit_params[:, :, iparam], old_param
        )
    old_states = new_params["states"][:, irange, rows, cols]
    new_params["chi2"][:, irange, rows, cols] = np.where(improved, new_chi2_arr, old_chi2)
    new_params["states"][:, irange, rows, cols] = np.where(improved, new_states_arr, old_states)

    return int(np.sum(improved))


def _refit_pass(
    fit_result: FitResult,
    data: xr.DataArray,
    frequencies: NDArray,
    fit_manager: FitManager,
    settings: RefitSettings,
) -> FitResult:
    """Execute one outlier-detection and refitting pass.

    Detects outlier pixels in ``fit_result``, computes neighbor-based initial
    guesses, and refits those pixels via the GPU. Returns the input ``fit_result``
    unchanged (same object) when no outliers are detected or when all detected
    outliers are interior cluster pixels with no good neighbors. Otherwise returns
    a new FitResult with updated parameters.

    Args:
        fit_result: Current FitResult (may already be an output from a previous pass).
        data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).
        frequencies: Frequency array in GHz, shape (n_frange, n_freq).
        fit_manager: Configured FitManager.
        settings: Refit configuration.

    Returns:
        New FitResult if any pixels were refitted; the input object otherwise.
    """
    chi2 = fit_result.parameters["chi2"]
    states = fit_result.parameters.get("states", np.zeros_like(chi2, dtype=np.int32))

    outlier_mask = identify_outlier_pixels(chi2, states, settings)
    n_outliers = int(np.sum(outlier_mask))
    n_total = int(chi2.size)

    if n_outliers == 0:
        logger.debug("refit pass: no outlier pixels detected")
        return fit_result

    logger.info(
        "refit pass: {} outlier pixels detected ({:.1f}%)",
        n_outliers,
        100.0 * n_outliers / n_total,
    )

    model_parameter_names = fit_manager.parameter_names
    guess_dict, refittable_mask = compute_neighbor_guesses(
        fit_result.parameters, outlier_mask, settings, model_parameter_names
    )

    # Mutable copies — FitResult protects arrays with flags.writeable = False
    new_params: dict[str, NDArray] = {
        name: np.array(arr) for name, arr in fit_result.parameters.items()
    }

    n_pol, n_frange, h, w = chi2.shape
    n_params = fit_manager.n_parameter
    f_ghz = np.atleast_2d(frequencies)
    data_values = data.values  # (n_pol, n_frange, h, w, n_freq)
    constraint_overrides = (
        FOLDED_CONSTRAINT_OVERRIDES if fit_result.metadata.get("folded_fit") else None
    )

    per_frange_info: dict[str, dict[str, int]] = {}

    for irange in range(n_frange):
        # Union across polarities: refit a pixel if it's an outlier in any pol.
        # Only require neighbor-based refittability for pols where the pixel IS
        # an outlier; non-outlier pols use their current value as the initial guess.
        union_outlier = np.any(outlier_mask[:, irange], axis=0)  # (h, w)
        not_blocking = ~outlier_mask[:, irange] | refittable_mask[:, irange]  # (n_pol, h, w)
        all_refittable = np.all(not_blocking, axis=0)  # (h, w)
        pixels_2d = union_outlier & all_refittable

        flat_pixel_indices = np.where(pixels_2d.ravel())[0]
        n_refit = len(flat_pixel_indices)
        n_outlier_frange = int(np.sum(union_outlier))

        per_frange_info[f"frange_{irange}"] = {
            "n_outlier": n_outlier_frange,
            "n_refitted": n_refit,
        }

        if n_refit == 0:
            logger.debug(
                "frange {}: no refittable outlier pixels (outliers={}, refittable=0)",
                irange,
                n_outlier_frange,
            )
            continue

        rows, cols = np.unravel_index(flat_pixel_indices, (h, w))

        # Extract data subset: (n_pol, n_refit, n_freq)
        flat_data = data_values[:, irange].reshape(n_pol, h * w, -1)
        subset_data = flat_data[:, flat_pixel_indices, :]

        # Build initial guesses from neighbor-derived estimates: (n_pol, n_refit, n_params)
        subset_guesses = np.empty((n_pol, n_refit, n_params), dtype=np.float32)
        for iparam, pname in enumerate(model_parameter_names):
            flat_guesses = guess_dict[pname][:, irange].reshape(n_pol, h * w)
            subset_guesses[:, :, iparam] = flat_guesses[:, flat_pixel_indices]

        # GPU refit of the outlier pixel subset
        raw = fit_manager.fit_frange(
            subset_data,
            f_ghz[irange],
            subset_guesses,
            irange=irange,
            n_frange=n_frange,
            constraint_overrides=constraint_overrides,
        )

        # Reshape raw gpufit output: (n_pol * n_refit, ...) -> (n_pol, n_refit, ...)
        new_fit_params = np.asarray(raw[0]).reshape(n_pol, n_refit, n_params)
        new_states_arr = np.asarray(raw[1]).reshape(n_pol, n_refit)
        new_chi2_arr = np.asarray(raw[2]).reshape(n_pol, n_refit)

        n_accepted = _accept_improved_refits(
            new_params,
            model_parameter_names,
            irange=irange,
            rows=rows,
            cols=cols,
            new_fit_params=new_fit_params,
            new_states_arr=new_states_arr,
            new_chi2_arr=new_chi2_arr,
        )

        logger.info(
            "frange {}: refitted {} of {} outlier pixels ({} accepted, {} rejected as worse)",
            irange,
            n_refit,
            n_outlier_frange,
            n_accepted,
            new_chi2_arr.size - n_accepted,
        )

    n_total_refitted = sum(v["n_refitted"] for v in per_frange_info.values())

    if n_total_refitted == 0:
        # All detected outliers were interior cluster pixels with no good neighbors
        logger.debug("refit pass: {} outlier pixels detected but none refittable", n_outliers)
        return fit_result

    refit_info: dict[str, object] = {
        "chi2_percentile": settings.chi2_percentile,
        "n_outliers_detected": n_outliers,
        "n_refitted": n_total_refitted,
        "per_frange": per_frange_info,
    }
    return type(fit_result)(
        parameters=new_params,
        scan_dimensions=fit_result.scan_dimensions,
        pixel_spacing=fit_result.pixel_spacing,
        model_name=fit_result.model_name,
        metadata={**fit_result.metadata, "refit_info": refit_info},
    )


def refit_outliers(
    fit_result: FitResult,
    data: xr.DataArray,
    frequencies: NDArray,
    fit_manager: FitManager,
    settings: RefitSettings | None = None,
) -> FitResult:
    """Refit outlier pixels using neighbor-derived initial guesses.

    Identifies pixels with poor fit quality (high chi2 or non-convergence),
    computes initial guesses from their spatial neighbors, and refits just those
    pixels via the GPU fitter. Returns a new FitResult with updated parameters
    and refit metadata; the original FitResult is not mutated.

    For clusters of bad pixels, multiple passes peel the cluster inward: each
    pass refits border pixels (those adjacent to good data), which then become
    good neighbors for the next pass. The loop terminates early when no further
    pixels can be refitted. Set ``settings.max_iterations`` high enough to
    clear clusters whose radius exceeds ``window_size // 2``.

    Union across polarities: if a pixel is an outlier in any polarity for a given
    frequency range, it is refit for all polarities together (as required by
    fit_frange's joint polarity fitting). For pols where the pixel is an outlier,
    a neighbor-based guess is used; for pols where it is not, the current fitted
    value serves as the initial guess. A pixel is skipped only if an outlier pol
    has too few good neighbors to form a reliable guess.

    Args:
        fit_result: Original FitResult from a full fit.
        data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).
        frequencies: Frequency array in GHz, shape (n_frange, n_freq).
        fit_manager: Configured FitManager (model and constraints must already be set).
        settings: Refit configuration. Defaults to RefitSettings().

    Returns:
        New FitResult with outlier pixels replaced by refit values and
        'refit_info' key added to metadata.
    """
    if settings is None:
        settings = RefitSettings()

    current = fit_result
    for iteration in range(settings.max_iterations):
        nxt = _refit_pass(current, data, frequencies, fit_manager, settings)
        if nxt is current:
            if iteration == 0:
                logger.info("refit_outliers: no outlier pixels detected, returning original")
            else:
                logger.info(
                    "refit_outliers: converged after {} pass(es)",
                    iteration,
                )
            break
        current = nxt
        logger.info("refit_outliers: pass {} complete", iteration + 1)

    return current
