"""Model definitions for fitting ODMR spectra.

Convention: All frequency values are in GHz.

This module provides models for fitting Optically Detected Magnetic Resonance (ODMR)
spectra from Nitrogen-Vacancy (NV) centers in diamond. It includes models for different
nitrogen isotopes (14N and 15N) and different configurations, along with a registry
system for managing and retrieving models.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import reduce
from typing import Any, ClassVar

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.constants import AHYP_14N, AHYP_15N

# Mirrors pygpufit.gpufit.ModelID values. Fixed protocol constants, not runtime
# lookups, so Model never imports pygpufit — fitting/backends.py is the only
# module that does (QEP-068). Custom CPU-only models use -1 (see Model docs
# below) to signal "no gpufit model_id; fit via a CPU backend instead".
_GPUFIT_MODEL_ID_ESR14N = 15
_GPUFIT_MODEL_ID_ESR15N = 16
_GPUFIT_MODEL_ID_ESRSINGLE = 17


def _ensure_2d(parameter: NDArray) -> NDArray:
    """Framework-neutral ``np.atleast_2d`` for model parameter arrays.

    ``np.atleast_2d`` on a torch tensor would trigger ``__array__`` and
    silently copy the data to a CPU numpy array, defeating GPU fitting
    (QEP-069). This helper reshapes anything that already has ``.ndim``
    (numpy arrays *and* torch tensors) without conversion, and coerces
    plain Python sequences via ``np.asarray`` to preserve the previous
    behaviour for list inputs.
    """
    if not hasattr(parameter, "ndim"):
        parameter = np.asarray(parameter)
    if parameter.ndim == 0:
        return parameter.reshape(1, 1)
    if parameter.ndim == 1:
        return parameter.reshape(1, -1)
    return parameter


def esr14n(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
    ahyp: float = AHYP_14N,
) -> NDArray[np.floating]:
    """Evaluate the ESR14N model for 14N nitrogen-vacancy centers.

    This function calculates the ODMR spectrum response for NV centers with 14N nitrogen
    isotope (I=1), which exhibits three resonance dips due to hyperfine interaction.
    The model uses Lorentzian lineshapes for each resonance component.

    The three dips are positioned at:
    - Center frequency - ahyp (mI = -1)
    - Center frequency (mI = 0)
    - Center frequency + ahyp (mI = +1)

    Args:
        x: Array of frequency values in GHz.
        parameter: Parameter array with shape (N, 6) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (GHz)
                - [1] width: Linewidth parameter (GHz)
                - [2] contrast_-1: Contrast of the mI=-1 dip (0-1)
                - [3] contrast_0: Contrast of the mI=0 dip (0-1)
                - [4] contrast_+1: Contrast of the mI=+1 dip (0-1)
                - [5] offset: Baseline offset (0-1)
        ahyp: Hyperfine splitting constant (GHz). Defaults to AHYP_14N.

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - ∑ᵢ (contrastᵢ * width² / ((x - posᵢ)² + width²))
        where posᵢ are the three resonance positions.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87, 2.88, 100)
        >>> params = np.array([2.87, 0.002, 0.1, 0.2, 0.1, 0.0])
        >>> spectrum = esr14n(x, params)
    """
    parameter = _ensure_2d(parameter)
    center = parameter[:, 0:1]
    width_sq = parameter[:, 1:2] ** 2
    c0, c1, c2 = parameter[:, 2:3], parameter[:, 3:4], parameter[:, 4:5]
    offset = parameter[:, 5:6]

    dip1 = c0 * width_sq / ((x - center + ahyp) ** 2 + width_sq)
    dip2 = c1 * width_sq / ((x - center) ** 2 + width_sq)
    dip3 = c2 * width_sq / ((x - center - ahyp) ** 2 + width_sq)
    return 1 + offset - dip1 - dip2 - dip3


def esr15n(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
    ahyp: float = AHYP_15N,
) -> NDArray[np.floating]:
    """Evaluate the ESR15N model for 15N nitrogen-vacancy centers.

    This function calculates the ODMR spectrum response for NV centers with 15N nitrogen
    isotope (I=1/2), which exhibits two resonance dips due to hyperfine interaction.
    The model uses Lorentzian lineshapes for each resonance component.

    The two dips are positioned at:
    - Center frequency - ahyp (mI = -1/2)
    - Center frequency + ahyp (mI = +1/2)

    Args:
        x: Array of frequency values in GHz.
        parameter: Parameter array with shape (N, 5) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (GHz)
                - [1] width: Linewidth parameter (GHz)
                - [2] contrast_-1/2: Contrast of the mI=-1/2 dip (0-1)
                - [3] contrast_+1/2: Contrast of the mI=+1/2 dip (0-1)
                - [4] offset: Baseline offset (0-1)
        ahyp: Hyperfine splitting constant (GHz). Defaults to AHYP_15N.

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - ∑ᵢ (contrastᵢ * width² / ((x - posᵢ)² + width²))
        where posᵢ are the two resonance positions.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87, 2.88, 100)
        >>> params = np.array([2.87, 0.002, 0.15, 0.15, 0.0])
        >>> spectrum = esr15n(x, params)
    """
    parameter = _ensure_2d(parameter)
    center = parameter[:, 0:1]
    width_sq = parameter[:, 1:2] ** 2
    c0, c1 = parameter[:, 2:3], parameter[:, 3:4]
    offset = parameter[:, 4:5]

    dip1 = c0 * width_sq / ((x - center + ahyp) ** 2 + width_sq)
    dip2 = c1 * width_sq / ((x - center - ahyp) ** 2 + width_sq)
    return 1 + offset - dip1 - dip2


def esrsingle(x: NDArray[np.floating], parameter: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate the ESRSINGLE model for single resonance systems.

    This function calculates the ODMR spectrum response for systems with a single
    resonance dip, without hyperfine splitting. This model is useful for isolated
    spin systems or when hyperfine structure is not resolved.

    Args:
        x: Array of frequency values in GHz.
        parameter: Parameter array with shape (N, 4) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (GHz)
                - [1] width: Linewidth parameter (GHz)
                - [2] contrast: Contrast of the dip (0-1)
                - [3] offset: Baseline offset (0-1)

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - (contrast * width² / ((x - center)² + width²))

        This is a simple Lorentzian absorption line.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87, 2.88, 100)
        >>> params = np.array([2.875, 0.003, 0.2, 0.0])
        >>> spectrum = esrsingle(x, params)
    """
    parameter = _ensure_2d(parameter)
    center = parameter[:, 0:1]
    width_sq = parameter[:, 1:2] ** 2
    contrast = parameter[:, 2:3]
    offset = parameter[:, 3:4]

    dip = contrast * width_sq / ((x - center) ** 2 + width_sq)
    return 1 + offset - dip


def _lorentzian_dips_jacobian(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
    dip_offsets: tuple[float, ...],
) -> tuple[NDArray[np.floating], ...]:
    """Analytic derivatives of a sum-of-Lorentzian-dips model (QEP-073).

    All three ESR models share one functional form -- dips at fixed offsets
    from a common ``center``, sharing one HWHM ``width`` and one ``offset``::

        f(x) = 1 + offset - sum_i c_i * w^2 / ((x - center - delta_i)^2 + w^2)

    so one derivation covers them all. Writing ``d_i = x - center - delta_i``
    and ``D_i = d_i^2 + w^2``::

        df/dcenter = -sum_i c_i * 2 w^2 d_i / D_i^2
        df/dw      = -sum_i c_i * 2 w d_i^2 / D_i^2
        df/dc_i    = -w^2 / D_i
        df/doffset = 1

    Cross-checked term by term against gpufit's kernels
    (``Gpufit/models/esr14N.cuh``), which are the numerical reference.

    Returns a *tuple of columns* rather than a stacked array: stacking would
    need ``np.stack`` or ``torch.stack`` and break the framework-neutral
    contract that lets one implementation serve numpy and torch. Callers stack
    with their own framework.

    Args:
        x: Frequency values in GHz.
        parameter: Parameter array, shape (N, 3 + n_dips): center, width, one
            contrast per dip, then offset.
        dip_offsets: Dip positions relative to ``center``, in GHz.

    Returns:
        One (N, len(x)) derivative array per parameter, ordered to match
        ``parameter_names``.
    """
    parameter = _ensure_2d(parameter)
    center = parameter[:, 0:1]
    width = parameter[:, 1:2]
    width_sq = width**2

    center_terms = []
    width_terms = []
    contrast_cols = []
    for i, delta in enumerate(dip_offsets):
        contrast = parameter[:, 2 + i : 3 + i]
        d = x - center - delta
        inv_d = 1 / (d * d + width_sq)
        # -w^2/D_i: derivative w.r.t. this dip's own contrast.
        contrast_cols.append(-width_sq * inv_d)
        center_terms.append(contrast * (2 * width_sq * d) * inv_d * inv_d)
        width_terms.append(contrast * (2 * width * d * d) * inv_d * inv_d)

    # reduce, not sum(): a 0 seed would make the accumulator a python scalar
    # for one add and drop the array/tensor type on the way through.
    d_center = reduce(operator.add, center_terms)
    d_width = reduce(operator.add, width_terms)

    # d/doffset is identically 1, but must broadcast to (N, len(x)) in whichever
    # framework the caller uses -- derive it from the inputs rather than np.ones.
    ones = (x - x) + 1 + 0 * center

    return (-d_center, -d_width, *contrast_cols, ones)


def esr14n_jacobian(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
    ahyp: float = AHYP_14N,
) -> tuple[NDArray[np.floating], ...]:
    """Analytic derivatives of :func:`esr14n`; see :func:`_lorentzian_dips_jacobian`."""
    return _lorentzian_dips_jacobian(x, parameter, (-ahyp, 0.0, ahyp))


def esr15n_jacobian(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
    ahyp: float = AHYP_15N,
) -> tuple[NDArray[np.floating], ...]:
    """Analytic derivatives of :func:`esr15n`; see :func:`_lorentzian_dips_jacobian`."""
    return _lorentzian_dips_jacobian(x, parameter, (-ahyp, ahyp))


def esrsingle_jacobian(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
) -> tuple[NDArray[np.floating], ...]:
    """Analytic derivatives of :func:`esrsingle`; see :func:`_lorentzian_dips_jacobian`."""
    return _lorentzian_dips_jacobian(x, parameter, (0.0,))


class Model(ABC):
    """Abstract base class for ODMR spectral models.

    This class defines the interface for all models used to fit ODMR spectra.
    Each concrete model implementation must provide a function that evaluates
    the model given a set of parameters. The class also provides utilities for
    parameter management and constraint handling.

    All ODMR models in QDMpy should inherit from this class to ensure consistency
    and compatibility with the fitting infrastructure.

    **Custom model contract:**

    To add a new ESR line-shape model, subclass ``Model``, set the required
    class-level attributes, and register it with ``@ModelRegistry.register``.
    ``func`` must use only framework-neutral operations — arithmetic,
    broadcasting, and slicing that work identically on numpy arrays and
    torch tensors (no ``np.*`` calls on ``x`` or ``parameters``). Models
    written this way are fittable on the GPU via ``backend='torch'``
    (QEP-069); numpy-only models remain fittable via ``backend='scipy'``:

    .. code-block:: python

        from typing import ClassVar
        from qdmpy import Model, ModelRegistry
        import numpy as np
        from numpy.typing import NDArray

        @ModelRegistry.register
        class MyModel(Model):
            name: ClassVar[str] = 'MYMODEL'

            def __init__(self) -> None:
                super().__init__(
                    'MYMODEL',
                    n_peaks=1,
                    parameter_names=['center', 'width', 'contrast', 'offset'],
                )
                self.model_id = -1  # CPU-only; gpufit not used for custom models

            @property
            def parameter_types(self) -> dict[str, str]:
                return {'center': 'center', 'width': 'width',
                        'contrast': 'contrast', 'offset': 'offset'}

            @property
            def frequency_parameters(self) -> list[str]:
                return ['center']   # parameters stored in GHz units

            def func(self, x: NDArray, parameters: NDArray) -> NDArray:
                # x shape: (n_freq,); parameters shape: (N, n_params)
                # return shape: (N, n_freq)
                parameters = np.atleast_2d(parameters)
                center = parameters[:, 0:1]
                width_sq = parameters[:, 1:2] ** 2
                contrast = parameters[:, 2:3]
                offset = parameters[:, 3:4]
                dip = contrast * width_sq / ((x - center) ** 2 + width_sq)
                return 1 + offset - dip

    Once registered, the model is available by name:

    .. code-block:: python

        result = measurement.fit_odmr(model='MYMODEL')

    Attributes:
        name: Unique identifier for the model (e.g., 'ESR14N', 'ESR15N').
        parameter_names: List of parameter names (e.g., ['center', 'width', 'contrast_0', ...]).
        n_peaks: Number of resonance peaks in the model (1 for single, 2 for 15N, 3 for 14N).

    Example:
        >>> from qdmpy.fitting.models import ESR14N
        >>> model = ESR14N()
        >>> print(f"Model: {model.name}, Parameters: {model.n_parameters}")
        Model: ESR14N, Parameters: 6
    """

    model_id: int

    def __init__(self: Model, name: str, n_peaks: int, parameter_names: list[str]) -> None:
        """Initialize a model with basic properties.

        Args:
            name: Unique identifier for the model.
            n_peaks: Number of resonance peaks in the model.
            parameter_names: List of parameter names (e.g., ['center', 'width', 'contrast_0', ...]).
        """
        self.name = name
        self.parameter_names = parameter_names
        self.n_peaks = n_peaks

    @property
    @abstractmethod
    def parameter_types(self: Model) -> dict[str, str]:
        """Map each parameter name to its type category.

        Returns:
            Dict mapping param name -> type ('center', 'width', 'contrast', 'offset').
        """

    @property
    @abstractmethod
    def frequency_parameters(self: Model) -> list[str]:
        """Parameter names stored in GHz units (center, width).

        Used to derive the units dict for display purposes.  The pyGpufit kernels
        use GHz throughout (AHYP constants are in GHz), so no unit conversion is
        performed at the GPU boundary.

        Returns:
            List of parameter names in frequency (GHz) units.
        """

    @property
    def units(self: Model) -> dict[str, str]:
        """Derive units from frequency_parameters."""
        freq = set(self.frequency_parameters)
        return {p: "GHz" if p in freq else "a.u." for p in self.parameter_names}

    @property
    def parameter(self: Model) -> list[str]:
        """Get the type category for each parameter (backwards compatibility).

        Returns:
            List of parameter type strings (e.g., ['center', 'width', 'contrast', ...]).
        """
        return [self.parameter_types[p] for p in self.parameter_names]

    @abstractmethod
    def func(
        self: Model,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the model for given frequency values and parameters.

        This abstract method must be implemented by all concrete model classes.
        It defines the mathematical function that calculates the model response.

        Args:
            x: Array of frequency values in GHz.
            parameters: Array of model parameters with shape appropriate for the
                specific model.

        Returns:
            Model prediction array with shape (N, len(x)) where N is the number of
            parameter sets. Values represent normalized fluorescence intensity.
        """
        raise NotImplementedError

    def jacobian(
        self: Model,
        x: NDArray[np.floating],  # noqa: ARG002
        parameters: NDArray[np.floating],  # noqa: ARG002
    ) -> tuple[NDArray[np.floating], ...] | None:
        """Analytic partial derivatives of :meth:`func`, or None (QEP-073).

        Optional. Returning ``None`` -- the default -- makes backends fall back
        to finite differences, so custom models need not implement this. When
        implemented, the derivatives must use only framework-neutral arithmetic,
        like :meth:`func`, so one implementation serves numpy and torch.

        The columns are returned as a tuple rather than a stacked array because
        stacking would require choosing a framework; each backend stacks with
        its own.

        Args:
            x: Array of frequency values in GHz.
            parameters: Parameter array, shape (N, n_parameters).

        Returns:
            One (N, len(x)) derivative array per parameter, ordered to match
            ``parameter_names``, or None if no analytic form is available.
        """
        return None

    @property
    def n_parameters(self: Model) -> int:
        """Get the number of parameters in the model.

        Returns:
            Number of parameters.
        """
        return len(self.parameter_names)

    def get_constraint_array(self: Model, constraint: dict[str, Any]) -> NDArray[np.floating]:
        """Create an array of constraints for model parameters.

        Converts a dictionary of parameter constraints into a flattened array
        suitable for optimization algorithms. Each parameter gets two values:
        minimum and maximum bounds.

        Args:
            constraint: Dictionary mapping parameter names to [min, max] constraint
                pairs. If a parameter is not specified, infinite bounds are used.

        Returns:
            Flattened array of constraint values with shape (2 * n_parameters,).
            Organized as [param1_min, param1_max, param2_min, param2_max, ...].

        Example:
            >>> model = ESR14N()
            >>> constraints = {'center': [2.8e9, 2.9e9], 'width': [1e6, 1e7]}
            >>> bounds = model.get_constraint_array(constraints)
        """
        constraint_array = []
        for p in self.parameter_names:
            base_param = self.parameter_types[p]
            if base_param in constraint:
                constraint_array.append(constraint[base_param][0])  # Lower bound
                constraint_array.append(constraint[base_param][1])  # Upper bound
            else:
                # Default constraints if not specified
                constraint_array.append(-np.inf)  # No lower bound
                constraint_array.append(np.inf)  # No upper bound

        return np.array(constraint_array)

    def __repr__(self: Model) -> str:
        """Get a string representation of the model.

        Returns:
            String describing the model's key properties.
        """
        return f"Model({self.name}, n_parameters: {self.n_parameters}, n_peaks: {self.n_peaks})"


def resolve_analytic_jacobian_columns(
    model: Model,
    probe: Callable[[], tuple[Any, ...] | None],
    n_params: int,
    expected_shape: tuple[int, ...],
    shape_of: Callable[[Any], tuple[int, ...] | None],
) -> tuple[Any, ...] | None:
    """Probe, validate, and return :meth:`Model.jacobian`'s columns, or None.

    Shared by every :class:`~qdmpy.fitting.backends.FitBackend`'s jacobian
    resolution (QEP-073). Each backend probes ``model.jacobian`` with its own
    framework's arrays and stacks the result its own way, but the "try the
    probe, validate the column count/shape, warn and fall back to finite
    differences on any mismatch" skeleton is identical -- and had drifted
    between two independent copies (``fitting/backends.py``,
    ``fitting/torch_backend.py``) before being factored out here.

    Args:
        model: The model whose ``jacobian()`` is being probed (used only for
            its name, in log messages).
        probe: Zero-arg callable that calls ``model.jacobian(...)`` with the
            backend's own probe arrays and returns its raw result.
        n_params: Expected number of returned columns.
        expected_shape: Expected shape of each column.
        shape_of: Extracts a comparable shape tuple from one column, or
            ``None`` to reject it outright (e.g. the torch backend rejects
            anything that isn't a ``torch.Tensor`` before comparing shapes).

    Returns:
        The validated columns tuple, or ``None`` if ``jacobian()`` returned
        ``None``, raised, or returned something that doesn't match the
        contract -- in every case, the caller should fall back to finite
        differences.
    """
    try:
        cols = probe()
    except Exception as exc:  # any failure here just means "use finite differences"
        logger.warning(
            "Model '{}'.jacobian raised {!r}; falling back to finite differences",
            model.name,
            exc,
        )
        return None

    if cols is None:
        logger.debug("Model '{}' has no analytic Jacobian; using finite differences", model.name)
        return None

    if len(cols) != n_params or any(shape_of(col) != expected_shape for col in cols):
        logger.warning(
            "Model '{}'.jacobian must return {} columns of shape {}; "
            "falling back to finite differences",
            model.name,
            n_params,
            expected_shape,
        )
        return None

    return cols


class ModelRegistry:
    """Registry for managing ODMR spectral models.

    Models are registered via the ``@ModelRegistry.register`` decorator.
    The registry maps model names to their classes.

    Example:
        >>> model = ModelRegistry.get('ESR14N')
        >>> print(type(model).__name__)
        ESR14N
    """

    _registry: ClassVar[dict[str, type[Model]]] = {}

    @classmethod
    def register(cls: type[ModelRegistry], model_cls: type[Model]) -> type[Model]:
        """Register a model class (usable as a decorator).

        Args:
            model_cls: A Model subclass to register. The model's ``name``
                ClassVar is used as the registry key.

        Returns:
            The model class, unchanged.
        """
        model_name: str = model_cls.name  # type: ignore[attr-defined]
        cls._registry[model_name] = model_cls
        logger.info("Registered model: {}", model_name)
        return model_cls

    @classmethod
    def get(cls: type[ModelRegistry], name: str) -> Model:
        """Get a model instance by name.

        Args:
            name: Name of the model to retrieve.

        Returns:
            Instance of the requested model.

        Raises:
            KeyError: If the model name is not found in the registry.
        """
        if name not in cls._registry:
            error_msg = f"Model '{name}' not found in registry"
            raise KeyError(error_msg)
        logger.debug("Instantiating model: {}", name)
        return cls._registry[name]()  # type: ignore[call-arg]

    @classmethod
    def all(cls: type[ModelRegistry]) -> dict[str, type[Model]]:
        """Get all registered model classes.

        Returns:
            Dictionary mapping model names to model classes.
        """
        return cls._registry

    @classmethod
    def available_models(cls: type[ModelRegistry]) -> list[str]:
        """List all registered model names.

        Returns:
            Sorted list of model name strings (e.g., ['ESR14N', 'ESR15N', 'ESRSINGLE']).

        Example:
            >>> ModelRegistry.available_models()
            ['ESR14N', 'ESR15N', 'ESRSINGLE']
        """
        return sorted(cls._registry.keys())


@ModelRegistry.register
class ESR14N(Model):
    """Model for NV centers with 14N nitrogen isotope (3 hyperfine dips)."""

    name: ClassVar[str] = "ESR14N"

    def __init__(self: ESR14N) -> None:
        """Initialize ESR14N model with 14N-specific parameters."""
        super().__init__(
            "ESR14N",
            3,
            ["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"],
        )
        self.ahyp = AHYP_14N
        self.model_id = _GPUFIT_MODEL_ID_ESR14N

    @property
    def parameter_types(self: ESR14N) -> dict[str, str]:
        """Map each parameter to its type category."""
        return {
            "center": "center",
            "width": "width",
            "contrast_0": "contrast",
            "contrast_1": "contrast",
            "contrast_2": "contrast",
            "offset": "offset",
        }

    @property
    def frequency_parameters(self: ESR14N) -> list[str]:
        """Parameters in frequency units (GHz)."""
        return ["center"]

    def func(
        self: ESR14N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the 14N triplet Lorentzian model."""
        return esr14n(x, parameters, self.ahyp)

    def jacobian(
        self: ESR14N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], ...]:
        """Analytic derivatives of the 14N triplet model."""
        return esr14n_jacobian(x, parameters, self.ahyp)


@ModelRegistry.register
class ESR15N(Model):
    """Model for NV centers with 15N nitrogen isotope (2 hyperfine dips)."""

    name: ClassVar[str] = "ESR15N"

    def __init__(self: ESR15N) -> None:
        """Initialize ESR15N model with 15N-specific parameters."""
        super().__init__(
            "ESR15N",
            2,
            ["center", "width", "contrast_0", "contrast_1", "offset"],
        )
        self.ahyp = AHYP_15N
        self.model_id = _GPUFIT_MODEL_ID_ESR15N

    @property
    def parameter_types(self: ESR15N) -> dict[str, str]:
        """Map each parameter to its type category."""
        return {
            "center": "center",
            "width": "width",
            "contrast_0": "contrast",
            "contrast_1": "contrast",
            "offset": "offset",
        }

    @property
    def frequency_parameters(self: ESR15N) -> list[str]:
        """Parameters in frequency units (GHz)."""
        return ["center"]

    def func(
        self: ESR15N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the 15N doublet Lorentzian model."""
        return esr15n(x, parameters, self.ahyp)

    def jacobian(
        self: ESR15N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], ...]:
        """Analytic derivatives of the 15N doublet model."""
        return esr15n_jacobian(x, parameters, self.ahyp)


@ModelRegistry.register
class ESRSINGLE(Model):
    """Model for a single ODMR resonance dip (no hyperfine splitting)."""

    name: ClassVar[str] = "ESRSINGLE"

    def __init__(self: ESRSINGLE) -> None:
        """Initialize ESRSINGLE model with single-dip parameters."""
        super().__init__("ESRSINGLE", 1, ["center", "width", "contrast", "offset"])
        self.model_id = _GPUFIT_MODEL_ID_ESRSINGLE

    @property
    def parameter_types(self: ESRSINGLE) -> dict[str, str]:
        """Map each parameter to its type category."""
        return {
            "center": "center",
            "width": "width",
            "contrast": "contrast",
            "offset": "offset",
        }

    @property
    def frequency_parameters(self: ESRSINGLE) -> list[str]:
        """Parameters in frequency units (GHz)."""
        return ["center"]

    def func(
        self: ESRSINGLE,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the single Lorentzian model."""
        return esrsingle(x, parameters)

    def jacobian(
        self: ESRSINGLE,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], ...]:
        """Analytic derivatives of the single-dip model."""
        return esrsingle_jacobian(x, parameters)


def _main_demo() -> None:
    """Demo function that shows model usage when module is run as script."""
    model = ModelRegistry.get("ESRSINGLE")
    import sys

    sys.stdout.write(f"{model.n_parameters}\n")


if __name__ == "__main__":
    _main_demo()
