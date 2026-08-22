"""PyTorch fit backend: architecture-independent batched Levenberg-Marquardt.

QEP-069. Runs the same batched-LM code path on NVIDIA CUDA, Apple-silicon
MPS, and CPU. All fits in a chunk advance in parallel as a batch dimension,
so a production frame is a handful of large tensor ops per LM iteration
instead of millions of per-pixel optimizer calls (the ScipyBackend regime).

``torch`` is imported only inside method bodies — importing this module (or
qdmpy itself) never pays the 1-5 s torch import cost, mirroring how
``GpufitBackend`` isolates pygpufit.
"""

from __future__ import annotations

import importlib.util
import time
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.exceptions import DependencyError
from qdmpy.fitting.backends import (
    BackendFitOutput,
    FitBackendOptions,
    bounds_from_constraints,
)
from qdmpy.fitting.models import Model

if TYPE_CHECKING:
    import torch as torch_types  # ty: ignore[unresolved-import]

_LAMBDA_INIT = 1e-3
_LAMBDA_UP = 10.0
# Gentler relaxation than the classic /10: benchmarked 2.8x fewer iterations
# on ESR14N frames (aggressive relaxation overshoots and oscillates
# accept/reject near the optimum) at identical final chi2.
_LAMBDA_DOWN = 3.0
_LAMBDA_MAX = 1e7
_FD_EPS = 3.45e-4  # sqrt(float32 machine epsilon)
# Absolute floor on the finite-difference step. A relative-only step dies for
# parameters near zero (e.g. offset ~1e-4): h = 3.45e-4 * 1e-4 ~ 3e-8 changes
# a model value of ~1.0 by less than one float32 ulp, so f(p+h) == f(p)
# bitwise and that Jacobian column is exactly zero — the parameter can never
# move. 1e-5 is far below every physical parameter scale in this domain
# (centers ~2.87 GHz, widths ~2e-3 GHz, contrasts ~0.1, offsets ~1e-2) while
# guaranteeing a representable model change.
_FD_MIN_STEP = 1e-5
_DIAG_FLOOR = 1e-20

_STATE_CONVERGED = 0
_STATE_MAX_ITERATIONS = 1
_STATE_INVALID = 2

# Host-side convergence checks force a full GPU pipeline sync; amortize them
# over this many LM iterations (masked freezing in between).
_COMPACT_EVERY = 8

# A fit that rejects this many consecutive trial steps is at its float32
# optimum (lambda has grown 10^10-fold with no acceptable step) — declare it
# converged instead of burning iterations to the max_number_iterations cap.
_MAX_REJECT_STREAK = 10

_INSTALL_HINT = "Install the GPU extra: `uv sync --extra gpu` (or `pip install 'qdmpy[gpu]'`)."


def torch_gpu_device_available() -> bool:
    """Return True if torch is installed and a real GPU device (cuda/mps) exists.

    Used by the ``'auto'`` backend resolution: torch is only auto-selected
    when it can actually use a GPU — torch-CPU stays an explicit opt-in.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    import torch  # ty: ignore[unresolved-import]

    return torch.cuda.is_available() or torch.backends.mps.is_available()


class TorchBackend:
    """Batched Levenberg-Marquardt fitting via PyTorch (cuda / mps / cpu).

    One numeric code path (float32) for every device, so the CPU-device test
    run certifies the exact code the GPU executes. Fits are chunked to bound
    GPU memory; within a chunk, every pixel's LM iteration advances in
    parallel via batched tensor ops (`cholesky_ex` normal-equation solves,
    finite-difference Jacobians).

    Any model whose ``func`` uses only framework-neutral operations
    (arithmetic, broadcasting, slicing — see the ``Model`` custom-model
    contract) is supported, including pure-Python custom models that
    gpufit cannot fit.
    """

    name = "torch"
    install_hint = _INSTALL_HINT

    def __init__(
        self: Self,
        device: str | None = None,
        chunk_size: int = 262_144,
    ) -> None:
        """Configure the backend; no torch import happens here.

        Args:
            device: 'cuda', 'mps', 'cpu', or None/'auto' to pick the best
                available device at fit time (cuda > mps > cpu).
            chunk_size: Fits processed per GPU batch. The default (2^18)
                peaks around 0.7 GB of device memory at 50 frequencies and
                6 parameters; lower it on small GPUs.
        """
        self._device_spec = device
        self._chunk_size = chunk_size
        self._solve_on_cpu = False

    def is_available(self: Self) -> bool:
        """Return True if torch is installed (checked without importing it)."""
        return importlib.util.find_spec("torch") is not None

    def supports(self: Self, model: Model) -> bool:  # noqa: ARG002
        """Support any model; framework-neutrality is verified at first eval.

        A capability probe here would import torch on a hot config path and
        could not distinguish "chokes on tensors" from any other model bug —
        the guarded first evaluation in ``fit()`` gives a better error at
        effectively the same time.
        """
        return True

    def fit(
        self: Self,
        data: NDArray,
        freq_ghz: NDArray,
        initial_parameters: NDArray,
        constraints: NDArray,
        constraint_types: NDArray,
        model: Model,
        options: FitBackendOptions,
    ) -> BackendFitOutput:
        """Fit flattened data with batched LM on the resolved torch device.

        Raises:
            DependencyError: If torch is not installed, the requested device
                is unavailable, or the model is not framework-neutral.
        """
        torch = self._import_torch()
        device = self._resolve_device(torch)
        # MPS "supports" batched Cholesky but runs it ~80x slower than a CPU
        # round-trip of the small (c, p, p) systems; route the solve to CPU
        # there from the start (model evals and Jacobians stay on the GPU).
        self._solve_on_cpu = device.type == "mps"

        if options.estimator != "LSE":
            logger.warning(
                "TorchBackend only supports least-squares (LSE); ignoring estimator={!r}",
                options.estimator,
            )

        n_freqs = data.shape[-1]
        data_2d = np.ascontiguousarray(data, dtype=np.float32).reshape((-1, n_freqs))
        n_fits = data_2d.shape[0]
        n_params = initial_parameters.shape[-1]
        initial_2d = np.ascontiguousarray(initial_parameters, dtype=np.float32).reshape(
            (n_fits, n_params)
        )
        lower, upper = bounds_from_constraints(constraints, constraint_types, n_params)
        lower32 = lower.astype(np.float32)
        upper32 = upper.astype(np.float32)
        freq32 = np.ascontiguousarray(freq_ghz, dtype=np.float32).reshape(-1)

        params_out = np.empty((n_fits, n_params), dtype=np.float32)
        states_out = np.empty(n_fits, dtype=np.int32)
        chi2_out = np.empty(n_fits, dtype=np.float32)
        iters_out = np.empty(n_fits, dtype=np.int32)

        logger.info(
            "TorchBackend fitting {} spectra on device '{}' (chunk_size={})",
            n_fits,
            device,
            self._chunk_size,
        )
        start = time.perf_counter()
        x_t = torch.from_numpy(freq32).to(device)

        for chunk_start in range(0, n_fits, self._chunk_size):
            sl = slice(chunk_start, min(chunk_start + self._chunk_size, n_fits))
            p, states, chi2, iters = self._fit_chunk(
                torch,
                device,
                x_t,
                y=torch.from_numpy(data_2d[sl]).to(device),
                p0=torch.from_numpy(initial_2d[sl]).to(device),
                lo=torch.from_numpy(lower32[sl]).to(device),
                hi=torch.from_numpy(upper32[sl]).to(device),
                model=model,
                options=options,
            )
            params_out[sl] = p.cpu().numpy()
            states_out[sl] = states.cpu().numpy()
            chi2_out[sl] = chi2.cpu().numpy()
            iters_out[sl] = iters.cpu().numpy()

        return BackendFitOutput(
            parameters=params_out,
            states=states_out,
            chi2=chi2_out,
            iterations=iters_out,
            execution_time=time.perf_counter() - start,
        )

    @staticmethod
    def _import_torch() -> Any:  # noqa: ANN401
        """Import torch lazily, raising DependencyError with the install hint."""
        try:
            import torch  # ty: ignore[unresolved-import]
        except ImportError as exc:
            msg = f"torch is required for the 'torch' backend but is not installed. {_INSTALL_HINT}"
            raise DependencyError(msg) from exc
        return torch

    def _resolve_device(self: Self, torch: Any) -> torch_types.device:  # noqa: ANN401
        """Resolve the configured device spec to a concrete torch.device."""
        spec = self._device_spec
        if spec is None or spec == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if spec.startswith("cuda") and not torch.cuda.is_available():
            msg = "TorchBackend device 'cuda' was requested but CUDA is not available"
            raise DependencyError(msg)
        if spec == "mps" and not torch.backends.mps.is_available():
            msg = "TorchBackend device 'mps' was requested but MPS is not available"
            raise DependencyError(msg)
        return torch.device(spec)

    def _eval_model(
        self: Self,
        torch: Any,  # noqa: ANN401
        model: Model,
        x: torch_types.Tensor,
        p: torch_types.Tensor,
    ) -> torch_types.Tensor:
        """Evaluate ``model.func`` on tensors, guarding the neutrality contract."""
        try:
            # func is annotated for numpy but the framework-neutral contract
            # (Model docstring) guarantees tensors work via duck typing.
            out = model.func(cast(NDArray, x), cast(NDArray, p))
        except Exception as exc:
            msg = (
                f"Model '{model.name}'.func failed when called with torch tensors ({exc!r}). "
                "Torch-fittable models must use only framework-neutral arithmetic "
                "(see the Model custom-model contract); use backend='scipy' for "
                "numpy-only models."
            )
            raise DependencyError(msg) from exc
        if not isinstance(out, torch.Tensor):
            msg = (
                f"Model '{model.name}'.func returned {type(out).__name__} instead of a "
                "torch tensor — it silently converted to numpy (e.g. via an np.* call). "
                "Use only framework-neutral arithmetic, or fit with backend='scipy'."
            )
            raise DependencyError(msg)
        return out

    def _solve_damped(
        self: Self,
        torch: Any,  # noqa: ANN401
        a_mat: torch_types.Tensor,
        rhs: torch_types.Tensor,
    ) -> tuple[torch_types.Tensor, torch_types.Tensor]:
        """Batched Cholesky solve of a_mat @ delta = rhs; per-fit failure codes.

        Falls back to solving on CPU (small (c, p, p) tensors only) if the
        device lacks batched Cholesky support — model evaluations and the
        Jacobian stay on the GPU either way.
        """
        if not self._solve_on_cpu:
            try:
                chol, info = torch.linalg.cholesky_ex(a_mat)
                delta = torch.cholesky_solve(rhs, chol)
            except (NotImplementedError, RuntimeError):
                self._solve_on_cpu = True
                logger.debug(
                    "Batched Cholesky unsupported on {}; solving normal equations on CPU",
                    a_mat.device,
                )
            else:
                return delta, info
        chol, info = torch.linalg.cholesky_ex(a_mat.cpu())
        delta = torch.cholesky_solve(rhs.cpu(), chol)
        return delta.to(a_mat.device), info.to(a_mat.device)

    def _fd_jacobian(
        self: Self,
        torch: Any,  # noqa: ANN401
        model: Model,
        x_t: torch_types.Tensor,
        p_w: torch_types.Tensor,
        f0_w: torch_types.Tensor,
        eye_cols: torch_types.Tensor,
    ) -> torch_types.Tensor:
        """Forward-difference Jacobian (a, f, p) in one stacked model call.

        All n_params perturbed parameter sets are flattened into a single
        (n_params * a, n_params) batch: one kernel dispatch instead of
        n_params — kernel-launch overhead, not FLOPs, is the bottleneck on
        MPS. The step has an absolute floor (see ``_FD_MIN_STEP``) so
        near-zero parameters keep a live Jacobian column.
        """
        n_active, n_params = p_w.shape
        step = torch.clamp(_FD_EPS * p_w.abs(), min=_FD_MIN_STEP)  # (a, p)
        p_pert = p_w.unsqueeze(0).expand(n_params, -1, -1) + (
            step.mT.unsqueeze(-1) * eye_cols.unsqueeze(1)
        )  # (p, a, p): slab j perturbs parameter j only
        f_pert = self._eval_model(
            torch, model, x_t, p_pert.reshape(n_params * n_active, n_params)
        ).reshape(n_params, n_active, -1)
        return ((f_pert - f0_w.unsqueeze(0)) / step.mT.unsqueeze(-1)).permute(1, 2, 0)

    def _fit_chunk(  # noqa: PLR0915 — one cohesive LM loop; splitting it would scatter tightly coupled optimizer state
        self: Self,
        torch: Any,  # noqa: ANN401
        device: torch_types.device,
        x_t: torch_types.Tensor,
        *,
        y: torch_types.Tensor,
        p0: torch_types.Tensor,
        lo: torch_types.Tensor,
        hi: torch_types.Tensor,
        model: Model,
        options: FitBackendOptions,
    ) -> tuple[torch_types.Tensor, ...]:
        """Run batched LM on one chunk; returns (params, states, chi2, iterations).

        Converged fits are *compacted out* of the working set at window
        boundaries (index-select of the survivors), so per-iteration cost
        tracks the number of still-active fits instead of the chunk size.
        Without this, a small slow-converging tail forces the entire chunk
        through every LM iteration — orders of magnitude wasted at
        production scale.
        """
        c, n_params = p0.shape

        # gpufit tolerates out-of-bounds initial guesses; clamp like ScipyBackend
        p = torch.clamp(p0, lo, hi)
        f0 = self._eval_model(torch, model, x_t, p)
        r = f0 - y
        chi2 = torch.nansum(r * r, dim=-1)
        finite = torch.isfinite(chi2) & torch.isfinite(y).all(dim=-1)

        # Result buffers, prefilled with the initial state; rows are finalized
        # as their fits converge or when iterations run out.
        p_out = p.clone()
        chi2_out = chi2.clone()
        states_out = torch.full((c,), _STATE_MAX_ITERATIONS, dtype=torch.int32, device=device)
        states_out[~finite] = _STATE_INVALID
        iters_out = torch.zeros(c, dtype=torch.int32, device=device)

        # Working set: indices into the chunk that are still being fitted.
        # Convergence checks and compaction happen only every
        # _COMPACT_EVERY iterations: each host-side check forces a full GPU
        # pipeline sync (~ms on MPS), so doing it per iteration makes the
        # device slower than the CPU. Within a window, converged fits are
        # frozen by masking; at the window boundary they are written out and
        # compacted away in one sync.
        idx = torch.nonzero(finite).squeeze(-1)
        y_w, p_w, f0_w, r_w = y[idx], p[idx], f0[idx], r[idx]
        chi2_w, lo_w, hi_w = chi2[idx], lo[idx], hi[idx]
        lam = torch.full((idx.numel(),), _LAMBDA_INIT, device=device)
        conv_w = torch.zeros(idx.numel(), dtype=torch.bool, device=device)
        reject_streak = torch.zeros(idx.numel(), dtype=torch.int32, device=device)
        eye_cols = torch.eye(n_params, device=device)

        it = 0
        while it < options.max_number_iterations and idx.numel():
            for _ in range(_COMPACT_EVERY):
                if it >= options.max_number_iterations:
                    break
                it += 1
                alive = ~conv_w
                iters_out.index_add_(0, idx, alive.to(torch.int32))
                n_active = idx.numel()

                jac = self._fd_jacobian(torch, model, x_t, p_w, f0_w, eye_cols)

                # Active-set reduction for box constraints: a parameter pinned
                # at a bound whose gradient points out of the box would make
                # every projected trial step uphill (reject forever). Zeroing
                # its Jacobian column removes it from this iteration's step so
                # the free parameters still optimize; it re-enters once the
                # gradient flips.
                grad = (jac.mT @ r_w.unsqueeze(-1)).squeeze(-1)  # (a, p) = ∇½chi2
                pinned = ((p_w <= lo_w) & (grad > 0)) | ((p_w >= hi_w) & (grad < 0))
                jac = jac * (~pinned).unsqueeze(1)

                jtj = jac.mT @ jac  # (a, p, p)
                jtr = jac.mT @ r_w.unsqueeze(-1)  # (a, p, 1)
                diag = torch.clamp(torch.diagonal(jtj, dim1=-2, dim2=-1), min=_DIAG_FLOOR)
                a_mat = jtj + lam.view(n_active, 1, 1) * torch.diag_embed(diag)

                delta, info = self._solve_damped(torch, a_mat, -jtr)
                p_trial = torch.clamp(p_w + delta.squeeze(-1), lo_w, hi_w)
                f_trial = self._eval_model(torch, model, x_t, p_trial)
                r_trial = f_trial - y_w
                chi2_trial = torch.sum(r_trial * r_trial, dim=-1)

                # Non-strict acceptance: at the float32 chi2 floor, trial steps
                # give bit-identical chi2 — with strict `<` those fits reject
                # forever and never hit the accepted-step convergence check
                # below. Accepting equality makes delta-chi2 == 0 converge
                # immediately instead.
                accept = alive & (info == 0) & torch.isfinite(chi2_trial) & (chi2_trial <= chi2_w)
                reject_streak = torch.where(
                    accept | conv_w,
                    torch.zeros_like(reject_streak),
                    reject_streak + alive.to(torch.int32),
                )
                # Three convergence routes, all requiring an accepted step:
                # 1. bitwise-equal chi2: the fit is at its float32 floor — no
                #    progress is representable, so lambda is irrelevant;
                # 2. small-but-nonzero delta-chi2, only in the Gauss-Newton
                #    regime (lambda at/below init) — at large lambda steps are
                #    artificially tiny and delta-chi2 < tol would fire far
                #    from the optimum;
                # 3. stall terminator: a long unbroken rejection streak means
                #    no trial improves chi2 by even one ulp — convergence,
                #    not failure.
                at_floor = accept & (chi2_trial == chi2_w)
                small_gn_step = (
                    accept
                    & (lam <= _LAMBDA_INIT)
                    & (
                        (chi2_w - chi2_trial).abs()
                        < options.tolerance * torch.clamp(chi2_trial, min=1.0)
                    )
                )
                stalled = alive & (reject_streak >= _MAX_REJECT_STREAK)
                conv_w = conv_w | at_floor | small_gn_step | stalled

                accept_col = accept.unsqueeze(-1)
                p_w = torch.where(accept_col, p_trial, p_w)
                f0_w = torch.where(accept_col, f_trial, f0_w)
                r_w = torch.where(accept_col, r_trial, r_w)
                chi2_w = torch.where(accept, chi2_trial, chi2_w)
                lam_next = torch.where(
                    accept,
                    lam / _LAMBDA_DOWN,
                    torch.clamp(lam * _LAMBDA_UP, max=_LAMBDA_MAX),
                )
                lam = torch.where(alive, lam_next, lam)

            # Window boundary: single host sync — write out and drop converged
            if bool(conv_w.any()):
                done = idx[conv_w]
                p_out[done] = p_w[conv_w]
                chi2_out[done] = chi2_w[conv_w]
                states_out[done] = _STATE_CONVERGED
                keep = ~conv_w
                idx = idx[keep]
                y_w, p_w, f0_w, r_w = y_w[keep], p_w[keep], f0_w[keep], r_w[keep]
                chi2_w, lo_w, hi_w = chi2_w[keep], lo_w[keep], hi_w[keep]
                lam, reject_streak = lam[keep], reject_streak[keep]
                conv_w = torch.zeros(idx.numel(), dtype=torch.bool, device=device)

        if idx.numel():  # ran out of iterations: report best-so-far
            p_out[idx] = p_w
            chi2_out[idx] = chi2_w

        return p_out, states_out, chi2_out, iters_out
