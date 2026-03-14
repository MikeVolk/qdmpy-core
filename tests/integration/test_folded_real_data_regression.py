"""Regression checks for folded fitting on cropped real-data fixtures."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

import qdmpy

_HAS_GPUFIT = find_spec("pygpufit.gpufit") is not None

pytestmark = pytest.mark.skipif(not _HAS_GPUFIT, reason="Requires pygpufit installation")


def _compute_folded_vs_normal_metrics(data_path: Path, model_name: str) -> dict[str, float]:
    measurement = qdmpy.load(str(data_path), bin_factor=1)

    normal = measurement.fit_odmr(model_name=model_name).fit_result
    measurement.fold_odmr()
    folded = measurement.fit_folded_odmr(model_name=model_name).fit_result

    rem_normal = normal.b111_remanent
    rem_folded = folded.b111_remanent
    ind_normal = normal.b111_induced
    ind_folded = folded.b111_induced

    rem_corr = float(np.corrcoef(rem_normal.ravel(), rem_folded.ravel())[0, 1])
    ind_corr = float(np.corrcoef(ind_normal.ravel(), ind_folded.ravel())[0, 1])
    rem_rmse = float(np.sqrt(np.mean((rem_folded - rem_normal) ** 2)))
    ind_rmse = float(np.sqrt(np.mean((ind_folded - ind_normal) ** 2)))

    return {
        "remanent_corr": rem_corr,
        "induced_corr": ind_corr,
        "remanent_rmse": rem_rmse,
        "induced_rmse": ind_rmse,
    }


def test_real_fov1_fixture_matches_expected_folded_behavior() -> None:
    """FOV1 crop remains a mostly well-behaved folded-vs-normal comparison."""
    path = Path("tests/data/real_fov1_fov2037485_x365y1061")
    metrics = _compute_folded_vs_normal_metrics(path, model_name="ESR14N")

    assert metrics["remanent_corr"] > 0.75
    assert metrics["induced_corr"] > 0.90
    assert metrics["remanent_rmse"] < 6.0
    assert metrics["induced_rmse"] < 10.0


@pytest.mark.parametrize(
    "fixture_name",
    [
        "real_fov18x_fov5838_x78y24",
        "real_fov18x_fov7539_x99y31",
        "real_fov18x_fov14925_x45y62",
    ],
)
def test_real_fov18x_fixtures_preserve_known_folded_gap(fixture_name: str) -> None:
    """FOV18x crops currently show substantial folded-vs-normal mismatch."""
    path = Path("tests/data") / fixture_name
    metrics = _compute_folded_vs_normal_metrics(path, model_name="ESR15N")

    assert metrics["induced_rmse"] > 8.0
    assert metrics["remanent_rmse"] > 3.0
    assert metrics["induced_corr"] < 0.4
