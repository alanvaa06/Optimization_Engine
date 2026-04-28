"""Tests for project_to_bounds_iterated."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.optimizers._bounds import (
    InfeasibleBoundsError,
    project_to_bounds_iterated,
)


def test_already_feasible_input_unchanged():
    w = np.array([0.3, 0.3, 0.4])
    out = project_to_bounds_iterated(w, np.zeros(3), np.ones(3))
    np.testing.assert_allclose(out, w, atol=1e-10)


def test_residual_distributed_over_slack():
    w = np.array([0.5, 0.5, 0.5])  # sums to 1.5, must come down to 1.0
    out = project_to_bounds_iterated(w, np.zeros(3), np.full(3, 0.6))
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0
    assert (out >= -1e-9).all() and (out <= 0.6 + 1e-9).all()


def test_clip_then_rescale_breaks_bounds_but_iterated_does_not():
    # A naive clip(0,0.4) + rescale would push the first weight back over 0.4.
    w = np.array([0.9, 0.05, 0.05])
    lb = np.zeros(3)
    ub = np.array([0.4, 1.0, 1.0])
    out = project_to_bounds_iterated(w, lb, ub)
    assert (out <= ub + 1e-9).all()
    assert (out >= lb - 1e-9).all()
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0


def test_infeasible_lb_sum_raises():
    with pytest.raises(InfeasibleBoundsError):
        project_to_bounds_iterated(
            np.array([0.5, 0.5, 0.5]),
            np.full(3, 0.5),
            np.ones(3),
        )


def test_infeasible_ub_sum_raises():
    with pytest.raises(InfeasibleBoundsError):
        project_to_bounds_iterated(
            np.array([0.1, 0.1, 0.1]),
            np.zeros(3),
            np.full(3, 0.2),
        )


def test_lb_greater_than_ub_raises_valueerror():
    with pytest.raises(ValueError, match="lb must be"):
        project_to_bounds_iterated(
            np.array([0.5, 0.5]),
            np.array([0.6, 0.0]),
            np.array([0.4, 1.0]),
        )


def test_negative_residual_distributed_over_slack_above_lb():
    # w sums to 0.5 (deficit), pushed up via lower-side slack
    w = np.array([0.1, 0.1, 0.3])
    lb = np.array([0.05, 0.05, 0.05])
    ub = np.full(3, 1.0)
    out = project_to_bounds_iterated(w, lb, ub)
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0
    assert (out >= lb - 1e-9).all()
