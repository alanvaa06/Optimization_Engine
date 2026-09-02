"""Chart invariants: palette safety, series folding, axis honesty."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.frontier import FrontierResult
from optimization_engine.reporting.plots import (
    OTHER_COLOR,
    PALETTE,
    fold_to_slots,
    plot_efficient_frontier,
    plot_portfolio_composition,
    plot_rolling_metrics,
    plot_weight_evolution,
    plot_weights_bar,
    series_color,
)


def _weights(n: int, n_cols: int = 1) -> pd.DataFrame:
    frame = pd.DataFrame(
        {f"s{c}": np.linspace(0.2, 0.01, n) for c in range(n_cols)},
        index=[f"A{i}" for i in range(n)],
    )
    return frame / frame.sum()


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


def test_palette_slots_are_unique():
    assert len(set(PALETTE)) == len(PALETTE)


def test_adjacent_palette_slots_are_distinguishable_under_cvd():
    """Adjacent slots are what land next to each other in a stack.

    The previous Tableau-10 order put red (#E45756) beside green (#54A24B) at
    a deuteranope ΔE of 1.2 — indistinguishable — and those two were
    neighbours in every stacked weight chart.
    """

    def srgb_to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def to_lms(hex_color: str) -> np.ndarray:
        r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        rgb = np.array([srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)])
        m = np.array(
            [
                [0.31399022, 0.63951294, 0.04649755],
                [0.15537241, 0.75789446, 0.08670142],
                [0.01775239, 0.10944209, 0.87256922],
            ]
        )
        return m @ rgb

    def deuteranope(hex_color: str) -> np.ndarray:
        lms = to_lms(hex_color)
        sim = np.array([[1, 0, 0], [0.49421, 0, 1.24827], [0, 0, 1]]) @ lms
        return sim

    worst = min(
        float(np.linalg.norm(deuteranope(a) - deuteranope(b)))
        for a, b in zip(PALETTE, PALETTE[1:])
    )
    # The old adjacent red/green pair scored essentially zero on this measure.
    assert worst > 0.02, f"adjacent pair collapses under deuteranopia ({worst:.4f})"


def test_palette_passes_the_bundled_validator_if_available():
    """Prefer the real validator over the approximation above when present."""
    script = next(
        (
            p
            for p in Path("/tmp/claude-0/bundled-skills").rglob(
                "dataviz/scripts/validate_palette.js"
            )
        ),
        None,
    )
    if script is None:
        pytest.skip("dataviz validator not available in this environment")
    result = subprocess.run(
        ["node", str(script), ",".join(PALETTE), "--mode", "light", "--surface", "#ffffff"],
        capture_output=True,
        text=True,
    )
    assert "ALL CHECKS PASS" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# Series folding
# ---------------------------------------------------------------------------


def test_small_universes_are_not_folded():
    frame = _weights(5)
    pd.testing.assert_frame_equal(fold_to_slots(frame), frame)


def test_large_universes_fold_and_preserve_the_total():
    frame = _weights(13)
    folded = fold_to_slots(frame)
    assert len(folded) == len(PALETTE)
    assert folded.index[-1].startswith("Other (")
    assert float(folded["s0"].sum()) == pytest.approx(float(frame["s0"].sum()))


def test_fold_keeps_the_largest_positions():
    frame = _weights(13)
    folded = fold_to_slots(frame)
    assert frame.index[0] in folded.index, "the largest weight was folded away"
    assert frame.index[-1] not in folded.index, "the smallest weight survived"


def test_stacked_charts_never_repeat_a_colour():
    """Cycling the palette paints two different assets the same colour."""
    fig = plot_portfolio_composition(_weights(13))
    colors = [t.marker.color for t in fig.data]
    assert len(set(colors)) == len(colors)
    assert colors[-1] == OTHER_COLOR


def test_weight_evolution_folds_across_columns():
    idx = pd.date_range("2024-01-01", periods=30)
    frame = pd.DataFrame(
        np.random.default_rng(0).random((30, 13)),
        columns=[f"A{i}" for i in range(13)],
        index=idx,
    )
    frame = frame.div(frame.sum(axis=1), axis=0)
    fig = plot_weight_evolution(frame)
    colors = [t.line.color for t in fig.data]
    assert len(set(colors)) == len(colors)
    assert colors[-1] == OTHER_COLOR


def test_other_band_takes_the_neutral_whatever_its_position():
    """The fold leaves "Other" at an arbitrary index, so colour by label."""
    assert series_color(0, "Other (6 assets)") == OTHER_COLOR
    assert series_color(7, "Other (2 assets)") == OTHER_COLOR
    assert series_color(0, "US_Equity") == PALETTE[0]
    assert series_color(99, "US_Equity") == OTHER_COLOR


# ---------------------------------------------------------------------------
# Axis honesty
# ---------------------------------------------------------------------------


def test_rolling_metrics_uses_stacked_panels_not_twin_axes():
    """A second y-axis is the most common chart mistake there is.

    A Sharpe series and an annualized-return series occupy similar numeric
    ranges, so overlaid on twin axes they trace nearly the same path and read
    as one duplicated line.
    """
    idx = pd.date_range("2020-01-01", periods=600)
    returns = pd.Series(np.random.default_rng(1).normal(0.0004, 0.01, 600), index=idx)
    from optimization_engine.analytics.performance import rolling_metrics

    fig = plot_rolling_metrics(rolling_metrics(returns, 252))
    layout = fig.layout.to_plotly_json()
    axes = {k: v for k, v in layout.items() if k.startswith("yaxis")}
    assert len(axes) >= 3, "expected one panel per metric"
    for name, axis in axes.items():
        assert "overlaying" not in axis, f"{name} is a second scale on another axis"


def test_long_and_short_bars_use_the_most_separated_slots():
    w = pd.Series({"a": 0.6, "b": -0.2, "c": 0.6})
    colors = plot_weights_bar(w).data[0].marker.color
    assert set(colors) == {PALETTE[0], PALETTE[7]}


# ---------------------------------------------------------------------------
# Frontier: a missing anchor is explained, and the dominated branch is gone
# ---------------------------------------------------------------------------


def _frontier(anchor_failures=None, tangency=True) -> FrontierResult:
    """A three-point frontier, solved, with both anchors unless told otherwise."""
    summary = pd.DataFrame(
        {
            "target": [0.04, 0.06, 0.08],
            "expected_return": [0.04, 0.06, 0.08],
            "expected_volatility": [0.10, 0.12, 0.16],
            "sharpe_ratio": [0.40, 0.50, 0.50],
            "is_efficient": [True, True, True],
            "status": ["ok", "ok", "ok"],
        }
    )
    weights = pd.DataFrame(
        np.full((2, 3), 0.5), index=["A", "B"], columns=summary["target"].values
    )
    return FrontierResult(
        summary=summary,
        weights=weights,
        min_variance=pd.Series(
            {
                "label": "Minimum variance",
                "expected_return": 0.04,
                "expected_volatility": 0.10,
                "sharpe_ratio": 0.40,
            }
        ),
        tangency=(
            pd.Series(
                {
                    "label": "Maximum Sharpe",
                    "expected_return": 0.06,
                    "expected_volatility": 0.12,
                    "sharpe_ratio": 0.50,
                }
            )
            if tangency
            else None
        ),
        anchor_failures=anchor_failures or {},
    )


def test_frontier_footnotes_a_failed_anchor():
    """An anchor that did not solve is named on the chart, not just omitted.

    A marker that is missing because the solve failed and a marker that was
    never requested look identical. The footnote is the difference.
    """
    reason = "The problem is infeasible: no allocation satisfies every constraint."
    fig = plot_efficient_frontier(
        _frontier(anchor_failures={"tangency": reason}, tangency=False)
    )

    notes = list(fig.layout.annotations)
    assert len(notes) == 1, notes
    note = notes[0]
    # The field name is "tangency"; an analyst reads "Maximum Sharpe".
    assert "Maximum Sharpe" in note.text
    assert "infeasible" in note.text
    assert (note.xref, note.yref) == ("paper", "paper")
    # The legend sits at y=-0.18, so the footnote has to clear it.
    assert note.y < fig.layout.legend.y
    # And the figure has to leave room for it.
    assert fig.layout.margin.b >= 110

    # The anchor that *did* solve is still drawn.
    assert "Minimum variance" in {t.name for t in fig.data}
    assert "Maximum Sharpe" not in {t.name for t in fig.data}


def test_frontier_without_failed_anchors_has_no_footnote():
    fig = plot_efficient_frontier(_frontier())
    assert not fig.layout.annotations
    assert {"Minimum variance", "Maximum Sharpe"} <= {t.name for t in fig.data}


def test_frontier_no_longer_draws_a_dominated_branch():
    """``show_dominated`` is a deprecated no-op; the branch cannot occur.

    The mean-variance return target is a floor, so a target below the
    minimum-variance return returns the minimum-variance portfolio rather than
    a dominated one. The trace is gone. The keyword is still accepted so
    existing callers — the Streamlit app, the example notebook — keep working.
    """
    frontier = _frontier()
    # Even handed a frame that claims a dominated point, nothing is drawn for
    # it: the trace no longer exists.
    frontier.summary.loc[0, "is_efficient"] = False
    fig = plot_efficient_frontier(frontier, show_dominated=True)
    assert not any("Dominated" in str(t.name) for t in fig.data)
