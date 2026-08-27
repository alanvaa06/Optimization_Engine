"""Random-matrix denoising and detoning.

The tests are written against properties that must hold rather than against
recorded numbers, so they keep their meaning if the kernel bandwidth or the
optimizer inside the Marchenko-Pastur fit changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.data.denoise import (
    cov_to_corr,
    denoise_correlation,
    denoise_covariance,
    detone_correlation,
    fit_marchenko_pastur,
    marchenko_pastur_pdf,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=11))


def _factor_correlation(n_assets: int, n_obs: int, n_factors: int, seed: int):
    """A correlation matrix with a known number of real factors."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_assets, n_factors))
    factors = rng.normal(size=(n_obs, n_factors))
    noise = rng.normal(size=(n_obs, n_assets)) * 1.5
    data = factors @ loadings.T + noise
    frame = pd.DataFrame(data, columns=[f"a{i}" for i in range(n_assets)])
    return frame.corr(), frame


# ---------------------------------------------------------------------------
# The Marchenko-Pastur law itself
# ---------------------------------------------------------------------------


def test_marchenko_pastur_density_integrates_to_one():
    pdf = marchenko_pastur_pdf(1.0, q=10.0, n_points=20_000)
    # Trapezoid by hand: numpy renamed ``trapz`` to ``trapezoid`` in 2.0 and
    # this suite runs on both.
    x, y = pdf.index.values, pdf.values
    area = float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))
    assert area == pytest.approx(1.0, abs=5e-3)


def test_marchenko_pastur_support_widens_as_the_sample_shortens():
    wide = marchenko_pastur_pdf(1.0, q=2.0)
    narrow = marchenko_pastur_pdf(1.0, q=50.0)
    assert (wide.index.max() - wide.index.min()) > (
        narrow.index.max() - narrow.index.min()
    )


def test_marchenko_pastur_needs_more_observations_than_assets():
    with pytest.raises(ValueError, match="T/N"):
        marchenko_pastur_pdf(1.0, q=0.5)


def test_pure_noise_spectrum_fits_a_variance_near_one():
    rng = np.random.default_rng(3)
    n_assets, n_obs = 60, 600
    data = pd.DataFrame(
        rng.normal(size=(n_obs, n_assets)),
        columns=[f"a{i}" for i in range(n_assets)],
    )
    eigenvalues = np.linalg.eigvalsh(data.corr().values)
    variance, cutoff = fit_marchenko_pastur(eigenvalues, q=n_obs / n_assets)
    # With no signal at all, essentially all the variance is noise.
    assert variance > 0.9
    # And essentially every eigenvalue falls under the cutoff.
    assert (eigenvalues > cutoff).sum() <= 2


def test_signal_eigenvalues_are_detected_when_factors_are_present():
    corr, _ = _factor_correlation(n_assets=40, n_obs=800, n_factors=3, seed=5)
    eigenvalues = np.linalg.eigvalsh(corr.values)
    _, cutoff = fit_marchenko_pastur(eigenvalues, q=800 / 40)
    detected = int((eigenvalues > cutoff).sum())
    assert 1 <= detected <= 6, f"expected roughly 3 factors, detected {detected}"


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


def test_denoising_improves_conditioning_and_keeps_a_correlation():
    corr, _ = _factor_correlation(n_assets=40, n_obs=200, n_factors=3, seed=9)
    denoised, report = denoise_correlation(corr.values, q=200 / 40)

    assert np.allclose(np.diag(denoised), 1.0)
    assert np.allclose(denoised, denoised.T)
    assert report.condition_after < report.condition_before
    assert 1 <= report.n_signal_eigenvalues < corr.shape[0]


def test_denoising_preserves_the_trace_of_the_correlation():
    """Constant-residual denoising redistributes eigenvalues, it does not add.

    The trace of a correlation matrix is N by definition, so preserving it is
    the check that no variance was invented or destroyed.
    """
    corr, _ = _factor_correlation(n_assets=30, n_obs=300, n_factors=2, seed=13)
    denoised, _ = denoise_correlation(corr.values, q=10.0)
    assert np.trace(denoised) == pytest.approx(30.0)


def test_denoising_leaves_the_signal_eigenvectors_alone():
    corr, _ = _factor_correlation(n_assets=40, n_obs=400, n_factors=2, seed=17)
    denoised, report = denoise_correlation(corr.values, q=10.0)

    _, before = np.linalg.eigh(corr.values)
    _, after = np.linalg.eigh(denoised)
    # eigh returns ascending eigenvalues, so the leading eigenvector is last.
    leading_before = before[:, -1]
    leading_after = after[:, -1]
    alignment = abs(float(leading_before @ leading_after))
    assert alignment > 0.99, "denoising rotated the dominant factor"
    assert report.n_signal_eigenvalues >= 1


def test_targeted_shrinkage_with_alpha_one_is_a_no_op():
    corr, _ = _factor_correlation(n_assets=20, n_obs=400, n_factors=2, seed=21)
    denoised, _ = denoise_correlation(
        corr.values, q=20.0, method="targeted_shrinkage", alpha=1.0
    )
    assert np.allclose(denoised, corr.values, atol=1e-8)


def test_targeted_shrinkage_is_gentler_than_constant_residual():
    corr, _ = _factor_correlation(n_assets=30, n_obs=300, n_factors=2, seed=23)
    hard, _ = denoise_correlation(corr.values, q=10.0)
    soft, _ = denoise_correlation(
        corr.values, q=10.0, method="targeted_shrinkage", alpha=0.5
    )
    assert np.linalg.norm(soft - corr.values) < np.linalg.norm(hard - corr.values)


def test_unknown_denoise_method_is_rejected():
    corr, _ = _factor_correlation(n_assets=10, n_obs=200, n_factors=1, seed=1)
    with pytest.raises(ValueError, match="Unknown denoising method"):
        denoise_correlation(corr.values, q=20.0, method="wishful")


# ---------------------------------------------------------------------------
# Detoning
# ---------------------------------------------------------------------------


def test_detoning_removes_the_market_factor_and_leaves_it_singular():
    corr, _ = _factor_correlation(n_assets=25, n_obs=500, n_factors=1, seed=29)
    detoned = detone_correlation(corr.values, n_factors=1)

    assert np.allclose(np.diag(detoned), 1.0)
    eigenvalues = np.linalg.eigvalsh(detoned)
    assert eigenvalues.min() < 1e-8, "the removed factor should leave a null space"
    # The average pairwise correlation must fall: that is the whole point.
    off = ~np.eye(25, dtype=bool)
    assert abs(detoned[off].mean()) < abs(corr.values[off].mean())


def test_detoning_rejects_removing_every_factor():
    corr, _ = _factor_correlation(n_assets=8, n_obs=200, n_factors=1, seed=31)
    with pytest.raises(ValueError, match="Can remove between"):
        detone_correlation(corr.values, n_factors=8)


# ---------------------------------------------------------------------------
# Covariance-level integration
# ---------------------------------------------------------------------------


def test_denoise_covariance_preserves_the_volatilities(returns: pd.DataFrame):
    cov = covariance_matrix(returns, method="sample")
    denoised, _ = denoise_covariance(cov, n_observations=len(returns))
    assert np.allclose(np.diag(denoised.values), np.diag(cov.values))
    assert list(denoised.columns) == list(cov.columns)


def test_denoise_covariance_refuses_a_degenerate_sample():
    frame = pd.DataFrame(
        np.random.default_rng(0).normal(size=(5, 8)),
        columns=[f"a{i}" for i in range(8)],
    )
    cov = frame.cov()
    with pytest.raises(ValueError, match="more observations than assets"):
        denoise_covariance(cov, n_observations=5)


def test_covariance_matrix_denoised_method_attaches_its_report(returns: pd.DataFrame):
    cov = covariance_matrix(returns, method="denoised")
    report = cov.attrs.get("denoise_report")
    assert report is not None
    assert report.n_assets == returns.shape[1]
    assert "Marchenko-Pastur" in report.describe()


def test_detone_flag_makes_the_covariance_singular(returns: pd.DataFrame):
    with pytest.warns(UserWarning, match="singular by construction"):
        cov = covariance_matrix(returns, method="denoised", detone=1)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert eigenvalues.min() < 1e-12


def test_denoising_is_optional_and_off_by_default(returns: pd.DataFrame):
    plain = covariance_matrix(returns, method="sample")
    assert "denoise_report" not in plain.attrs


def test_cov_to_corr_rejects_a_constant_series():
    cov = np.array([[0.04, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="zero variance"):
        cov_to_corr(cov)
