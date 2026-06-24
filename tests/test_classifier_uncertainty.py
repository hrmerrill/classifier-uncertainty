"""Tests for classifier_uncertainty."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before any pyplot import

import numpy as np
import pytest

import classifier_uncertainty
from classifier_uncertainty import BinaryClassifier


def test_import():
    """Package imports without error."""
    assert classifier_uncertainty.__doc__ is not None


def test_from_cm_tpr_matches_paper():
    """Classifier 7a from Tötsch & Hoffmann: TPR CI should span ~89%–100%."""
    bc = BinaryClassifier.from_cm(tp=26, fn=0, tn=6, fp=2, seed=0)
    result = bc.at_threshold().tpr()
    lo, hi = result.credible_interval(0.95)
    assert lo > 0.85
    assert hi > 0.98
    assert result.point_estimate > 0.92


def test_metric_uncertainty_shrinks_with_n():
    """Larger sample size reduces metric uncertainty."""
    small = BinaryClassifier.from_cm(tp=5, fn=1, tn=6, fp=2, seed=0)
    large = BinaryClassifier.from_cm(tp=50, fn=10, tn=60, fp=20, seed=0)
    assert (
        small.at_threshold().accuracy().metric_uncertainty
        > large.at_threshold().accuracy().metric_uncertainty
    )


def test_aliases_return_same_samples():
    """tpr/sensitivity/recall and tnr/specificity return identical sample arrays."""
    bc = BinaryClassifier.from_cm(tp=26, fn=0, tn=6, fp=2, seed=42)
    t = bc.at_threshold()
    np.testing.assert_array_equal(t.tpr().samples, t.sensitivity().samples)
    np.testing.assert_array_equal(t.tpr().samples, t.recall().samples)
    np.testing.assert_array_equal(t.tnr().samples, t.specificity().samples)
    np.testing.assert_array_equal(t.precision().samples, t.ppv().samples)


def test_custom_metric():
    """Custom lambda produces valid distribution in [0, 1]."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3, seed=0)
    result = bc.at_threshold().metric(lambda tp, fn, tn, fp: tp / (tp + fp))
    assert result.samples.shape == (20_000,)
    assert 0.0 < result.point_estimate < 1.0


def test_credible_interval_width_increases_with_level():
    """Wider confidence level produces wider HPDI."""
    bc = BinaryClassifier.from_cm(tp=10, fn=5, tn=8, fp=3, seed=0)
    result = bc.at_threshold().tpr()
    lo_90, hi_90 = result.credible_interval(0.90)
    lo_95, hi_95 = result.credible_interval(0.95)
    assert (hi_95 - lo_95) > (hi_90 - lo_90)


def test_from_cm_default_threshold():
    """at_threshold() with no argument uses threshold=0.5 (or fixed CM)."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3)
    result = bc.at_threshold()
    assert 0.0 < result.accuracy().point_estimate < 1.0


def test_roc_auc_in_bounds():
    """ROC AUC samples lie in [0, 1]."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100).astype(bool)
    y_score = rng.uniform(0, 1, 100)
    auc = BinaryClassifier(y_true, y_score, seed=42).roc_curve(n_thresholds=20).auc
    assert np.all(auc.samples >= 0)
    assert np.all(auc.samples <= 1)
    assert 0.0 < auc.point_estimate < 1.0


def test_pr_auc_in_bounds():
    """PR AUC samples lie in [0, 1]."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, 100).astype(bool)
    y_score = rng.uniform(0, 1, 100)
    auc = BinaryClassifier(y_true, y_score, seed=7).pr_curve(n_thresholds=20).auc
    assert np.all(auc.samples >= 0)
    assert np.all(auc.samples <= 1)


def test_from_cm_curves_raise():
    """roc_curve and pr_curve raise NotImplementedError when built from_cm."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3)
    with pytest.raises(NotImplementedError):
        bc.roc_curve()
    with pytest.raises(NotImplementedError):
        bc.pr_curve()


def test_value_score_curve_perfect_classifier():
    """Perfect classifier VS curve should be near 1 across all C/L."""
    bc = BinaryClassifier.from_cm(tp=1000, fn=0, tn=1000, fp=0, seed=0)
    curve = bc.at_threshold().value_score_curve(n_cl=50)
    assert curve._vs.shape == (50, 20_000)
    assert curve._vs.mean() > 0.95


def test_value_score_curve_shape():
    """VS matrix has shape (n_cl, n_samples)."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3, seed=0)
    curve = bc.at_threshold().value_score_curve(n_cl=30)
    assert curve._vs.shape == (30, 20_000)


def test_relative_value_perfect_classifier():
    """Perfect classifier (all TP and TN) should have VS = 1 for all C/L."""
    # p11=phi, p10=0, p01=0 → VS = 1 exactly; with posterior, VS ≈ 1
    bc = BinaryClassifier.from_cm(tp=1000, fn=0, tn=1000, fp=0, seed=0)
    vs = bc.at_threshold().relative_value(0.3)
    assert abs(vs.point_estimate - 1.0) < 0.02


def test_mean_expense():
    """mean_expense is non-negative and decreases toward 0 for a perfect classifier."""
    bc_perfect = BinaryClassifier.from_cm(tp=1000, fn=0, tn=1000, fp=0, seed=0)
    bc_poor = BinaryClassifier.from_cm(tp=5, fn=15, tn=5, fp=15, seed=0)
    cost, loss = 1.0, 5.0
    me_perfect = bc_perfect.at_threshold().mean_expense(cost, loss).point_estimate
    me_poor = bc_poor.at_threshold().mean_expense(cost, loss).point_estimate
    assert me_perfect >= 0
    assert me_poor > me_perfect  # poor classifier has higher mean expense
    # TP-only: expense ≈ TP_prop * cost ≈ prevalence * cost
    assert me_perfect < cost * 0.6


def test_relative_value_invalid_ratio():
    """cost_loss_ratio outside (0, 1) raises ValueError."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3)
    with pytest.raises(ValueError):
        bc.at_threshold().relative_value(0.0)
    with pytest.raises(ValueError):
        bc.at_threshold().relative_value(1.0)


def test_good_classifier_auc_above_half():
    """A discriminating classifier has ROC AUC mean > 0.5."""
    rng = np.random.default_rng(99)
    y_true = rng.integers(0, 2, 200).astype(bool)
    # Scores correlated with labels
    y_score = y_true.astype(float) + rng.normal(0, 0.5, 200)
    auc = BinaryClassifier(y_true, y_score, seed=99).roc_curve().auc
    assert auc.point_estimate > 0.5


def test_remaining_threshold_metrics():
    """npv, f1, balanced_accuracy, bookmaker_informedness, and mcc return valid results."""
    bc = BinaryClassifier.from_cm(tp=20, fn=5, tn=15, fp=4, seed=0)
    t = bc.at_threshold()
    for method in [t.npv, t.f1, t.balanced_accuracy, t.bookmaker_informedness, t.mcc]:
        result = method()
        assert result.samples.shape == (20_000,)
        assert np.isfinite(result.point_estimate)


def test_at_threshold_from_scores():
    """at_threshold() computes CM from y_score when scores are provided."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50).astype(bool)
    y_score = rng.uniform(0, 1, 50)
    result = BinaryClassifier(y_true, y_score, seed=0).at_threshold(0.5)
    assert 0.0 < result.accuracy().point_estimate < 1.0


def test_cm_sampler_default_rng():
    """CMSampler creates its own rng when none is provided."""
    from classifier_uncertainty._sampler import CMSampler

    sampler = CMSampler(10, 2, 8, 3)
    assert sampler.cm_samples.shape == (20_000, 4)


def test_metric_result_plot():
    """MetricResult.plot() returns axes without error."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3, seed=0)
    ax = bc.at_threshold().tpr().plot()
    assert ax is not None


def test_value_score_curve_plot():
    """ValueScoreCurve.plot() returns axes without error."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3, seed=0)
    ax = bc.at_threshold().value_score_curve(n_cl=20).plot()
    assert ax is not None


def test_roc_curve_plot():
    """ROCResult.plot() returns axes without error."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50).astype(bool)
    y_score = rng.uniform(0, 1, 50)
    ax = BinaryClassifier(y_true, y_score, seed=0).roc_curve(n_thresholds=10).plot()
    assert ax is not None


def test_at_prevalence_fixed_phi():
    """at_prevalence(float) fixes phi exactly while preserving TPR posterior."""
    bc = BinaryClassifier.from_cm(tp=26, fn=4, tn=60, fp=10, seed=0)
    result = bc.at_threshold()
    result2 = result.at_prevalence(0.1)
    phi_implied = result2._sampler.cm_samples[:, 0] + result2._sampler.cm_samples[:, 1]
    np.testing.assert_allclose(phi_implied, 0.1, atol=1e-12)
    np.testing.assert_allclose(result.tpr().samples, result2.tpr().samples, rtol=1e-10)


def test_at_prevalence_beta_prior():
    """at_prevalence(tuple) samples phi from Beta while preserving TPR posterior."""
    bc = BinaryClassifier.from_cm(tp=26, fn=4, tn=60, fp=10, seed=0)
    result = bc.at_threshold()
    result2 = result.at_prevalence((2, 18), seed=0)  # Beta(2, 18) → mean ≈ 0.1
    phi_implied = result2._sampler.cm_samples[:, 0] + result2._sampler.cm_samples[:, 1]
    assert not np.allclose(phi_implied, phi_implied[0])  # phi varies across samples
    assert abs(phi_implied.mean() - 0.1) < 0.02
    np.testing.assert_allclose(result.tpr().samples, result2.tpr().samples, rtol=1e-10)


def test_at_prevalence_invalid_phi():
    """at_prevalence raises ValueError for phi outside (0, 1)."""
    bc = BinaryClassifier.from_cm(tp=10, fn=2, tn=8, fp=3)
    result = bc.at_threshold()
    with pytest.raises(ValueError):
        result.at_prevalence(0.0)
    with pytest.raises(ValueError):
        result.at_prevalence(1.0)


def test_pr_curve_plot():
    """PRResult.plot() returns axes without error."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50).astype(bool)
    y_score = rng.uniform(0, 1, 50)
    ax = BinaryClassifier(y_true, y_score, seed=0).pr_curve(n_thresholds=10).plot()
    assert ax is not None
