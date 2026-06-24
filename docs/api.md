# API Reference

## BinaryClassifier

The main entry point. Construct from `(y_true, y_score)` or from bare confusion
matrix counts via `from_cm`.

::: classifier_uncertainty._classifier.BinaryClassifier
    options:
      show_source: false

---

## ThresholdResult

Returned by `BinaryClassifier.at_threshold()`. All metrics share the same
posterior CM samples, preserving correlations.

::: classifier_uncertainty._results.ThresholdResult
    options:
      show_source: false

---

## MetricResult

Returned by every metric method. Wraps posterior samples and provides
credible intervals and plotting.

::: classifier_uncertainty._results.MetricResult
    options:
      show_source: false

---

## ValueScoreCurve

Returned by `ThresholdResult.value_score_curve()`.

::: classifier_uncertainty._results.ValueScoreCurve
    options:
      show_source: false

---

## ROCResult

Returned by `BinaryClassifier.roc_curve()`.

::: classifier_uncertainty._curves.ROCResult
    options:
      show_source: false

---

## PRResult

Returned by `BinaryClassifier.pr_curve()`.

::: classifier_uncertainty._curves.PRResult
    options:
      show_source: false
