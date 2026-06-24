# classifier-uncertainty

Bayesian uncertainty quantification for binary classifier metrics.

Implements the approach from [Tötsch & Hoffmann (2021)](https://peerj.com/articles/cs-398/):
sample the confusion matrix from Beta posteriors, then propagate to any metric.
The result is a full posterior distribution over each metric, not just a point estimate.

## Installation

```bash
pip install classifier-uncertainty
```

## Quick start

```python
from classifier_uncertainty import BinaryClassifier

# From ground-truth labels and classifier scores
bc = BinaryClassifier(y_true, y_score)

# Or from published confusion matrix counts (e.g. from a paper)
bc = BinaryClassifier.from_cm(tp=26, fn=0, tn=6, fp=2)
```

## Threshold metrics

```python
t = bc.at_threshold(0.5)   # ThresholdResult — all metrics share the same CM samples

result = t.tpr()            # MetricResult
result.point_estimate       # posterior mean ≈ 0.963
result.credible_interval()  # 95% HPDI ≈ (0.89, 1.0)
result.metric_uncertainty   # HPDI length ≈ 0.11
result.plot()               # posterior histogram with CI shading
result.samples              # posterior samples
```

Built-in metrics:

| Method | Aliases | Formula |
|---|---|---|
| `accuracy()` | | (TP + TN) / N |
| `tpr()` | `sensitivity`, `recall` | TP / (TP + FN) |
| `tnr()` | `specificity` | TN / (TN + FP) |
| `precision()` | `ppv` | TP / (TP + FP) |
| `npv()` | | TN / (TN + FN) |
| `f1()` | | 2TP / (2TP + FP + FN) |
| `balanced_accuracy()` | | (TPR + TNR) / 2 |
| `bookmaker_informedness()` | | TPR + TNR − 1 |
| `mcc()` | | Matthews correlation coefficient |

**Custom metrics** receive CM entry proportions as numpy arrays, so any ratio metric works directly:

```python
# False discovery rate
t.metric(lambda tp, fn, tn, fp: fp / (tp + fp))
```

## Threshold-agnostic curves

```python
roc = bc.roc_curve()        # sweep a quantile-spaced threshold grid
roc.plot()                  # ROC curve + 2D covariance ellipses at each threshold
roc.auc                     # MetricResult — full AUC-ROC posterior

bc.pr_curve().plot()        # Precision-Recall curve with uncertainty ellipses
```

## Relative economic value

Based on [Wilks (2001)](https://doi.org/10.1017/S1350482701000366):

```python
t.relative_value(0.3)        # Value Score at cost/loss ratio C/L = 0.3
t.value_score_curve().plot() # VS curve over all C/L ∈ (0, 1) with credible band
```

---

See [Examples](examples.md) for worked examples with visualizations, and
[API Reference](api.md) for full documentation.
