"""ROC and Precision-Recall curve results with uncertainty."""

from __future__ import annotations

import numpy as np

from ._results import MetricResult
from ._sampler import CMSampler


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Integrate y over x along axis, compatible with numpy <2.0 and >=2.0."""
    # ponytail: np.trapezoid added in numpy 2.0, np.trapz removed in numpy 2.0
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    assert fn is not None
    return fn(y, x, axis=axis)


def _draw_ellipse(ax, x: np.ndarray, y: np.ndarray, level: float = 0.95, **kwargs) -> None:
    """Draw a covariance confidence ellipse from 2D posterior samples.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add the ellipse to.
    x : np.ndarray
        Samples for the horizontal axis.
    y : np.ndarray
        Samples for the vertical axis.
    level : float
        Confidence level. Default is ``0.95``.
    **kwargs
        Forwarded to ``matplotlib.patches.Ellipse``.
    """
    from matplotlib.patches import Ellipse

    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # chi2(df=2) quantile: CDF(x) = 1 - exp(-x/2), so Q(p) = -2 ln(1-p)
    chi2_val = -2.0 * np.log(1.0 - level)
    angle = float(np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1])))
    width = 2.0 * float(np.sqrt(chi2_val * max(float(eigenvalues[-1]), 0.0)))
    height = 2.0 * float(np.sqrt(chi2_val * max(float(eigenvalues[0]), 0.0)))
    ax.add_patch(
        Ellipse(
            (float(np.mean(x)), float(np.mean(y))),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            **kwargs,
        )
    )


class ROCResult:
    """Uncertainty-aware ROC curve.

    Produced by :meth:`BinaryClassifier.roc_curve`. Each threshold in the
    grid yields a posterior distribution over (FPR, TPR), visualised as a
    2D covariance ellipse.

    Attributes
    ----------
    auc : MetricResult
        Posterior distribution of AUC-ROC, computed via per-sample
        trapezoid integration.
    """

    def __init__(self, thresholds: np.ndarray, samplers: list[CMSampler]) -> None:
        self._thresholds = np.asarray(thresholds)
        cms = np.array([s.cm_samples for s in samplers])  # (n_thresholds, n_samples, 4)
        tp, fn, tn, fp = cms[:, :, 0], cms[:, :, 1], cms[:, :, 2], cms[:, :, 3]
        self._fpr = fp / (fp + tn)  # (n_thresholds, n_samples)
        self._tpr = tp / (tp + fn)

    @property
    def auc(self) -> MetricResult:
        """Posterior distribution of AUC-ROC via per-sample trapezoid integration."""
        order = np.argsort(self._fpr, axis=0)
        fpr_s = np.take_along_axis(self._fpr, order, axis=0)
        tpr_s = np.take_along_axis(self._tpr, order, axis=0)
        return MetricResult(np.abs(_trapz(tpr_s, fpr_s, axis=0)))

    def plot(self, ax=None, level: float = 0.95, color: str = "C0", alpha: float = 0.4):
        """Plot the ROC curve with 2D confidence ellipses at each threshold.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Uses ``plt.gca()`` if ``None``.
        level : float
            Confidence level for ellipses. Default is ``0.95``.
        color : str
            Curve and ellipse colour. Default is ``"C0"``.
        alpha : float
            Ellipse opacity. Default is ``0.4``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        for i in range(self._fpr.shape[0]):
            _draw_ellipse(ax, self._fpr[i], self._tpr[i], level=level, edgecolor=color, alpha=alpha)
        mean_fpr = self._fpr.mean(axis=1)
        mean_tpr = self._tpr.mean(axis=1)
        order = np.argsort(mean_fpr)
        ax.plot(mean_fpr[order], mean_tpr[order], color=color, linewidth=1.5)
        ax.set_xlabel("FPR (1 − Specificity)")
        ax.set_ylabel("TPR (Sensitivity / Recall)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        return ax


class PRResult:
    """Uncertainty-aware Precision-Recall curve.

    Produced by :meth:`BinaryClassifier.pr_curve`. Each threshold in the
    grid yields a posterior distribution over (Recall, Precision), visualised
    as a 2D covariance ellipse.

    Attributes
    ----------
    auc : MetricResult
        Posterior distribution of AUC-PR (average precision), computed via
        per-sample trapezoid integration.
    """

    def __init__(self, thresholds: np.ndarray, samplers: list[CMSampler]) -> None:
        self._thresholds = np.asarray(thresholds)
        cms = np.array([s.cm_samples for s in samplers])  # (n_thresholds, n_samples, 4)
        tp, fn, fp = cms[:, :, 0], cms[:, :, 1], cms[:, :, 3]
        self._recall = tp / (tp + fn)  # = TPR
        denom = tp + fp
        self._precision = np.where(denom > 0, tp / denom, 1.0)

    @property
    def auc(self) -> MetricResult:
        """Posterior distribution of AUC-PR via per-sample trapezoid integration."""
        order = np.argsort(self._recall, axis=0)
        rec_s = np.take_along_axis(self._recall, order, axis=0)
        prec_s = np.take_along_axis(self._precision, order, axis=0)
        return MetricResult(np.abs(_trapz(prec_s, rec_s, axis=0)))

    def plot(self, ax=None, level: float = 0.95, color: str = "C0", alpha: float = 0.4):
        """Plot the PR curve with 2D confidence ellipses at each threshold.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Uses ``plt.gca()`` if ``None``.
        level : float
            Confidence level for ellipses. Default is ``0.95``.
        color : str
            Curve and ellipse colour. Default is ``"C0"``.
        alpha : float
            Ellipse opacity. Default is ``0.4``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        for i in range(self._recall.shape[0]):
            _draw_ellipse(
                ax, self._recall[i], self._precision[i], level=level, edgecolor=color, alpha=alpha
            )
        mean_recall = self._recall.mean(axis=1)
        mean_prec = self._precision.mean(axis=1)
        order = np.argsort(mean_recall)
        ax.plot(mean_recall[order], mean_prec[order], color=color, linewidth=1.5)
        ax.set_xlabel("Recall (TPR)")
        ax.set_ylabel("Precision (PPV)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax
