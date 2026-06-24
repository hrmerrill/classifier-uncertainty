"""ROC and Precision-Recall curve results with uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._results import MetricResult
from ._sampler import CMSampler

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Integrate y over x along axis, compatible with numpy <2.0 and >=2.0."""
    # ponytail: np.trapezoid added in numpy 2.0, np.trapz removed in numpy 2.0
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    assert fn is not None
    return fn(y, x, axis=axis)


def _interp_band(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    grid: np.ndarray,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate y samples onto a fixed x grid and return (mean, lo, hi) HPDI.

    Parameters
    ----------
    x_mat : np.ndarray
        Shape ``(n_thresholds, n_samples)`` — x-axis samples per threshold.
    y_mat : np.ndarray
        Shape ``(n_thresholds, n_samples)`` — y-axis samples per threshold.
    grid : np.ndarray
        Fixed x values onto which y is interpolated per sample.
    level : float
        HPDI level. Default is ``0.95``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(mean, lo, hi)`` each of shape ``(len(grid),)``.
    """
    n_samples = x_mat.shape[1]
    y_on_grid = np.zeros((len(grid), n_samples))
    for j in range(n_samples):
        xs = x_mat[:, j]
        ys = y_mat[:, j]
        order = np.argsort(xs)
        y_on_grid[:, j] = np.interp(grid, xs[order], ys[order])
    mean = y_on_grid.mean(axis=1)
    # Vectorised HPDI: sort all grid rows at once, then find minimum-width window
    y_sorted = np.sort(y_on_grid, axis=1)
    window = int(np.ceil(level * n_samples))
    widths = y_sorted[:, window:] - y_sorted[:, : n_samples - window]
    i = np.argmin(widths, axis=1)
    rows = np.arange(len(grid))
    return mean, y_sorted[rows, i], y_sorted[rows, i + window]


class ROCResult:
    """Uncertainty-aware ROC curve.

    Produced by :meth:`BinaryClassifier.roc_curve`. Uncertainty is shown as a
    95 % HPDI band computed by interpolating TPR samples onto a fixed FPR grid.

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

    def plot(
        self, ax: Axes | None = None, level: float = 0.95, color: str = "C0", alpha: float = 0.3
    ) -> Axes:
        """Plot the ROC curve with a posterior HPDI band.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Uses ``plt.gca()`` if ``None``.
        level : float
            HPDI level for the shaded band. Default is ``0.95``.
        color : str
            Curve and band colour. Default is ``"C0"``.
        alpha : float
            Band opacity. Default is ``0.3``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        grid = np.linspace(0, 1, 300)
        mean, lo, hi = _interp_band(self._fpr, self._tpr, grid, level)
        ax.fill_between(
            grid, lo, hi, alpha=alpha, color=color, edgecolor="none", label=f"{level:.0%} HPDI"
        )
        ax.plot(grid, mean, color=color, linewidth=1.5, label="mean")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("FPR (1 − Specificity)")
        ax.set_ylabel("TPR (Sensitivity / Recall)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        return ax


class PRResult:
    """Uncertainty-aware Precision-Recall curve.

    Produced by :meth:`BinaryClassifier.pr_curve`. Uncertainty is shown as a
    95 % HPDI band computed by interpolating Precision samples onto a fixed
    Recall grid.

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

    def plot(
        self, ax: Axes | None = None, level: float = 0.95, color: str = "C0", alpha: float = 0.3
    ) -> Axes:
        """Plot the PR curve with a posterior HPDI band.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Uses ``plt.gca()`` if ``None``.
        level : float
            HPDI level for the shaded band. Default is ``0.95``.
        color : str
            Curve and band colour. Default is ``"C0"``.
        alpha : float
            Band opacity. Default is ``0.3``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        grid = np.linspace(0, 1, 300)
        mean, lo, hi = _interp_band(self._recall, self._precision, grid, level)
        ax.fill_between(
            grid, lo, hi, alpha=alpha, color=color, edgecolor="none", label=f"{level:.0%} HPDI"
        )
        ax.plot(grid, mean, color=color, linewidth=1.5, label="mean")
        ax.set_xlabel("Recall (TPR)")
        ax.set_ylabel("Precision (PPV)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        return ax
