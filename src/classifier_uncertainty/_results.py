"""Metric and threshold result types."""

from __future__ import annotations

import numpy as np

from ._sampler import CMSampler


def _hpdi(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Return the minimum-width interval containing ``level`` of sorted samples."""
    s = np.sort(samples)
    n = len(s)
    window = int(np.ceil(level * n))
    widths = s[window:] - s[: n - window]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + window])


class MetricResult:
    """Posterior distribution of a scalar classifier metric.

    Attributes
    ----------
    samples : np.ndarray
        Raw posterior samples of shape ``(n_samples,)``.
    point_estimate : float
        Posterior mean.
    metric_uncertainty : float
        Length of the 95 % HPDI — the metric uncertainty (MU) of
        Tötsch & Hoffmann (2020).
    """

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = np.asarray(samples, dtype=float)

    @property
    def samples(self) -> np.ndarray:
        """Raw posterior samples of shape ``(n_samples,)``."""
        return self._samples

    @property
    def point_estimate(self) -> float:
        """Posterior mean."""
        return float(np.mean(self._samples))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the highest posterior density interval (HPDI).

        Parameters
        ----------
        level : float
            Probability mass to enclose. Default is ``0.95``.

        Returns
        -------
        tuple[float, float]
            ``(lower, upper)`` bounds of the HPDI.
        """
        return _hpdi(self._samples, level)

    @property
    def metric_uncertainty(self) -> float:
        """Length of the 95 % HPDI — metric uncertainty (MU) of Tötsch & Hoffmann."""
        lo, hi = self.credible_interval(0.95)
        return hi - lo

    def plot(self, ax=None, level: float = 0.95, **kwargs):
        """Plot a histogram of posterior samples with HPDI shading.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Uses ``plt.gca()`` if ``None``.
        level : float
            HPDI level to shade. Default is ``0.95``.
        **kwargs
            Forwarded to ``ax.hist``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        lo, hi = self.credible_interval(level)
        ax.hist(self._samples, bins=80, density=True, alpha=0.7, **kwargs)
        ax.axvline(self.point_estimate, color="k", linewidth=1.5, label="mean")
        ax.axvspan(lo, hi, alpha=0.25, color="C1", label=f"{level:.0%} HPDI")
        ax.legend(fontsize=9)
        return ax


class ValueScoreCurve:
    """Value Score as a function of cost/loss ratio, with posterior uncertainty.

    Produced by :meth:`ThresholdResult.value_score_curve`. The VS curve
    (Wilks 2001) shows the relative economic value of a classifier as a
    function of the decision-maker's cost/loss ratio.
    """

    def __init__(self, cl_values: np.ndarray, vs_matrix: np.ndarray) -> None:
        self._cl = cl_values  # (n_cl,)
        self._vs = vs_matrix  # (n_cl, n_samples)

    def plot(self, ax=None, level: float = 0.95, color: str = "C0", alpha: float = 0.25):
        """Plot the VS curve with a posterior credible band.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Uses ``plt.gca()`` if ``None``.
        level : float
            HPDI level for the shaded band. Default is ``0.95``.
        color : str
            Line and fill colour. Default is ``"C0"``.
        alpha : float
            Fill opacity. Default is ``0.25``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        n = len(self._cl)
        lo = np.empty(n)
        hi = np.empty(n)
        for i in range(n):
            lo[i], hi[i] = _hpdi(self._vs[i], level)
        mean_vs = self._vs.mean(axis=1)
        ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.fill_between(self._cl, lo, hi, alpha=alpha, color=color, label=f"{level:.0%} HPDI")
        ax.plot(self._cl, mean_vs, color=color, linewidth=1.5, label="mean VS")
        ax.set_xlabel("Cost/Loss ratio (C/L)")
        ax.set_ylabel("Value Score")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        return ax


class ThresholdResult:
    """Metric distributions at a fixed classification threshold.

    All metrics share the same CM samples, preserving their correlations.
    Custom metrics receive CM entry proportions (θ values) as numpy arrays
    summing to ~1 per sample, so standard ratio metrics work unchanged.
    """

    def __init__(self, sampler: CMSampler) -> None:
        self._sampler = sampler

    def metric(self, func) -> MetricResult:
        """Compute a custom metric from CM entry proportions.

        Parameters
        ----------
        func : callable
            A function ``f(tp, fn, tn, fp) -> array`` where each argument is a
            numpy array of CM entry proportions (θ values summing to ~1 per
            sample). Standard ratio metrics require no rescaling.

        Returns
        -------
        MetricResult
            Posterior distribution of the custom metric.
        """
        tp, fn, tn, fp = self._sampler.cm_samples.T
        return MetricResult(func(tp, fn, tn, fp))

    def accuracy(self) -> MetricResult:
        """Return the posterior distribution of accuracy: ``(TP + TN) / N``."""
        return self.metric(lambda tp, fn, tn, fp: tp + tn)

    def tpr(self) -> MetricResult:
        """Return the posterior distribution of TPR: ``TP / (TP + FN)``."""
        return self.metric(lambda tp, fn, tn, fp: tp / (tp + fn))

    sensitivity = tpr
    recall = tpr

    def tnr(self) -> MetricResult:
        """Return the posterior distribution of TNR: ``TN / (TN + FP)``."""
        return self.metric(lambda tp, fn, tn, fp: tn / (tn + fp))

    specificity = tnr

    def precision(self) -> MetricResult:
        """Return the posterior distribution of precision (PPV): ``TP / (TP + FP)``."""

        def _ppv(tp, fn, tn, fp):
            """Compute PPV with zero-denominator guard."""
            denom = tp + fp
            return np.where(denom > 0, tp / denom, 0.0)

        return self.metric(_ppv)

    ppv = precision

    def npv(self) -> MetricResult:
        """Return the posterior distribution of NPV: ``TN / (TN + FN)``."""

        def _npv(tp, fn, tn, fp):
            """Compute NPV with zero-denominator guard."""
            denom = tn + fn
            return np.where(denom > 0, tn / denom, 0.0)

        return self.metric(_npv)

    def f1(self) -> MetricResult:
        """Return the posterior distribution of F1: ``2TP / (2TP + FP + FN)``."""

        def _f1(tp, fn, tn, fp):
            """Compute F1 with zero-denominator guard."""
            denom = 2 * tp + fp + fn
            return np.where(denom > 0, 2 * tp / denom, 0.0)

        return self.metric(_f1)

    def balanced_accuracy(self) -> MetricResult:
        """Return the posterior distribution of balanced accuracy: ``(TPR + TNR) / 2``."""
        return self.metric(lambda tp, fn, tn, fp: (tp / (tp + fn) + tn / (tn + fp)) / 2)

    def bookmaker_informedness(self) -> MetricResult:
        """Return the posterior distribution of bookmaker informedness: ``TPR + TNR − 1``."""
        return self.metric(lambda tp, fn, tn, fp: tp / (tp + fn) + tn / (tn + fp) - 1)

    def mcc(self) -> MetricResult:
        """Return the posterior distribution of Matthews correlation coefficient."""

        def _mcc(tp, fn, tn, fp):
            """Compute MCC with zero-denominator guard."""
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            return np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)

        return self.metric(_mcc)

    def value_score_curve(self, n_cl: int = 100) -> ValueScoreCurve:
        """Return the Value Score curve across all cost/loss ratios (Wilks 2001).

        Parameters
        ----------
        n_cl : int
            Number of C/L grid points in the open interval ``(0, 1)``.
            Default is ``100``.

        Returns
        -------
        ValueScoreCurve
            VS posterior distributions over the C/L grid.
        """
        cl_values = np.linspace(0.01, 0.99, n_cl)
        tp, fn, tn, fp = self._sampler.cm_samples.T
        phi = tp + fn  # (n_samples,)
        cl_mat = cl_values[:, np.newaxis]  # (n_cl, 1) — broadcasts over samples
        vs_5b = (cl_mat * (tp + fp - 1.0) + fn) / (cl_mat * (phi - 1.0))
        vs_5c = (cl_mat * (tp + fp) + fn - phi) / (phi * (cl_mat - 1.0))
        vs_matrix = np.where(cl_mat < phi, vs_5b, vs_5c)  # (n_cl, n_samples)
        return ValueScoreCurve(cl_values, vs_matrix)

    def relative_value(self, cost_loss_ratio: float) -> MetricResult:
        """Return the Value Score distribution at a given cost/loss ratio (Wilks 2001).

        Parameters
        ----------
        cost_loss_ratio : float
            C/L in the open interval ``(0, 1)``. Cost of protective action
            divided by loss suffered when the event occurs without protection.

        Returns
        -------
        MetricResult
            Posterior distribution of the Value Score at the given C/L.

        Raises
        ------
        ValueError
            If ``cost_loss_ratio`` is not in ``(0, 1)``.
        """
        cl = float(cost_loss_ratio)
        if not (0.0 < cl < 1.0):
            raise ValueError("cost_loss_ratio must be in the open interval (0, 1)")

        def _vs(tp, fn, tn, fp):
            """Select Wilks (2001) eq. 5b or 5c per sample based on C/L vs prevalence."""
            phi = tp + fn
            vs_5b = (cl * (tp + fp - 1.0) + fn) / (cl * (phi - 1.0))
            vs_5c = (cl * (tp + fp) + fn - phi) / (phi * (cl - 1.0))
            return np.where(cl < phi, vs_5b, vs_5c)

        return self.metric(_vs)
