"""BinaryClassifier: main entry point."""

from __future__ import annotations

import numpy as np

from ._curves import PRResult, ROCResult
from ._results import ThresholdResult
from ._sampler import CMSampler


def _cm_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """Return ``(tp, fn, tn, fp)`` counts at a given score threshold."""
    y_pred = y_score >= threshold
    return (
        int(np.sum(y_pred & y_true)),
        int(np.sum(~y_pred & y_true)),
        int(np.sum(~y_pred & ~y_true)),
        int(np.sum(y_pred & ~y_true)),
    )


def _threshold_grid(scores: np.ndarray, n: int) -> np.ndarray:
    """Return ``n`` thresholds spanning from all-positive to all-negative predictions."""
    unique = np.unique(scores)
    interior = np.quantile(unique, np.linspace(0, 1, n - 2))
    return np.concatenate([[unique.max() + 1], interior, [unique.min() - 1]])


class BinaryClassifier:
    """Uncertainty-aware binary classifier evaluator.

    Implements Bayesian uncertainty quantification for classifier metrics
    following Tötsch & Hoffmann (2020). Metrics are derived by sampling the
    confusion matrix probability matrix (θ) from three independent Beta
    posteriors for prevalence (φ), TPR, and TNR.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (``bool`` or ``0``/``1``).
    y_score : np.ndarray
        Classifier scores; higher values indicate a more positive prediction.
    n_samples : int
        Number of posterior CM samples. Default is ``20_000``.
    prior : tuple[float, float]
        Beta(α, β) prior applied uniformly to φ, TPR, and TNR. Default is the
        Laplace prior ``(1.0, 1.0)``. The same prior is used for all three
        distributions; per-distribution priors are not currently supported. If
        you need that, open an issue on GitHub.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_samples: int = 20_000,
        prior: tuple[float, float] = (1.0, 1.0),
        seed: int | None = None,
    ) -> None:
        self._y_true = np.asarray(y_true, dtype=bool)
        self._y_score = np.asarray(y_score, dtype=float)
        self._n_samples = n_samples
        self._prior = prior
        self._rng = np.random.default_rng(seed)
        self._fixed_cm: tuple[int, int, int, int] | None = None

    @classmethod
    def from_cm(
        cls,
        tp: int,
        fn: int,
        tn: int,
        fp: int,
        n_samples: int = 20_000,
        prior: tuple[float, float] = (1.0, 1.0),
        seed: int | None = None,
    ) -> "BinaryClassifier":
        """Construct from observed confusion matrix counts.

        Parameters
        ----------
        tp : int
            True positive count.
        fn : int
            False negative count.
        tn : int
            True negative count.
        fp : int
            False positive count.
        n_samples : int
            Number of posterior CM samples. Default is ``20_000``.
        prior : tuple[float, float]
            Beta(α, β) prior applied uniformly to φ, TPR, and TNR. Default is
            Laplace ``(1.0, 1.0)``. Per-distribution priors are not currently
            supported; open a GitHub issue if you need them.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        BinaryClassifier
            Instance with a fixed CM; :meth:`roc_curve` and :meth:`pr_curve`
            are not available.
        """
        obj = cls.__new__(cls)
        obj._y_true = None
        obj._y_score = None
        obj._n_samples = n_samples
        obj._prior = prior
        obj._rng = np.random.default_rng(seed)
        obj._fixed_cm = (int(tp), int(fn), int(tn), int(fp))
        return obj

    def at_threshold(self, threshold: float = 0.5) -> ThresholdResult:
        """Return metric distributions at a fixed score threshold.

        Parameters
        ----------
        threshold : float
            Decision boundary applied to ``y_score``. Default is ``0.5``.

        Returns
        -------
        ThresholdResult
            Posterior metric distributions at this threshold.
        """
        if self._fixed_cm is not None:
            tp, fn, tn, fp = self._fixed_cm
        else:
            tp, fn, tn, fp = _cm_at_threshold(self._y_true, self._y_score, threshold)
        sampler = CMSampler(
            tp,
            fn,
            tn,
            fp,
            prior=self._prior,
            n_samples=self._n_samples,
            rng=self._rng,
        )
        return ThresholdResult(sampler)

    def _require_scores(self, method: str) -> None:
        """Raise ``NotImplementedError`` if ``y_score`` is unavailable."""
        if self._y_score is None:
            raise NotImplementedError(
                f"{method}() requires y_true and y_score; not available from from_cm()."
            )

    def roc_curve(self, n_thresholds: int = 50) -> ROCResult:
        """Return an uncertainty-aware ROC curve over a quantile-spaced threshold grid.

        Parameters
        ----------
        n_thresholds : int
            Number of thresholds in the grid. Default is ``50``.

        Returns
        -------
        ROCResult
            ROC curve with per-threshold posterior uncertainty ellipses.
        """
        self._require_scores("roc_curve")
        thresholds = _threshold_grid(self._y_score, n_thresholds)
        samplers = [
            CMSampler(
                *_cm_at_threshold(self._y_true, self._y_score, t),
                prior=self._prior,
                n_samples=self._n_samples,
                rng=self._rng,
            )
            for t in thresholds
        ]
        return ROCResult(thresholds, samplers)

    def pr_curve(self, n_thresholds: int = 50) -> PRResult:
        """Return an uncertainty-aware PR curve over a quantile-spaced threshold grid.

        Parameters
        ----------
        n_thresholds : int
            Number of thresholds in the grid. Default is ``50``.

        Returns
        -------
        PRResult
            PR curve with per-threshold posterior uncertainty ellipses.
        """
        self._require_scores("pr_curve")
        thresholds = _threshold_grid(self._y_score, n_thresholds)
        samplers = [
            CMSampler(
                *_cm_at_threshold(self._y_true, self._y_score, t),
                prior=self._prior,
                n_samples=self._n_samples,
                rng=self._rng,
            )
            for t in thresholds
        ]
        return PRResult(thresholds, samplers)
