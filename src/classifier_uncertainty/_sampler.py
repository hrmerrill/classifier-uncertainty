"""Bayesian confusion matrix sampler."""

from __future__ import annotations

import numpy as np


class CMSampler:
    """Posterior CM sampler via three Beta distributions (Tötsch & Hoffmann 2020).

    Samples prevalence (φ), TPR, and TNR independently from Beta posteriors,
    then reconstructs all CM entries as proportions (θ values summing to ~1).

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
    prior : tuple[float, float]
        Beta(α, β) prior applied uniformly to φ, TPR, and TNR. Default is
        Laplace ``(1, 1)``. Per-distribution priors are not currently
        supported; open a GitHub issue if you need them.
    n_samples : int
        Number of posterior samples to draw.
    rng : np.random.Generator, optional
        Random number generator. Creates a new default generator if ``None``.

    Attributes
    ----------
    cm_samples : np.ndarray
        Posterior CM samples of shape ``(n_samples, 4)`` with columns
        ``[θ_TP, θ_FN, θ_TN, θ_FP]`` — proportions summing to ~1 per row.
    """

    def __init__(
        self,
        tp: int,
        fn: int,
        tn: int,
        fp: int,
        prior: tuple[float, float] = (1.0, 1.0),
        n_samples: int = 20_000,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        a, b = prior
        phi = rng.beta(tp + fn + a, tn + fp + b, n_samples)
        tpr_s = rng.beta(tp + a, fn + b, n_samples)
        tnr_s = rng.beta(tn + a, fp + b, n_samples)
        tp_s = tpr_s * phi
        fn_s = (1.0 - tpr_s) * phi
        tn_s = tnr_s * (1.0 - phi)
        fp_s = (1.0 - tnr_s) * (1.0 - phi)
        self.cm_samples = np.stack([tp_s, fn_s, tn_s, fp_s], axis=1)

    @classmethod
    def _from_samples(cls, cm_samples: np.ndarray) -> "CMSampler":
        """Construct a CMSampler directly from pre-computed CM samples.

        Parameters
        ----------
        cm_samples : np.ndarray
            Array of shape ``(n_samples, 4)`` with columns
            ``[θ_TP, θ_FN, θ_TN, θ_FP]``.

        Returns
        -------
        CMSampler
            Instance whose ``cm_samples`` attribute is set to the given array.
        """
        obj = cls.__new__(cls)
        obj.cm_samples = cm_samples
        return obj
