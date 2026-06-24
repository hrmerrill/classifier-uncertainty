"""Quantify uncertainty around classification performance metrics.

Implements methods from Tötsch & Hoffmann (2020) to compute posterior
distributions for classifier metrics via Bayesian confusion matrix sampling.
"""

from ._classifier import BinaryClassifier

__all__ = ["BinaryClassifier"]
