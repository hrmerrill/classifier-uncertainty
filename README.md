# Classifier Uncertainty

<p align="center">
  <img src="https://img.shields.io/pypi/v/classifier-uncertainty?style=flat-square&color=111111&label=pypi" alt="pypi">
  <img src="https://github.com/hrmerrill/classifier-uncertainty/actions/workflows/ci.yml/badge.svg?style=flat-square" alt="CI">
  <img src="https://raw.githubusercontent.com/hrmerrill/classifier-uncertainty/main/.github/badges/coverage.svg" alt="coverage">
  <img src="https://raw.githubusercontent.com/hrmerrill/classifier-uncertainty/main/.github/badges/interrogate_badge.svg" alt="docstring coverage">
  <img src="https://img.shields.io/badge/license-unlicense-111111?style=flat-square" alt="Unlicense">
</p>

## About

This package implements methods from [Tötsch N and Hoffmann D. 2021](https://peerj.com/articles/cs-398/) to quantify the uncertainty around classification performance metrics. Classifiers are often tested on relatively small data sets, which should lead to uncertain performance metrics. Even when tested on large data sets, performance is often presented as a percentage with three decimals, and competing classifiers are ranked assuming such a precision. Reducing metric uncertainty below 0.001% would require tens of billions of data points.

The original authors' Python implementation is available at [niklastoe/classifier_metric_uncertainty](https://github.com/niklastoe/classifier_metric_uncertainty). This package was built independently and extends that work with:

- **Score-based input** — accepts raw `(y_true, y_score)` pairs and sweeps thresholds; the original takes confusion matrix counts only
- **ROC and PR curves with uncertainty bands** — including AUC posterior distributions
- **Economic value analysis** — Value Score (Wilks 2001) and mean expense posteriors
- **Custom metrics** — evaluate any `f(tp, fn, tn, fp)` over the posterior CM samples

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

# fix the binarization threshold
t = bc.at_threshold(0.5)
```

## What questions can this answer?

**How well is a classifier likely to perform on a new, similar dataset?**
```python
t.tpr().point_estimate, t.tpr().credible_interval()
```

**How likely is classifier A better than classifier B on a given metric?**
```python
(bc_a.at_threshold().tpr().samples > bc_b.at_threshold().tpr().samples).mean()
```

**How likely is this model more cost-effective than business-as-usual?**
```python
(t_model.mean_expense(C, L).samples < t_bau.mean_expense(C, L).samples).mean()
```

**Does this classifier meet my minimum recall requirement?**
```python
(t.tpr().samples > 0.8).mean()
```

**Do precision and recall meet requirements simultaneously?**
```python
((t.tpr().samples > 0.8) & (t.precision().samples > 0.8)).mean()
```

**Is this classifier better than random guessing?**
```python
(t.bookmaker_informedness().samples > 0).mean()
```

**Should I trust this published result?**
```python
BinaryClassifier.from_cm(tp=26, fn=0, tn=6, fp=2).at_threshold().tpr().credible_interval()
```

## For Developers

### Setup

```bash
uv sync  # install package + dev dependencies into .venv
```

### Development workflow

All changes should be made on a branch and merged via pull request — do not commit directly to `main`.

```bash
git checkout -b feat/my-feature   # or fix/, docs/, refactor/, etc.

# ... make changes ...

make format      # auto-fix formatting and lint violations
make check       # lint, type-check, and verify docstring coverage
make test        # run tests with coverage (90% minimum)
make docs-serve  # preview docs locally at http://127.0.0.1:8000

git push -u origin feat/my-feature
# open a pull request on GitHub
```

CI runs `make check` and `make test` automatically on every push and pull request. A PR cannot be merged if CI fails.

### What triggers what

| Action | CI checks | Docs deployed | Package published |
|---|---|---|---|
| Push to any branch / open PR | ✓ | | |
| Merge to `main` | ✓ | ✓ | |
| Push a `v*` tag | | | ✓ |

**Docs-only change** (e.g. fix a typo in `docs/` or a docstring): open a PR and merge to `main` — docs redeploy automatically, no tag needed.

**Code-only change** (e.g. bug fix): merge to `main`, then tag when ready to publish (see below). Docs will also redeploy on merge, reflecting any updated docstrings.

### Publishing a new package version

1. Bump the version in `pyproject.toml`:
   ```bash
   make patch   # 0.1.0 → 0.1.1  (bug fixes)
   make minor   # 0.1.0 → 0.2.0  (new features)
   make major   # 0.1.0 → 1.0.0  (breaking changes)
   ```
2. Commit, tag, and push:
   ```bash
   git add pyproject.toml
   git commit -m "chore: bump version to v0.x.x"
   git tag v0.x.x
   git push && git push --tags
   ```

Pushing the tag triggers the publish workflow, which runs the test suite and publishes the package to PyPI. Check that the release appeared:

- Package: https://pypi.org/project/classifier-uncertainty/
- Docs: https://hrmerrill.github.io/classifier-uncertainty/