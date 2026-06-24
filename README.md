# Classifier Uncertainty

## About

This package implements methods from [Tötsch N and Hoffmann D. 2021](https://peerj.com/articles/cs-398/) to quantify the uncertainty around classification performance metrics. Classifiers are often tested on relatively small data sets, which should lead to uncertain performance metrics. Even when tested on large data sets, performance is often presented as a percentage with three decimals, and competing classifiers are ranked assuming such a precision. Reducing metric uncertainty below 0.001% would require tens of billions of data points.

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