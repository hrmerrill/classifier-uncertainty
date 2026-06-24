"""MkDocs hook: generate example figures after each build."""

import os


def on_post_build(config, **kwargs):
    """Write example figures into site/assets/ after every build.

    Writing to site_dir (not docs/) keeps the file watcher from detecting
    changes and triggering an infinite rebuild loop during ``mkdocs serve``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from classifier_uncertainty import BinaryClassifier

    assets_dir = os.path.join(config["site_dir"], "assets")
    os.makedirs(assets_dir, exist_ok=True)

    def save(name, fig):
        fig.savefig(os.path.join(assets_dir, name), dpi=110, bbox_inches="tight")
        plt.close(fig)

    # Synthetic dataset: moderately good binary classifier, N=150, prevalence ≈ 40%
    rng = np.random.default_rng(42)
    n = 150
    y_true = (rng.uniform(0, 1, n) < 0.40).astype(bool)
    y_score = np.where(y_true, rng.beta(5, 2, n), rng.beta(2, 5, n))
    bc = BinaryClassifier(y_true, y_score, seed=42)
    t = bc.at_threshold(0.5)

    # --- Figure 1: Posterior distributions of three metrics at threshold 0.5 ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, method, label in zip(
        axes,
        [t.accuracy, t.tpr, t.tnr],
        ["Accuracy", "TPR (Sensitivity)", "TNR (Specificity)"],
    ):
        method().plot(ax=ax)
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.set_ylabel("")
    fig.suptitle("Posterior metric distributions — N=150, threshold=0.5", y=1.02)
    fig.tight_layout()
    save("metric_posteriors.png", fig)

    # --- Figure 2: ROC curve with 2D uncertainty ellipses ---
    fig, ax = plt.subplots(figsize=(5, 5))
    bc.roc_curve(n_thresholds=30).plot(ax=ax)
    fig.tight_layout()
    save("roc_curve.png", fig)

    # --- Figure 3: Precision-Recall curve with 2D uncertainty ellipses ---
    fig, ax = plt.subplots(figsize=(5, 5))
    bc.pr_curve(n_thresholds=30).plot(ax=ax)
    fig.tight_layout()
    save("pr_curve.png", fig)

    # --- Figure 4: Value Score curve ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    t.value_score_curve().plot(ax=ax)
    fig.tight_layout()
    save("vs_curve.png", fig)

    # --- Figure 5: Metric uncertainty vs. sample size ---
    ns = [10, 20, 30, 50, 75, 100, 150, 200, 350, 500]
    mus = []
    for ni in ns:
        tp_i = max(1, round(ni * 0.35))
        fn_i = max(1, round(ni * 0.15))
        tn_i = max(1, round(ni * 0.35))
        fp_i = max(0, ni - tp_i - fn_i - tn_i)
        mus.append(
            BinaryClassifier.from_cm(tp_i, fn_i, tn_i, fp_i, seed=0)
            .at_threshold()
            .tpr()
            .metric_uncertainty
        )
    ns_arr = np.array(ns, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ns_arr, mus, "o-", label="Posterior 95% HPDI length")
    ax.plot(ns_arr, 2 / np.sqrt(ns_arr), "k--", label=r"$2/\sqrt{N}$ (theoretical bound)")
    ax.set_xlabel("Test set size N")
    ax.set_ylabel("Metric uncertainty (MU)")
    ax.set_title("Metric uncertainty vs. sample size")
    ax.legend()
    fig.tight_layout()
    save("mu_vs_n.png", fig)

    # --- Figure 6: Joint vs permuted precision-recall scatter ---
    # Use a small-N CM so the posteriors are spread enough to show the correlation clearly.
    # tp=10 fn=10 tn=20 fp=10 is a balanced hypothetical classifier, N=50.
    bc_joint = BinaryClassifier.from_cm(tp=10, fn=10, tn=20, fp=10, seed=42)
    t_joint = bc_joint.at_threshold()
    recall_s = t_joint.tpr().samples
    prec_s = t_joint.precision().samples
    corr = float(np.corrcoef(recall_s, prec_s)[0, 1])

    scatter_rng = np.random.default_rng(0)
    prec_shuffled = scatter_rng.permutation(prec_s)

    fig, (ax_joint, ax_perm) = plt.subplots(1, 2, figsize=(9, 4.5))
    kw = dict(alpha=0.1, s=2, linewidths=0)
    for ax, xs, ys, color, title in [
        (ax_joint, recall_s, prec_s, "C0", f"Joint samples  (r = {corr:.2f})"),
        (ax_perm, recall_s, prec_shuffled, "C1", "Permuted samples  (r ≈ 0)"),
    ]:
        ax.scatter(xs, ys, color=color, **kw)
        ax.set_xlabel("Recall (TPR)")
        ax.set_ylabel("Precision (PPV)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle(
        "Posterior precision and recall — joint (left) vs independently permuted (right)\n"
        "Same marginal distributions; only the pairing differs",
        fontsize=10,
    )
    fig.tight_layout()
    save("joint_precision_recall.png", fig)

    # --- Figure 7: Kaggle probabilistic ranking (Tötsch & Hoffmann 2020, §2D) ---
    # Recursion Cellular Image Classification competition; N ≈ 15,123 private images.
    # Accuracy modelled as Beta(correct+1, incorrect+1) — the conjugate posterior
    # for a binary correct/incorrect outcome with a Laplace prior.
    kaggle_N = 15_123
    kaggle_acc = [0.99954, 0.99907, 0.99887, 0.99867, 0.99847,
                  0.99827, 0.99807, 0.99787, 0.99767, 0.99747]
    kaggle_rng = np.random.default_rng(42)
    kaggle_samples = np.array([
        kaggle_rng.beta(round(kaggle_N * acc) + 1, kaggle_N - round(kaggle_N * acc) + 1, 20_000)
        for acc in kaggle_acc
    ])  # (10, 20_000)
    rank_per_sample = np.argsort(np.argsort(-kaggle_samples, axis=0), axis=0)
    p_best = (rank_per_sample == 0).mean(axis=1)

    colors10 = plt.cm.Blues(np.linspace(0.85, 0.3, 10))
    fig, (ax_dens, ax_bar) = plt.subplots(2, 1, figsize=(9, 7))

    # Top: overlapping density curves (histogram, zoomed x-axis)
    x_lo = min(s.min() for s in kaggle_samples) * 100
    x_hi = max(s.max() for s in kaggle_samples) * 100
    for i, samp in enumerate(kaggle_samples):
        hist, edges = np.histogram(samp * 100, bins=300, density=True, range=(x_lo, x_hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_dens.fill_between(centers, hist / 100, alpha=0.55, color=colors10[i],
                             edgecolor="none", label=f"Sub {i + 1}")
    ax_dens.set_xlabel("Accuracy (%)")
    ax_dens.set_ylabel("Density")
    ax_dens.set_title(
        "Posterior accuracy distributions — top 10 Kaggle submissions (N ≈ 15,123)"
    )
    ax_dens.legend(fontsize=8, ncol=2, loc="upper left")

    # Bottom: P(ranked 1st) for each submission
    bars = ax_bar.bar(np.arange(1, 11), p_best * 100, color=colors10, edgecolor="none")
    ax_bar.set_xlabel("Submission rank (by point estimate)")
    ax_bar.set_ylabel("P(truly best)  %")
    ax_bar.set_xticks(np.arange(1, 11))
    ax_bar.set_title(
        f"Probabilistic leaderboard — submission 1 is truly best in {p_best[0]:.0%} of posterior draws"
    )
    for bar, p in zip(bars, p_best):
        if p > 0.005:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{p:.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save("kaggle_ranking.png", fig)
