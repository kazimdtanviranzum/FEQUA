"""
metrics.py
==========
Statistical machinery shared across FEQUA experiments:

* calibration metrics (multiclass Brier score, log loss, expected calibration
  error);
* patient-cluster bootstrap confidence intervals (resampling *patients*, not
  epochs, to respect within-patient dependence);
* Wilson binomial confidence intervals for coverage / proportion metrics;
* a paired bootstrap test for comparing two methods on identical splits.

These implement the statistical-analysis-plan requirements from the revision
(repeat-and-report means, SDs, and 95 % CIs; patient as the resampling unit).
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import log_loss, f1_score


# ---------------------------------------------------------------- calibration
def multiclass_brier(y_true, proba, classes):
    """Multiclass Brier score = mean squared error of the probability vector."""
    idx = {c: j for j, c in enumerate(classes)}
    onehot = np.zeros_like(proba)
    for i, y in enumerate(y_true):
        onehot[i, idx[y]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, classes, n_bins=10):
    """Top-label expected calibration error (ECE)."""
    conf = proba.max(axis=1)
    pred = np.array(classes)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def safe_log_loss(y_true, proba, classes):
    try:
        return float(log_loss(y_true, proba, labels=classes))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- bootstrap CI
def patient_bootstrap_ci(patient_ids, metric_fn, n_boot=300, seed=0, alpha=0.05):
    """Cluster (patient-level) bootstrap CI for a scalar metric.

    `metric_fn(mask)` receives a boolean mask over rows (True = include) and
    returns a scalar. We resample *patients* with replacement and rebuild the
    row mask, so within-patient correlation is respected.
    """
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids)
    uniq = np.unique(patient_ids)
    # index rows by patient for speed
    rows_by_pid = {p: np.where(patient_ids == p)[0] for p in uniq}
    point = metric_fn(np.ones(len(patient_ids), dtype=bool))
    stats_ = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_pid[p] for p in samp])
        mask = np.zeros(len(patient_ids), dtype=bool)
        # duplicate patients: use index list rather than mask to keep multiplicity
        val = metric_fn(rows)  # metric_fn must accept integer row indices too
        stats_.append(val)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                sd=float(np.std(stats_)))


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion (coverage/override)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def summarize(values):
    """Mean, SD and 95 % CI (normal approx) over repeated-seed runs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(mean=float("nan"), sd=float("nan"), lo=float("nan"),
                    hi=float("nan"), n=0)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    se = sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
    t = stats.t.ppf(0.975, max(1, len(v) - 1))
    return dict(mean=mean, sd=sd, lo=mean - t * se, hi=mean + t * se, n=len(v))


def paired_diff_test(a, b):
    """Paired comparison of two methods across identical seeds.

    Returns mean difference (a-b), a 95 % CI, Cohen's d_z effect size, and a
    Wilcoxon signed-rank p-value (falls back to sign test note if ties).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    se = sd / np.sqrt(len(d)) if len(d) > 1 else 0.0
    t = stats.t.ppf(0.975, max(1, len(d) - 1))
    dz = mean / sd if sd > 0 else 0.0
    try:
        p = float(stats.wilcoxon(a, b).pvalue) if len(d) >= 6 and np.any(d != 0) \
            else float("nan")
    except Exception:
        p = float("nan")
    return dict(mean_diff=mean, lo=mean - t * se, hi=mean + t * se,
                cohen_dz=float(dz), p_wilcoxon=p, n=len(d))
