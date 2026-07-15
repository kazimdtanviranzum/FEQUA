"""
conformal.py
============
Conformal prediction rebuilt to the revision's specification.

Additions over the first draft
------------------------------
* **Explicit finite-sample quantile convention** q-hat = ceil((n+1)(1-alpha))/n
  and set rule C(x) = {y : 1 - f(x)[y] <= q-hat}.
* **Binomial (Wilson) confidence intervals** on empirical coverage.
* **Mondrian (subgroup-conditional) calibration** in addition to marginal
  calibration, so coverage can be equalised across groups.
* **Class-conditional coverage** so that rare intensive-screening actions
  (ultrasound / MRI) are not hidden behind strong coverage on the Wait class.
* **Set-size distribution and deferral workload** (singleton / multi / empty
  rates, reviews per 1,000 decisions, review burden by subgroup).
* **Distribution-shift stress test**: coverage under an age / demographic /
  practice-pattern shift applied to the test fold, treated as a monitoring
  signal rather than a guarantee.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from common import CLF_FEATURES, LGBM_FULL, three_way_split, patient_split
from metrics import wilson_ci
from decision_analysis import race_band


def _qhat(scores, alpha):
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def _fit_base(train, seed):
    m = LGBMClassifier(random_state=seed, **LGBM_FULL)
    m.fit(train[CLF_FEATURES], train["action"])
    return m, list(m.classes_)


def _scores(model, classes, X, y):
    p = model.predict_proba(X)
    P = np.zeros((len(X), len(classes)))
    for j, c in enumerate(model.classes_):
        P[:, classes.index(c)] = p[:, j]
    idx = [classes.index(v) for v in y]
    return 1 - P[np.arange(len(y)), idx], P


def split_conformal(cfg_df, alpha=0.05, seed=0, mondrian=False, group_col=None):
    """Split-conformal prediction sets, marginal or Mondrian(group-conditional)."""
    train, cal, test = three_way_split(cfg_df, cal_frac=0.2, test_frac=0.2,
                                       seed=seed)
    model, classes = _fit_base(train, seed)

    s_cal, _ = _scores(model, classes, cal[CLF_FEATURES], cal["action"].values)
    Ptest = model.predict_proba(test[CLF_FEATURES])
    P = np.zeros((len(test), len(classes)))
    for j, c in enumerate(model.classes_):
        P[:, classes.index(c)] = Ptest[:, j]

    ytest = test["action"].values
    if mondrian and group_col is not None:
        cal_g = _grp(cal, group_col)
        test_g = _grp(test, group_col)
        sets = np.zeros((len(test), len(classes)), bool)
        for g in np.unique(cal_g):
            gm_cal = cal_g == g
            if gm_cal.sum() < 30:
                q = _qhat(s_cal, alpha)                # fallback to marginal
            else:
                q = _qhat(s_cal[gm_cal], alpha)
            gm_te = test_g == g
            sets[gm_te] = (1 - P[gm_te]) <= q
    else:
        q = _qhat(s_cal, alpha)
        sets = (1 - P) <= q

    set_sizes = sets.sum(1)
    covered = np.array([sets[i, classes.index(ytest[i])]
                        for i in range(len(ytest))])
    return dict(classes=classes, sets=sets, set_sizes=set_sizes,
                covered=covered, test=test, P=P, ytest=ytest)


def _grp(df, group_col):
    if group_col == "race_band":
        return race_band(df["race"].values)
    if group_col == "age_grp":
        return pd.cut(df["age"], [40, 55, 65, 75, 120],
                      labels=["45-55", "55-65", "65-75", "75+"]).astype(str).values
    return df[group_col].values


def summarize_conformal(res, alpha):
    """Coverage (+Wilson CI), set-size distribution, deferral workload."""
    cov = res["covered"]; ss = res["set_sizes"]
    n = len(cov)
    k = int(cov.sum())
    lo, hi = wilson_ci(k, n)
    return dict(
        alpha=float(alpha),
        target=float(1 - alpha),
        coverage=float(cov.mean()),
        coverage_lo=lo, coverage_hi=hi,
        avg_set_size=float(ss.mean()),
        singleton_rate=float((ss == 1).mean()),
        multi_rate=float((ss > 1).mean()),
        empty_rate=float((ss == 0).mean()),
        override_rate=float((ss > 1).mean()),
        reviews_per_1000=float((ss != 1).mean() * 1000),
    )


def class_conditional_coverage(res):
    """Empirical coverage per action class (+Wilson CI)."""
    out = {}
    for c in res["classes"]:
        m = res["ytest"] == c
        if m.sum() == 0:
            continue
        cov = res["covered"][m]
        lo, hi = wilson_ci(int(cov.sum()), int(m.sum()))
        out[int(c)] = dict(coverage=float(cov.mean()), n=int(m.sum()),
                           lo=lo, hi=hi)
    return out


def subgroup_coverage(res, group_col, min_n=30):
    """Empirical coverage per subgroup with Wilson CI and review burden."""
    g = _grp(res["test"], group_col)
    out = {}
    for grp in np.unique(g):
        m = g == grp
        if m.sum() < min_n:
            continue
        cov = res["covered"][m]
        lo, hi = wilson_ci(int(cov.sum()), int(m.sum()))
        out[str(grp)] = dict(coverage=float(cov.mean()), n=int(m.sum()),
                             lo=lo, hi=hi,
                             reviews_per_1000=float(
                                 (res["set_sizes"][m] != 1).mean() * 1000))
    return out


def coverage_under_shift(cfg_df, alpha=0.05, seed=0, shift="age"):
    """Coverage when the test fold is shifted (non-exchangeable) vs unshifted."""
    train, cal, test = three_way_split(cfg_df, cal_frac=0.2, test_frac=0.3,
                                       seed=seed)
    model, classes = _fit_base(train, seed)
    s_cal, _ = _scores(model, classes, cal[CLF_FEATURES], cal["action"].values)
    q = _qhat(s_cal, alpha)

    def cover(sub):
        P = model.predict_proba(sub[CLF_FEATURES])
        M = np.zeros((len(sub), len(classes)))
        for j, c in enumerate(model.classes_):
            M[:, classes.index(c)] = P[:, j]
        sets = (1 - M) <= q
        y = sub["action"].values
        c = np.array([sets[i, classes.index(y[i])] for i in range(len(y))])
        return float(c.mean()), float(sets.sum(1).mean())

    base_cov, base_size = cover(test)
    if shift == "age":                       # over-represent older patients
        shifted = test[test["age"] >= 60]
    elif shift == "demographic":             # over-represent minority groups
        shifted = test[test["race"] >= 6]
    elif shift == "practice":                # over-represent dense-breast / HR-like
        shifted = test[test["density"] >= 3]
    else:
        shifted = test
    if len(shifted) < 30:
        shifted = test
    shift_cov, shift_size = cover(shifted)
    return dict(shift=shift, base_coverage=base_cov, base_set_size=base_size,
                shift_coverage=shift_cov, shift_set_size=shift_size,
                coverage_drop=base_cov - shift_cov)
