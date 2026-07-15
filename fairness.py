"""
fairness.py
===========
Fairness analysis rebuilt to the revision's specification.

Corrections addressed
----------------------
* **Equalized odds vs equality of opportunity.** The first draft measured only
  per-class recall (TPR) gaps but called the criterion "equalized odds". Here we
  report *both* TPR and FPR gaps per action and per group. An analysis that
  optimises only TPR parity is labelled *equality of opportunity*; only the joint
  TPR+FPR report is called *equalized odds*.
* **Oracle audit by group.** Before any mitigation we check whether the oracle
  policy itself prescribes different action mixes across groups (a clinically
  justified difference is not unfairness). This separates "the policy differs by
  group" from "the surrogate reproduces the policy unequally".
* **Three separated scenarios.** (1) clean labels, (2) induced label bias (a
  controlled mechanism stress test, reported separately and not as evidence of
  real-world bias), and (3) naturally heterogeneous sites. The induced-bias
  mechanism and strength are parameters, not a single hard-coded setting.
* **Group-aware vs group-blind mitigation.** Group-aware threshold
  post-processing (needs the attribute at inference) is compared against a
  group-blind reweighting alternative (does not).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

from common import patient_split, three_way_split

FAIR_FEATURES = ["age", "time_since_last_screening", "time_since_last_wp",
                 "time_since_last_sp", "race", "density", "b1", "b2", "wp_tot"]


# ---------------------------------------------------------------- gap metrics
def tpr_fpr_gaps(y_true, y_pred, groups, min_n=20):
    """Max between-group gap in per-action TPR and FPR (averaged over actions)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    groups = np.asarray(groups, dtype=object)
    uniq_g = [g for g in pd.unique(groups)]
    tpr_gaps, fpr_gaps = [], []
    for c in np.unique(y_true):
        tprs, fprs = [], []
        for g in uniq_g:
            gm = (groups == g)
            pos = gm & (y_true == c)
            neg = gm & (y_true != c)
            if pos.sum() > min_n:
                tprs.append((y_pred[pos] == c).mean())
            if neg.sum() > min_n:
                fprs.append((y_pred[neg] == c).mean())
        if len(tprs) > 1:
            tpr_gaps.append(max(tprs) - min(tprs))
        if len(fprs) > 1:
            fpr_gaps.append(max(fprs) - min(fprs))
    return dict(tpr_gap=float(np.mean(tpr_gaps)) if tpr_gaps else 0.0,
                fpr_gap=float(np.mean(fpr_gaps)) if fpr_gaps else 0.0,
                worst_tpr_gap=float(np.max(tpr_gaps)) if tpr_gaps else 0.0,
                worst_fpr_gap=float(np.max(fpr_gaps)) if fpr_gaps else 0.0)


def _age_band(df):
    return pd.cut(df["age"], [40, 55, 65, 75, 120],
                  labels=["45-55", "55-65", "65-75", "75+"]).astype(str)


# ---------------------------------------------------------------- oracle audit
def oracle_group_audit(cfg_df, group_col="age_grp"):
    """Does the oracle policy itself differ across groups? (action prevalence)"""
    df = cfg_df.copy()
    df["age_grp"] = _age_band(df)
    out = {}
    for g, sub in df.groupby(group_col):
        dist = sub["action"].value_counts(normalize=True).sort_index()
        out[str(g)] = {int(a): float(p) for a, p in dist.items()}
    return out


# ---------------------------------------------------------------- bias injector
def inject_label_bias(df, rng, mechanism="downgrade", strength=0.5):
    """Return a biased label vector on an *under-served* mask.

    mechanisms:
      'downgrade' : screening action -> Wait (under-screening)
      'flip_up'   : Wait -> Mammography (over-screening)
      'noise'     : random relabel among observed actions
    """
    ag = _age_band(df).values
    minority = df["race"].values >= 6
    under = np.isin(ag, ["55-65", "65-75"]) | minority
    biased = df["action"].values.copy()
    hit = under & (rng.random(len(df)) < strength)
    if mechanism == "downgrade":
        biased[hit & (biased != 0)] = 0
    elif mechanism == "flip_up":
        biased[hit & (biased == 0)] = 1
    elif mechanism == "noise":
        classes = np.unique(df["action"].values)
        biased[hit] = rng.choice(classes, size=hit.sum())
    return biased, under


# ---------------------------------------------------------------- mitigations
def _group_aware_threshold(proba, classes, groups, under_groups, boost):
    """Boost non-Wait probabilities for under-served groups (needs attribute)."""
    adj = proba.copy()
    ug = np.isin(groups, under_groups)
    for ci, c in enumerate(classes):
        if c != 0:
            adj[ug, ci] *= (1.0 + boost)
    adj = adj / adj.sum(1, keepdims=True)
    return np.array(classes)[adj.argmax(1)]


def _train_reweighted(train, sample_weight, params, seed):
    m = LGBMClassifier(random_state=seed, **params)
    m.fit(train[FAIR_FEATURES], train["action_used"], sample_weight=sample_weight)
    return m


# ---------------------------------------------------------------- main sweeps
def fairness_sweep(cfg_df, lambdas, scenario="induced_bias",
                   mechanism="downgrade", strength=0.5, mitigation="group_aware",
                   seed=0):
    """Fairness-utility sweep for one scenario / mechanism / mitigation.

    Returns a list of {lambda, mf1, tpr_gap, fpr_gap} evaluated against the
    *clean* ground-truth policy.
    """
    rng = np.random.default_rng(seed)
    df = cfg_df.copy()
    df["age_grp"] = _age_band(df)
    df = df[df["age_grp"] != "nan"].copy()

    if scenario == "induced_bias":
        biased, under = inject_label_bias(df, rng, mechanism, strength)
        df["action_used"] = biased
    else:                                    # clean / natural
        df["action_used"] = df["action"].values

    params = dict(n_estimators=120, learning_rate=0.05, max_depth=6, verbose=-1,
                  n_jobs=1)
    train, cal, test = three_way_split(df, cal_frac=0.2, test_frac=0.3, seed=seed)
    groups = test["age_grp"].astype(str).values
    y_clean = test["action"].values
    under_groups = ["55-65", "65-75"]

    # group-aware post-processing reuses ONE base model across lambdas
    base = None
    if mitigation == "group_aware":
        base = LGBMClassifier(random_state=seed, **params)
        base.fit(train[FAIR_FEATURES], train["action_used"])
        base_classes = list(base.classes_)
        p_test_base = base.predict_proba(test[FAIR_FEATURES])

    results = []
    for lam in lambdas:
        if mitigation == "group_aware":
            yhat = _group_aware_threshold(p_test_base, base_classes, groups,
                                          under_groups, boost=1.6 * lam)
        else:                                # group_blind reweighting
            ag_tr = _age_band(train).values
            minority_tr = train["race"].values >= 6
            under_tr = np.isin(ag_tr, under_groups) | minority_tr
            sw = np.where(under_tr, 1.0 + lam, 1.0)
            m = _train_reweighted(train, sw, params, seed)
            yhat = m.predict(test[FAIR_FEATURES])
        g = tpr_fpr_gaps(y_clean, yhat, groups)
        results.append(dict(lam=float(lam),
                            mf1=round(float(f1_score(y_clean, yhat,
                                                     average="macro")), 4),
                            tpr_gap=round(g["tpr_gap"], 4),
                            fpr_gap=round(g["fpr_gap"], 4)))
    return results
