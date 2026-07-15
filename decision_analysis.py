"""
decision_analysis.py
====================
Turns predicted *actions* into decision-science and cancer-outcome quantities,
so that FEQUA can be evaluated on decision value, resource use, and screening
benefit-harm rather than label agreement alone.

Three families of output
------------------------
1. **Decision value / oracle regret.** An interpretable per-decision reward
   ``reward(state, action) = benefit(action | belief, density) - cost(action)``
   whose components are (i) expected early-detection benefit from a screen given
   the disease belief and modality/density sensitivity, and (ii) modality cost
   plus expected false-positive harm. Regret of a policy is the mean shortfall in
   this reward relative to the oracle (POMDP) action on identical states. The
   oracle has zero regret by construction; any deviation is a quantified value
   loss, not merely a misclassification. The reward weights are exposed so that
   managerial sensitivity (cost of screening / false reassurance / over-imaging)
   can be varied.

2. **Capacity / resource outcomes.** Screens per 1,000 decisions, modality mix,
   MRI/ultrasound demand, and total budget consumed under a policy.

3. **Cancer benefit-harm outcomes.** Computed from the patient-level cancer
   summary emitted by the simulator: early-detection rate, interval-cancer rate,
   stage at detection, false-positive recalls, and benign biopsies, reportable
   overall and by subgroup.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

MODALITY_COST = {0: 0.0, 1: 1.0, 2: 1.5, 3: 5.0}          # hundreds of $ (relative)
# modality detection sensitivity (density-modulated), mirrors the simulator
_BASE_SENS = {0: 0.0, 1: 0.80, 2: 0.86, 3: 0.90}
_DENS_PEN = {0: 0.0, 1: 0.12, 2: 0.05, 3: 0.02}

DEFAULT_WEIGHTS = dict(
    benefit=120.0,       # value of a unit of expected early detection
    fp_harm=0.6,         # harm weight of an expected false-positive recall
    over_image=0.15,     # penalty for intensive imaging when belief is low
)


def _sensitivity(action, density):
    a = np.asarray(action)
    d = np.asarray(density)
    base = np.vectorize(lambda x: _BASE_SENS[int(x)])(a)
    pen = np.vectorize(lambda x: _DENS_PEN[int(x)])(a) * np.clip(d - 2, 0, None)
    return np.clip(base - pen, 0.0, 0.98)


def action_reward(df, action, weights=None):
    """Per-decision reward vector for taking `action` at each row's state."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    action = np.asarray(action)
    belief = (df["b1"].values + df["b2"].values)             # P(cancer preclinical)
    density = df["density"].values
    cost = np.vectorize(lambda x: MODALITY_COST[int(x)])(action)
    sens = _sensitivity(action, density)
    benefit = w["benefit"] * belief * sens
    fp_harm = w["fp_harm"] * (1 - belief) * (action != 0) * 0.08   # 8% FP rate
    over = w["over_image"] * (action >= 2) * (belief < belief.mean())
    return benefit - cost - fp_harm - over


def oracle_regret(df, pred_action, weights=None):
    """Mean per-decision value shortfall of `pred_action` vs the oracle action."""
    r_oracle = action_reward(df, df["action"].values, weights)
    r_pred = action_reward(df, pred_action, weights)
    reg = r_oracle - r_pred
    return dict(mean_regret=float(reg.mean()),
                regret_per_1000=float(reg.mean() * 1000),
                oracle_value=float(r_oracle.mean()),
                policy_value=float(r_pred.mean()),
                value_retained=float(r_pred.mean() / r_oracle.mean())
                if r_oracle.mean() != 0 else float("nan"))


def capacity_outcomes(df, pred_action):
    """Screens per 1,000 decisions, modality mix, budget consumed."""
    a = np.asarray(pred_action)
    n = len(a)
    per1000 = lambda m: float((a == m).sum() / n * 1000)
    cost = np.vectorize(lambda x: MODALITY_COST[int(x)])(a).sum()
    return dict(
        n_decisions=int(n),
        screens_per_1000=float((a != 0).sum() / n * 1000),
        mammography_per_1000=per1000(1),
        ultrasound_per_1000=per1000(2),
        mri_per_1000=per1000(3),
        wait_frac=float((a == 0).mean()),
        budget_units=float(cost),
        budget_per_1000=float(cost / n * 1000),
    )


def benefit_harm_outcomes(patients, mask=None):
    """Cancer benefit-harm summary from the patient-level cancer table."""
    p = patients if mask is None else patients[mask]
    cancers = p[p["cancer"] == 1]
    n_c = len(cancers)
    early = cancers["detect_stage"].isin([0, 1]).sum() if n_c else 0
    return dict(
        n_patients=int(len(p)),
        cancer_incidence=float(p["cancer"].mean()),
        n_cancers=int(n_c),
        early_detection_rate=float(early / n_c) if n_c else float("nan"),
        interval_cancer_rate=float(cancers["interval_cancer"].mean())
        if n_c else float("nan"),
        detected_rate=float(cancers["detected"].mean()) if n_c else float("nan"),
        mean_stage_at_detection=float(
            cancers.loc[cancers["detect_stage"] >= 0, "detect_stage"].mean())
        if n_c else float("nan"),
        false_positives_per_patient=float(p["false_positives"].mean()),
        benign_biopsies_per_patient=float(p["benign_biopsies"].mean()),
    )


def benefit_harm_by_group(patients, group_col, min_n=40):
    """Benefit-harm outcomes split by a demographic column (e.g. race band)."""
    out = {}
    for g, sub in patients.groupby(group_col):
        if len(sub) >= min_n:
            out[str(g)] = benefit_harm_outcomes(sub)
    return out


def race_band(race_code):
    """Map the 12 race codes to 4 reporting bands (documented mapping)."""
    r = np.asarray(race_code)
    band = np.where(r <= 3, "White",
            np.where(r <= 6, "Black",
            np.where(r <= 9, "Hispanic", "Asian/Other")))
    return band
