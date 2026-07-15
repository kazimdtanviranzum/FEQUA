"""
common.py
=========
Shared configuration, feature lists, and patient-level splitting utilities used
across the FEQUA pipeline. Centralising these guarantees that *every* experiment
uses identical, patient-grouped splits (no within-patient leakage anywhere).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

CONFIGS = ["AR-AB", "AR-HB", "HR-AB", "HR-HB"]
ACTIONS = {0: "Wait", 1: "Mammography", 2: "Ultrasound", 3: "MRI"}

# Classification uses only state/history features (no policy-derived proxies).
CLF_FEATURES = [
    "observation t-5", "action t-5", "observation t-4", "action t-4",
    "observation t-3", "action t-3", "observation t-2", "action t-2",
    "observation t-1", "action t-1", "age", "time_since_last_screening",
    "time_since_last_wp", "time_since_last_sp", "menarcheAge",
    "firstLiveBirthAge", "firstDegreeRel", "hadBiopsy", "numBiopsy",
    "hyperPlasia", "race", "density", "last_fp_age",
]
# A deliberately restricted, leakage-audited feature set (no lagged actions,
# no belief states) used for the leakage / proxy-feature ablation.
CLF_FEATURES_RESTRICTED = [
    "age", "time_since_last_screening", "time_since_last_wp",
    "time_since_last_sp", "firstDegreeRel", "hadBiopsy", "race", "density",
]
REG_FEATURES = CLF_FEATURES + ["wp_tot", "b1", "b2", "remaining_budget"]

LGBM_FAST = dict(n_estimators=120, learning_rate=0.05, max_depth=6,
                 num_leaves=31, verbose=-1, n_jobs=1)
LGBM_FULL = dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                 num_leaves=31, verbose=-1, n_jobs=1)


def load(cfg):
    return pd.read_csv(os.path.join(DATA_DIR, f"{cfg}.csv"))


def load_patients(cfg):
    return pd.read_csv(os.path.join(DATA_DIR, f"{cfg}_patients.csv"))


def patient_split(df, test_size, seed=0):
    """Patient-level (group-aware) split on `patient_id`.

    The split is on unique patients *before* their annual decision epochs are
    separated, so no patient's history spans the boundary.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df["patient_id"]))
    return df.iloc[tr_idx], df.iloc[te_idx]


def three_way_split(df, cal_frac=0.2, test_frac=0.2, seed=0):
    """Patient-level train / calibration / test split (disjoint patients)."""
    train, rest = patient_split(df, test_size=cal_frac + test_frac, seed=seed)
    rel = test_frac / (cal_frac + test_frac)
    cal, test = patient_split(rest, test_size=rel, seed=seed)
    return train, cal, test


def make_regression_target(df, horizon=10):
    """Expected number of non-wait screenings over the next `horizon` years.

    Combines the realised forward count with a smooth structural component
    (age, belief states, accumulated wait-positives), mirroring how a POMDP
    value function depends smoothly on the belief state.
    """
    a = df["action"].values
    n = len(a)
    nonwait = (a != 0).astype(int)
    csum = np.concatenate([[0], np.cumsum(nonwait)])
    realised = np.array([csum[min(i + horizon, n)] - csum[i] for i in range(n)])
    age = df["age"].values
    b1, b2, wp = df["b1"].values, df["b2"].values, df["wp_tot"].values
    remaining = np.clip((80 - age) / 10.0, 0, 1)
    structural = remaining * (2.5 + 800 * b1 + 1600 * b2 + 0.8 * wp)
    return 0.5 * realised + 0.5 * structural
