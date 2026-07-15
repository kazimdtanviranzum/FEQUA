"""
baselines.py
============
Status-quo and simple-rule comparator policies required by the revision so that
FEQUA is compared against meaningful clinical baselines rather than only a
centralized ML model.

Each policy is a function ``policy(row) -> action in {0,1,2,3}`` evaluated on the
same decision-epoch rows used everywhere else. They are deterministic and depend
only on observable state, so they can be scored with the same regret / capacity /
benefit-harm machinery as the learned models.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def oracle_policy(df):
    """The POMDP-recommended action (the label). Zero regret by construction."""
    return df["action"].values


def fixed_interval_policy(df, interval=1):
    """Screen every `interval` years with mammography, else wait.

    interval=1 -> annual mammography; interval=2 -> biennial. A confirmatory
    mammography is still triggered by a recent wait-positive.
    """
    a = np.zeros(len(df), dtype=int)
    tss = df["time_since_last_screening"].values
    twp = df["time_since_last_wp"].values
    a[(tss >= interval)] = 1
    a[twp <= 2] = 1
    return a


def risk_stratified_policy(df):
    """Simple interpretable heuristic on age / risk / time-since-screen.

    High-risk or dense-breast patients screen more often and escalate to
    ultrasound; average-risk patients follow a biennial mammography rule. This
    is the kind of rule a clinician could write on a whiteboard.
    """
    a = np.zeros(len(df), dtype=int)
    tss = df["time_since_last_screening"].values
    twp = df["time_since_last_wp"].values
    hr = (df["risk"].values == "HR")
    dense = (df["density"].values >= 3)
    young = (df["age"].values < 50)

    avg_due = (~hr) & (tss >= 2)
    hr_due = hr & (tss >= 1)
    a[avg_due] = 1
    a[hr_due] = 1
    a[hr & dense & (tss >= 1)] = 2          # supplemental ultrasound
    a[young & hr & (tss >= 1)] = 3          # young high-risk -> MRI
    a[twp <= 2] = 1                          # confirmatory
    return a


BASELINE_POLICIES = {
    "oracle": oracle_policy,
    "annual": lambda df: fixed_interval_policy(df, interval=1),
    "biennial": lambda df: fixed_interval_policy(df, interval=2),
    "risk_stratified": risk_stratified_policy,
}
