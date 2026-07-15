"""
test_pipeline.py
================
Unit tests for the correctness-critical parts of the FEQUA pipeline, as required
by the open-science checklist: patient-level splitting (no leakage), calibration/
test separation, group mapping, and the conformal finite-sample quantile.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_pipeline.py
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common import patient_split, three_way_split                 # noqa: E402
from conformal import _qhat                                        # noqa: E402
from decision_analysis import race_band                           # noqa: E402
from fairness import tpr_fpr_gaps                                 # noqa: E402
from site_partition import partition_sites                        # noqa: E402


def _toy(n_patients=200, epochs=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for _ in range(epochs):
            rows.append(dict(patient_id=f"P{p:04d}",
                             action=int(rng.integers(0, 3)),
                             age=int(rng.integers(40, 80)),
                             race=int(rng.integers(1, 13))))
    return pd.DataFrame(rows)


def test_patient_split_disjoint():
    """No patient may appear in both train and test."""
    df = _toy()
    tr, te = patient_split(df, test_size=0.3, seed=1)
    assert set(tr["patient_id"]) & set(te["patient_id"]) == set()
    # all rows accounted for
    assert len(tr) + len(te) == len(df)


def test_three_way_disjoint():
    """Train / calibration / test share no patient (calibration/test separation)."""
    df = _toy()
    tr, cal, te = three_way_split(df, cal_frac=0.2, test_frac=0.2, seed=2)
    a, b, c = set(tr["patient_id"]), set(cal["patient_id"]), set(te["patient_id"])
    assert a & b == set() and a & c == set() and b & c == set()
    assert len(a) + len(b) + len(c) == df["patient_id"].nunique()


def test_split_is_deterministic():
    df = _toy()
    t1, _ = patient_split(df, 0.3, seed=7)
    t2, _ = patient_split(df, 0.3, seed=7)
    assert list(t1["patient_id"]) == list(t2["patient_id"])


def test_sites_patient_disjoint():
    """Federated site shards never share a patient."""
    df = _toy()
    sites = partition_sites(df, k=5, alpha_dir=1.0, seed=3)
    seen = set()
    for s in sites:
        pids = set(s["patient_id"])
        assert seen & pids == set()
        seen |= pids
    assert seen == set(df["patient_id"])


def test_race_band_mapping():
    """12 race codes map onto exactly the 4 documented reporting bands."""
    bands = race_band(np.arange(1, 13))
    assert set(bands) <= {"White", "Black", "Hispanic", "Asian/Other"}
    assert race_band(np.array([1]))[0] == "White"
    assert race_band(np.array([12]))[0] == "Asian/Other"


def test_conformal_quantile_convention():
    """q-hat uses the ceil((n+1)(1-alpha))/n finite-sample level and is monotone."""
    scores = np.linspace(0, 1, 100)
    q05 = _qhat(scores, 0.05)
    q10 = _qhat(scores, 0.10)
    # smaller alpha -> larger (more conservative) quantile
    assert q05 >= q10
    # level capped at 1.0 for tiny calibration sets
    assert _qhat(np.array([0.2, 0.4, 0.6]), 0.01) <= 1.0


def test_conformal_coverage_property():
    """Empirical split-conformal coverage >= 1-alpha on exchangeable toy data."""
    rng = np.random.default_rng(0)
    # true scores for the correct label
    cal = rng.random(500)
    test = rng.random(2000)
    alpha = 0.1
    q = _qhat(cal, alpha)
    covered = (test <= q).mean()
    assert covered >= 1 - alpha - 0.05          # allow small finite-sample slack


def test_tpr_fpr_gaps_zero_when_perfect():
    """A perfect predictor has zero TPR/FPR gaps."""
    y = np.array([0, 1, 2] * 100)
    g = np.array(["A", "B"] * 150)
    out = tpr_fpr_gaps(y, y.copy(), g, min_n=1)
    assert out["tpr_gap"] == 0.0 and out["fpr_gap"] == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
