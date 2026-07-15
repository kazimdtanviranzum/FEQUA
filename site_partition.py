"""
site_partition.py
=================
Reproducible partitioning of a configuration into K federated "sites".

The first draft split each configuration into K uniform shards, which made the
site-design ambiguous (the architecture figure implied one-config-per-site while
the methods implied random shards). This module resolves that: every site is a
patient-disjoint shard drawn with a controllable degree of non-IID-ness and a
controllable size imbalance, and the resulting label divergence across sites is
measured explicitly so heterogeneity can be reported as a number.

Design
------
* Partition is at the *patient* level (whole patients to one site).
* Non-IID severity is controlled by a Dirichlet concentration parameter
  ``alpha_dir`` applied over a per-patient *skew key* (here the patient's modal
  action). Small alpha -> each site concentrates on a few action profiles
  (severe non-IID); large alpha -> near-uniform (IID).
* Size imbalance is controlled by a geometric site-size ratio.
* ``site_label_divergence`` reports the mean Jensen-Shannon divergence between
  site action distributions and the pooled distribution.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _patient_skew_key(df):
    """Modal action per patient -> the attribute used to induce non-IID sites."""
    return df.groupby("patient_id")["action"].agg(
        lambda s: s.value_counts().index[0])


def partition_sites(df, k=5, alpha_dir=100.0, imbalance=1.0, seed=0):
    """Return a list of K patient-disjoint site DataFrames.

    Parameters
    ----------
    k : number of sites.
    alpha_dir : Dirichlet concentration. Large (>=100) ~ IID; small (<=0.5)
        strongly non-IID.
    imbalance : geometric size ratio between the largest and smallest site
        (1.0 = balanced; 20.0 = 20:1).
    """
    rng = np.random.default_rng(seed)
    skew = _patient_skew_key(df)
    pids = np.asarray(skew.index.tolist(), dtype=object)
    keys = np.asarray(skew.values)
    uniq_keys = np.unique(keys)

    # target site size weights (geometric imbalance)
    if imbalance <= 1.0:
        size_w = np.ones(k)
    else:
        ratio = imbalance ** (1.0 / max(1, k - 1))
        size_w = ratio ** np.arange(k)
    size_w = size_w / size_w.sum()

    # for each label group, draw a Dirichlet split across sites
    assign = {}
    for key in uniq_keys:
        members = pids[keys == key]
        rng.shuffle(members)
        props = rng.dirichlet(alpha_dir * size_w)
        counts = np.floor(props * len(members)).astype(int)
        while counts.sum() < len(members):      # distribute remainder
            counts[rng.integers(k)] += 1
        idx = 0
        for s in range(k):
            for pid in members[idx:idx + counts[s]]:
                assign[pid] = s
            idx += counts[s]

    site_of = df["patient_id"].map(assign).values
    return [df[site_of == s].copy() for s in range(k)]


def _js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def site_label_divergence(sites, classes=None):
    """Mean Jensen-Shannon divergence of site action distributions vs pooled."""
    nonempty = [s for s in sites if len(s) > 0]
    if not nonempty:
        return 0.0
    if classes is None:
        classes = sorted(pd.concat(s["action"] for s in nonempty).unique())
    pooled = pd.concat(s["action"] for s in nonempty)
    pooled_dist = np.array([(pooled == c).mean() for c in classes])
    divs = []
    for s in nonempty:
        d = np.array([(s["action"] == c).mean() for c in classes])
        divs.append(_js_divergence(d, pooled_dist))
    return float(np.mean(divs))


def site_size_gini(sites):
    """Gini coefficient of site sizes (0 = balanced)."""
    sizes = np.array([len(s) for s in sites], float)
    if sizes.sum() == 0:
        return 0.0
    sizes = np.sort(sizes)
    n = len(sizes)
    cum = np.cumsum(sizes)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
