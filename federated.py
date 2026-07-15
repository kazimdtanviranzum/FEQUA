"""
federated.py
============
Federated learning components for FEQUA.

The single most important revision correction: the first draft labelled the
method "FedAvg/FedProx" while actually combining LightGBM *probability outputs*
by a weighted vote. Tree ensembles have no parameters to average, so that naming
was wrong. This module fixes it two ways, and lets the paper compare them:

* **Federated ensemble (honest name).** Each site trains a local LightGBM and
  transmits only its class-probability vector on the shared evaluation points.
  The server forms a size-weighted (or uniform) soft-vote. No tree structures,
  gradients, or raw records are exchanged. This is *decentralized soft-voting*,
  not FedAvg, and is named accordingly.

* **Genuine FedAvg / FedProx (parametric).** A multiclass logistic-regression
  client is trained with local SGD; the server averages the *weight matrices*
  across rounds (FedAvg). FedProx adds a proximal term pulling each client's
  weights toward the current global weights. This is mathematically the standard
  FedAvg/FedProx procedure and is the only place those names are used.

Reporting `local_only` and `centralized` provides the lower / upper reference
benchmarks the revision asks for.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from lightgbm import LGBMClassifier

from common import CLF_FEATURES, LGBM_FAST, LGBM_FULL
from site_partition import partition_sites


# ------------------------------------------------ softmax / logistic utilities
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _FedLogistic:
    """Minimal multinomial logistic regression with SGD client updates."""

    def __init__(self, n_features, n_classes, lr=0.1, l2=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.lr, self.l2 = lr, l2

    def proba(self, X):
        return _softmax(X @ self.W + self.b)

    def local_update(self, X, y_onehot, epochs=3, batch=256,
                     mu=0.0, global_W=None, global_b=None, seed=0):
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                Xb, Yb = X[idx], y_onehot[idx]
                P = _softmax(Xb @ self.W + self.b)
                gW = Xb.T @ (P - Yb) / len(idx) + self.l2 * self.W
                gb = (P - Yb).mean(axis=0)
                if mu > 0 and global_W is not None:      # FedProx proximal term
                    gW += mu * (self.W - global_W)
                    gb += mu * (self.b - global_b)
                self.W -= self.lr * gW
                self.b -= self.lr * gb
        return self.W.copy(), self.b.copy()


def _onehot(y, classes):
    idx = {c: j for j, c in enumerate(classes)}
    M = np.zeros((len(y), len(classes)))
    for i, v in enumerate(y):
        M[i, idx[v]] = 1.0
    return M


# ------------------------------------------------ genuine FedAvg / FedProx
def federated_parametric(train, test, classes, mode="fedavg", rounds=15,
                         k=5, alpha_dir=100.0, imbalance=1.0, seed=0):
    """Standard FedAvg/FedProx on a logistic client with true weight averaging."""
    sites = partition_sites(train, k=k, alpha_dir=alpha_dir,
                            imbalance=imbalance, seed=seed)
    scaler = StandardScaler().fit(train[CLF_FEATURES])
    Xte = scaler.transform(test[CLF_FEATURES])
    yte = test["action"].values

    site_data = []
    for s in sites:
        if len(s) == 0:
            continue
        Xs = scaler.transform(s[CLF_FEATURES])
        Ys = _onehot(s["action"].values, classes)
        site_data.append((Xs, Ys, len(s)))

    nfeat, ncls = Xte.shape[1], len(classes)
    global_W = np.zeros((nfeat, ncls))
    global_b = np.zeros(ncls)
    total = sum(n for _, _, n in site_data)
    mu = 0.05 if mode == "fedprox" else 0.0

    mf1_hist, acc_hist = [], []
    for r in range(rounds):
        Ws, bs, ws = [], [], []
        for j, (Xs, Ys, n) in enumerate(site_data):
            client = _FedLogistic(nfeat, ncls, seed=seed + j)
            client.W, client.b = global_W.copy(), global_b.copy()
            W, b = client.local_update(Xs, Ys, epochs=3, mu=mu,
                                        global_W=global_W, global_b=global_b,
                                        seed=seed + r * 100 + j)
            Ws.append(W); bs.append(b); ws.append(n / total)
        global_W = sum(wi * Wi for wi, Wi in zip(ws, Ws))     # FedAvg averaging
        global_b = sum(wi * bi for wi, bi in zip(ws, bs))
        P = _softmax(Xte @ global_W + global_b)
        yhat = np.array(classes)[P.argmax(1)]
        mf1_hist.append(f1_score(yte, yhat, average="macro"))
        acc_hist.append(accuracy_score(yte, yhat))

    P = _softmax(Xte @ global_W + global_b)
    return dict(mf1=mf1_hist, acc=acc_hist,
                final_mf1=mf1_hist[-1], final_acc=acc_hist[-1],
                proba=P, classes=classes)


# ------------------------------------------------ federated ensemble (soft-vote)
def federated_ensemble(train, test, classes, weighting="size", rounds=1,
                       k=5, alpha_dir=100.0, imbalance=1.0, seed=0,
                       lgbm_params=None):
    """Decentralized soft-voting ensemble of local LightGBM models.

    `weighting`: 'size' (size-weighted, the FedAvg-analogue vote) or 'uniform'.
    Trees are never averaged; only per-site class-probability vectors combine.
    """
    lgbm_params = lgbm_params or LGBM_FAST
    sites = partition_sites(train, k=k, alpha_dir=alpha_dir,
                            imbalance=imbalance, seed=seed)
    Xte = test[CLF_FEATURES]
    yte = test["action"].values

    site_probas, weights = [], []
    for s in sites:
        if s["action"].nunique() < 2:
            continue
        m = LGBMClassifier(random_state=seed, **lgbm_params)
        m.fit(s[CLF_FEATURES], s["action"])
        p = np.zeros((len(Xte), len(classes)))
        mp = m.predict_proba(Xte)
        for j, c in enumerate(m.classes_):
            p[:, classes.index(c)] = mp[:, j]
        site_probas.append(p)
        weights.append(len(s))

    if not site_probas:
        raise ValueError("no site had >=2 classes")
    if weighting == "size":
        w = np.array(weights) / sum(weights)
    else:
        w = np.ones(len(site_probas)) / len(site_probas)
    agg = sum(wi * pi for wi, pi in zip(w, site_probas))
    yhat = np.array(classes)[agg.argmax(1)]
    return dict(mf1=float(f1_score(yte, yhat, average="macro")),
                acc=float(accuracy_score(yte, yhat)),
                proba=agg, classes=classes)


def local_only(train, test, classes, k=5, alpha_dir=100.0, imbalance=1.0,
               seed=0, lgbm_params=None):
    """Each site keeps its own model; report macro-averaged site performance.

    This is the 'no collaboration' lower reference: a patient is scored by the
    model of the site they belong to (here we average per-site test performance).
    """
    lgbm_params = lgbm_params or LGBM_FAST
    sites = partition_sites(train, k=k, alpha_dir=alpha_dir,
                            imbalance=imbalance, seed=seed)
    mf1s, accs, sizes = [], [], []
    Xte, yte = test[CLF_FEATURES], test["action"].values
    for s in sites:
        if s["action"].nunique() < 2:
            continue
        m = LGBMClassifier(random_state=seed, **lgbm_params)
        m.fit(s[CLF_FEATURES], s["action"])
        yhat = m.predict(Xte)
        mf1s.append(f1_score(yte, yhat, average="macro"))
        accs.append(accuracy_score(yte, yhat))
        sizes.append(len(s))
    w = np.array(sizes) / sum(sizes)
    return dict(mf1=float(np.average(mf1s, weights=w)),
                acc=float(np.average(accs, weights=w)),
                mf1_sites=[float(x) for x in mf1s])


def centralized(train, test, classes, seed=0, lgbm_params=None):
    """Pooled model = infeasible upper reference under data locality."""
    lgbm_params = lgbm_params or LGBM_FULL
    m = LGBMClassifier(random_state=seed, **lgbm_params)
    m.fit(train[CLF_FEATURES], train["action"])
    p = np.zeros((len(test), len(classes)))
    mp = m.predict_proba(test[CLF_FEATURES])
    for j, c in enumerate(m.classes_):
        p[:, classes.index(c)] = mp[:, j]
    yhat = np.array(classes)[p.argmax(1)]
    yte = test["action"].values
    return dict(mf1=float(f1_score(yte, yhat, average="macro")),
                acc=float(accuracy_score(yte, yhat)),
                proba=p, classes=classes, model=m)
