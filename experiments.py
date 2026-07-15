"""
experiments.py
==============
Runs the full FEQUA analysis and writes every result JSON consumed by
``make_figures.py``. All principal metrics are computed over repeated
patient-level splits and reported as mean, standard deviation, and 95 %
confidence interval; the resampling unit is the patient throughout.

Result files written to ``results/``
------------------------------------
classification.json   central / local-only / FedAvg / FedProx (parametric) /
                      federated ensemble (size & uniform), mean+CI + convergence
regression.json       central vs federated-ensemble screening-count regression
calibration.json      Brier, log loss, ECE for the central classifier
leakage_ablation.json full-feature vs restricted-feature macro F1 (proxy audit)
heterogeneity.json    K-sweep, non-IID (Dirichlet) sweep, site-imbalance sweep
fairness.json         oracle-by-group audit + TPR/FPR sweeps (3 scenarios,
                      3 mechanisms, group-aware vs group-blind)
conformal.json        marginal vs Mondrian coverage (+Wilson CIs), class- and
                      subgroup-conditional coverage, set-size, shift stress test
decision.json         oracle regret + capacity/modality/budget per policy (+CI)
benefit_harm.json     cancer benefit-harm per policy, overall and by race band,
                      plus managerial cost-weight sensitivity
incremental.json      learning curves (classification & regression)
shap.json             global TreeSHAP importance

Scale constants below are tuned so the whole pipeline runs in a few minutes on a
single core; increase N_SEEDS / rollout sizes for tighter intervals.
"""

from __future__ import annotations
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from lightgbm import LGBMClassifier, LGBMRegressor

from common import (CONFIGS, CLF_FEATURES, CLF_FEATURES_RESTRICTED, REG_FEATURES,
                    LGBM_FAST, LGBM_FULL, RES_DIR, load, load_patients,
                    patient_split, three_way_split, make_regression_target)
from metrics import (summarize, paired_diff_test, multiclass_brier,
                     expected_calibration_error, safe_log_loss)
import federated as fed
import fairness as fair
import conformal as cp
import decision_analysis as da
import baselines as bl
import rollout as ro
from site_partition import partition_sites, site_label_divergence, site_size_gini

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---- scale (tunable) -------------------------------------------------------
N_SEEDS = 8            # repeated patient splits for headline metrics
N_SEEDS_HET = 5        # repeated splits for heterogeneity sweeps
N_SEEDS_ROLL = 3       # repeated cohorts for benefit-harm rollouts
ROLL_N = 500           # patients per fixed-policy rollout
ROLL_N_MODEL = 400     # patients per learned-model rollout (closed loop)
ROUNDS = 15
SEEDS = list(range(N_SEEDS))


def _classes(cfg_df):
    return sorted(cfg_df["action"].unique())


# ============================================================ classification
def run_classification():
    out = {}
    for cfg in CONFIGS:
        df = load(cfg)
        classes = _classes(df)
        acc = {m: [] for m in ["central", "local", "ens_size", "ens_uniform",
                               "fedavg", "fedprox"]}
        mf1 = {m: [] for m in acc}
        conv = {"fedavg": [], "fedprox": []}
        for s in SEEDS:
            train, test = patient_split(df, test_size=0.3, seed=s)
            c = fed.centralized(train, test, classes, seed=s)
            l = fed.local_only(train, test, classes, k=5, seed=s)
            es = fed.federated_ensemble(train, test, classes, weighting="size",
                                        k=5, seed=s)
            eu = fed.federated_ensemble(train, test, classes, weighting="uniform",
                                        k=5, seed=s)
            fa = fed.federated_parametric(train, test, classes, mode="fedavg",
                                          rounds=ROUNDS, k=5, seed=s)
            fp = fed.federated_parametric(train, test, classes, mode="fedprox",
                                          rounds=ROUNDS, k=5, seed=s)
            for m, r in [("central", c), ("local", l), ("ens_size", es),
                         ("ens_uniform", eu)]:
                mf1[m].append(r["mf1"]); acc[m].append(r["acc"])
            for m, r in [("fedavg", fa), ("fedprox", fp)]:
                mf1[m].append(r["final_mf1"]); acc[m].append(r["final_acc"])
            conv["fedavg"].append(fa["mf1"]); conv["fedprox"].append(fp["mf1"])
        rec = {}
        for m in acc:
            rec[m] = dict(mf1=summarize(mf1[m]), acc=summarize(acc[m]))
        rec["convergence"] = {
            "fedavg": np.mean(conv["fedavg"], axis=0).tolist(),
            "fedprox": np.mean(conv["fedprox"], axis=0).tolist(),
        }
        # paired ensemble-vs-fedavg test
        rec["paired_ens_vs_fedavg"] = paired_diff_test(mf1["ens_size"],
                                                       mf1["fedavg"])
        out[cfg] = rec
        print(f"  {cfg}: central MF1={rec['central']['mf1']['mean']:.3f} "
              f"ens={rec['ens_size']['mf1']['mean']:.3f} "
              f"fedavg={rec['fedavg']['mf1']['mean']:.3f} "
              f"local={rec['local']['mf1']['mean']:.3f}")
    return out


# ============================================================ regression
def _fed_ensemble_reg(train, test, k=5, seed=0):
    sites = partition_sites(train, k=k, seed=seed)
    preds, w = [], []
    for s in sites:
        if len(s) < 20:
            continue
        m = LGBMRegressor(random_state=seed, **LGBM_FAST)
        m.fit(s[REG_FEATURES], s["y_reg"])
        preds.append(m.predict(test[REG_FEATURES])); w.append(len(s))
    w = np.array(w) / sum(w)
    agg = sum(wi * pi for wi, pi in zip(w, preds))
    return agg


def run_regression():
    out = {}
    for cfg in CONFIGS:
        df = load(cfg).copy()
        df["y_reg"] = make_regression_target(df)
        r2c, rmsec, r2f, rmsef = [], [], [], []
        for s in SEEDS:
            train, test = patient_split(df, test_size=0.3, seed=s)
            m = LGBMRegressor(random_state=s, **LGBM_FULL)
            m.fit(train[REG_FEATURES], train["y_reg"])
            pc = m.predict(test[REG_FEATURES])
            r2c.append(r2_score(test["y_reg"], pc))
            rmsec.append(np.sqrt(mean_squared_error(test["y_reg"], pc)))
            pf = _fed_ensemble_reg(train, test, k=5, seed=s)
            r2f.append(r2_score(test["y_reg"], pf))
            rmsef.append(np.sqrt(mean_squared_error(test["y_reg"], pf)))
        out[cfg] = dict(central=dict(r2=summarize(r2c), rmse=summarize(rmsec)),
                        ensemble=dict(r2=summarize(r2f), rmse=summarize(rmsef)))
        print(f"  {cfg}: central R2={out[cfg]['central']['r2']['mean']:.3f} "
              f"ens R2={out[cfg]['ensemble']['r2']['mean']:.3f}")
    return out


# ============================================================ calibration
def run_calibration():
    out = {}
    for cfg in CONFIGS:
        df = load(cfg); classes = _classes(df)
        brier, ll, ece = [], [], []
        for s in SEEDS:
            train, test = patient_split(df, test_size=0.3, seed=s)
            r = fed.centralized(train, test, classes, seed=s)
            y = test["action"].values
            brier.append(multiclass_brier(y, r["proba"], classes))
            ll.append(safe_log_loss(y, r["proba"], classes))
            ece.append(expected_calibration_error(y, r["proba"], classes))
        out[cfg] = dict(brier=summarize(brier), log_loss=summarize(ll),
                        ece=summarize(ece))
        print(f"  {cfg}: Brier={out[cfg]['brier']['mean']:.3f} "
              f"ECE={out[cfg]['ece']['mean']:.3f}")
    return out


# ============================================================ leakage ablation
def run_leakage_ablation():
    out = {}
    for cfg in CONFIGS:
        df = load(cfg)
        full, restr = [], []
        for s in SEEDS:
            train, test = patient_split(df, test_size=0.3, seed=s)
            mf = LGBMClassifier(random_state=s, **LGBM_FULL)
            mf.fit(train[CLF_FEATURES], train["action"])
            full.append(f1_score(test["action"], mf.predict(test[CLF_FEATURES]),
                                 average="macro"))
            mr = LGBMClassifier(random_state=s, **LGBM_FULL)
            mr.fit(train[CLF_FEATURES_RESTRICTED], train["action"])
            restr.append(f1_score(test["action"],
                                  mr.predict(test[CLF_FEATURES_RESTRICTED]),
                                  average="macro"))
        out[cfg] = dict(full_features=summarize(full),
                        restricted_features=summarize(restr))
        print(f"  {cfg}: full={out[cfg]['full_features']['mean']:.3f} "
              f"restricted={out[cfg]['restricted_features']['mean']:.3f}")
    return out


# ============================================================ heterogeneity
def run_heterogeneity():
    out = {"k_sweep": {}, "noniid_sweep": {}, "imbalance_sweep": {}}
    cfg = "HR-HB"
    df = load(cfg); classes = _classes(df)

    def eval_setting(k, alpha_dir, imbalance):
        ens, fa, loc, div = [], [], [], []
        for s in range(N_SEEDS_HET):
            train, test = patient_split(df, test_size=0.3, seed=s)
            sites = partition_sites(train, k=k, alpha_dir=alpha_dir,
                                    imbalance=imbalance, seed=s)
            div.append(site_label_divergence(sites, classes))
            e = fed.federated_ensemble(train, test, classes, weighting="size",
                                       k=k, alpha_dir=alpha_dir,
                                       imbalance=imbalance, seed=s)
            f = fed.federated_parametric(train, test, classes, mode="fedavg",
                                         rounds=ROUNDS, k=k, alpha_dir=alpha_dir,
                                         imbalance=imbalance, seed=s)
            l = fed.local_only(train, test, classes, k=k, alpha_dir=alpha_dir,
                               imbalance=imbalance, seed=s)
            ens.append(e["mf1"]); fa.append(f["final_mf1"]); loc.append(l["mf1"])
        return dict(ensemble=summarize(ens), fedavg=summarize(fa),
                    local=summarize(loc), divergence=summarize(div))

    for k in [2, 5, 10, 20]:
        out["k_sweep"][str(k)] = eval_setting(k, 100.0, 1.0)
        print(f"  K={k}: ens={out['k_sweep'][str(k)]['ensemble']['mean']:.3f}")
    for a in [0.1, 0.5, 2.0, 100.0]:
        out["noniid_sweep"][str(a)] = eval_setting(5, a, 1.0)
        r = out["noniid_sweep"][str(a)]
        print(f"  alpha={a}: div={r['divergence']['mean']:.3f} "
              f"ens={r['ensemble']['mean']:.3f} local={r['local']['mean']:.3f}")
    for im in [1.0, 5.0, 20.0]:
        out["imbalance_sweep"][str(im)] = eval_setting(5, 100.0, im)
    return out


# ============================================================ fairness
def run_fairness():
    lambdas = [0, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0]
    out = {"oracle_audit": {}, "sweeps": {}}
    for cfg in ["AR-HB", "HR-HB"]:
        df = load(cfg)
        out["oracle_audit"][cfg] = fair.oracle_group_audit(df)
        out["sweeps"][cfg] = {}
        # scenario x mechanism x mitigation (aggregated over seeds -> mean sweep)
        combos = [("clean", "downgrade", "group_aware"),
                  ("induced_bias", "downgrade", "group_aware"),
                  ("induced_bias", "downgrade", "group_blind"),
                  ("induced_bias", "flip_up", "group_aware"),
                  ("induced_bias", "noise", "group_aware")]
        for scen, mech, mit in combos:
            per_seed = []
            for s in range(N_SEEDS_HET):
                per_seed.append(fair.fairness_sweep(df, lambdas, scenario=scen,
                                                    mechanism=mech, strength=0.5,
                                                    mitigation=mit, seed=s))
            # average metric across seeds at each lambda
            agg = []
            for i, lam in enumerate(lambdas):
                agg.append(dict(
                    lam=lam,
                    mf1=float(np.mean([p[i]["mf1"] for p in per_seed])),
                    tpr_gap=float(np.mean([p[i]["tpr_gap"] for p in per_seed])),
                    fpr_gap=float(np.mean([p[i]["fpr_gap"] for p in per_seed])),
                ))
            out["sweeps"][cfg][f"{scen}|{mech}|{mit}"] = agg
        key = f"{cfg} induced_bias|downgrade|group_aware"
        a = out["sweeps"][cfg]["induced_bias|downgrade|group_aware"]
        print(f"  {cfg}: TPR gap {a[0]['tpr_gap']:.3f}->{a[-1]['tpr_gap']:.3f} "
              f"FPR gap {a[0]['fpr_gap']:.3f}->{a[-1]['fpr_gap']:.3f}")
    return out


# ============================================================ conformal
def run_conformal():
    alphas = [0.01, 0.05, 0.10, 0.15, 0.20]
    out = {}
    for cfg in CONFIGS:
        df = load(cfg)
        by_alpha = []
        for a in alphas:
            cov, size, override = [], [], []
            for s in range(N_SEEDS_HET):
                res = cp.split_conformal(df, alpha=a, seed=s)
                summ = cp.summarize_conformal(res, a)
                cov.append(summ["coverage"]); size.append(summ["avg_set_size"])
                override.append(summ["override_rate"])
            by_alpha.append(dict(alpha=a, coverage=summarize(cov),
                                 avg_set_size=summarize(size),
                                 override_rate=summarize(override)))
        # detailed at alpha=0.05 (marginal vs Mondrian, class & subgroup, shift)
        res_m = cp.split_conformal(df, alpha=0.05, seed=0)
        res_mon = cp.split_conformal(df, alpha=0.05, seed=0, mondrian=True,
                                     group_col="race_band")
        detail = dict(
            marginal=cp.summarize_conformal(res_m, 0.05),
            mondrian=cp.summarize_conformal(res_mon, 0.05),
            class_conditional=cp.class_conditional_coverage(res_m),
            subgroup_marginal=cp.subgroup_coverage(res_m, "race_band"),
            subgroup_mondrian=cp.subgroup_coverage(res_mon, "race_band"),
            shift=[cp.coverage_under_shift(df, 0.05, seed=0, shift=sh)
                   for sh in ["age", "demographic", "practice"]],
        )
        out[cfg] = dict(by_alpha=by_alpha, detail=detail)
        c = detail["marginal"]
        print(f"  {cfg}: cov={c['coverage']:.3f} "
              f"[{c['coverage_lo']:.3f},{c['coverage_hi']:.3f}] "
              f"reviews/1000={c['reviews_per_1000']:.0f}")
    return out


# ============================================================ decision & harm
def run_decision():
    decision = {}
    policies_fixed = ["oracle", "annual", "biennial", "risk_stratified"]
    weight_sets = {"base": None,
                   "costly_review": dict(fp_harm=1.2),
                   "cheap_screen": dict(benefit=180.0)}

    for cfg in CONFIGS:
        risk, bud = cfg.split("-")
        df = load(cfg)
        classes = _classes(df)

        # ---- decision regret + capacity on the learning table (fast, per seed)
        dec = {}
        for pol in policies_fixed + ["central", "ensemble"]:
            reg, cap = [], []
            for s in SEEDS[:N_SEEDS_HET]:
                train, test = patient_split(df, test_size=0.3, seed=s)
                if pol in policies_fixed:
                    pred = bl.BASELINE_POLICIES[pol](test)
                elif pol == "central":
                    r = fed.centralized(train, test, classes, seed=s)
                    pred = np.array(classes)[r["proba"].argmax(1)]
                else:
                    r = fed.federated_ensemble(train, test, classes,
                                               weighting="size", k=5, seed=s)
                    pred = np.array(classes)[r["proba"].argmax(1)]
                reg.append(da.oracle_regret(test, pred)["regret_per_1000"])
                cap.append(da.capacity_outcomes(test, pred))
            dec[pol] = dict(
                regret_per_1000=summarize(reg),
                screens_per_1000=summarize([c["screens_per_1000"] for c in cap]),
                mri_per_1000=summarize([c["mri_per_1000"] for c in cap]),
                ultrasound_per_1000=summarize(
                    [c["ultrasound_per_1000"] for c in cap]),
                budget_per_1000=summarize([c["budget_per_1000"] for c in cap]))
        # managerial cost-weight sensitivity (central policy regret)
        sens = {}
        train, test = patient_split(df, test_size=0.3, seed=0)
        r = fed.centralized(train, test, classes, seed=0)
        pred = np.array(classes)[r["proba"].argmax(1)]
        for name, w in weight_sets.items():
            sens[name] = da.oracle_regret(test, pred, weights=w)["regret_per_1000"]
        dec["cost_sensitivity_central"] = sens
        decision[cfg] = dec
        print(f"  {cfg}: regret/1000 oracle=0 "
              f"central={dec['central']['regret_per_1000']['mean']:.2f} "
              f"annual={dec['annual']['regret_per_1000']['mean']:.2f}")
    return decision


def run_benefit_harm():
    harm = {}
    policies_fixed = ["oracle", "annual", "biennial", "risk_stratified"]
    for cfg in CONFIGS:
        risk, bud = cfg.split("-")
        df = load(cfg)
        # ---- cancer benefit-harm via closed-loop rollout
        bh = {}
        for pol in policies_fixed:
            rows = []
            for s in range(N_SEEDS_ROLL):
                rows.append(ro.rollout_policy(risk, bud, pol,
                                              n_patients=ROLL_N, seed=s))
            bh[pol] = _agg_bh(rows)
        # learned model rollout (train once per seed on full config)
        model_rows = []
        for s in range(N_SEEDS_ROLL):
            train, _ = patient_split(df, test_size=0.3, seed=s)
            m = LGBMClassifier(random_state=s, **LGBM_FULL)
            m.fit(train[CLF_FEATURES], train["action"])
            model_rows.append(ro.rollout_policy(risk, bud, "model",
                                                n_patients=ROLL_N_MODEL,
                                                seed=s, model=m))
        bh["fequa_model"] = _agg_bh(model_rows)
        # by race band for oracle & model
        bh["by_race_band"] = {
            "oracle": _bh_by_band(rows_last=ro.rollout_policy(risk, bud, "oracle",
                                                              n_patients=ROLL_N,
                                                              seed=0)),
            "fequa_model": _bh_by_band(rows_last=model_rows[0]),
        }
        harm[cfg] = bh
        print(f"  {cfg}: early-detect oracle={bh['oracle']['early_detection_rate']['mean']:.3f} "
              f"model={bh['fequa_model']['early_detection_rate']['mean']:.3f} "
              f"biennial={bh['biennial']['early_detection_rate']['mean']:.3f}")
    return harm


def _agg_bh(rows_list):
    keys = ["cancer_incidence", "early_detection_rate", "interval_cancer_rate",
            "detected_rate", "mean_stage_at_detection",
            "false_positives_per_patient", "benign_biopsies_per_patient"]
    cap_keys = ["total_screens", "mammography", "ultrasound", "mri", "budget_used"]
    vals = {k: [] for k in keys}
    caps = {k: [] for k in cap_keys}
    for df in rows_list:
        summ = da.benefit_harm_outcomes(df)
        for k in keys:
            vals[k].append(summ[k])
        for k in cap_keys:
            caps[k].append(float(df[k].mean()))
    out = {k: summarize(vals[k]) for k in keys}
    out.update({f"{k}_per_patient": summarize(caps[k]) for k in cap_keys})
    return out


def _bh_by_band(rows_last):
    df = rows_last.copy()
    df["race_band"] = da.race_band(df["race"].values)
    out = {}
    for g, sub in df.groupby("race_band"):
        if len(sub) >= 30:
            out[str(g)] = da.benefit_harm_outcomes(sub)
    return out


# ============================================================ incremental & shap
def run_incremental():
    cfg = "AR-HB"
    df = load(cfg)
    train, test = patient_split(df, test_size=0.5, seed=0)
    pids = train["patient_id"].unique()
    rng = np.random.default_rng(0); rng.shuffle(pids)
    clsc = []
    for n in [4, 10, 20, 50, 100, 200, 500, 1000]:
        sub = train[train["patient_id"].isin(set(pids[:n]))]
        if sub["action"].nunique() < 2:
            continue
        m = LGBMClassifier(random_state=0, **LGBM_FULL)
        m.fit(sub[CLF_FEATURES], sub["action"])
        clsc.append(dict(n=n, mf1=round(float(f1_score(
            test["action"], m.predict(test[CLF_FEATURES]), average="macro")), 4)))
    dfr = df.copy(); dfr["y_reg"] = make_regression_target(dfr)
    tr, te = patient_split(dfr, test_size=0.5, seed=0)
    pr = tr["patient_id"].unique(); rng.shuffle(pr)
    regc = []
    for fr in [0.04, 0.08, 0.15, 0.25, 0.4, 0.6, 0.8]:
        n = max(2, int(len(pr) * fr))
        sub = tr[tr["patient_id"].isin(set(pr[:n]))]
        m = LGBMRegressor(random_state=0, **LGBM_FULL)
        m.fit(sub[REG_FEATURES], sub["y_reg"])
        regc.append(dict(frac=fr, r2=round(float(r2_score(
            te["y_reg"], m.predict(te[REG_FEATURES]))), 4)))
    return dict(classification=clsc, regression=regc)


def run_shap():
    import shap
    cfg = "AR-HB"
    clf_feats = ["age", "time_since_last_screening", "time_since_last_sp",
                 "density", "firstDegreeRel"]
    label = {"age": "Age",
             "time_since_last_screening": "Time Since\nLast Screening",
             "time_since_last_sp": "Time Since\nLast SP",
             "density": "Breast Density", "firstDegreeRel": "Family History"}
    df = load(cfg)
    tr, te = patient_split(df, test_size=0.5, seed=0)
    mc = LGBMClassifier(random_state=0, **LGBM_FULL)
    mc.fit(tr[clf_feats], tr["action"])
    sv = np.array(shap.TreeExplainer(mc).shap_values(te[clf_feats]))
    imp = np.abs(sv).mean(axis=(0, 1)) if sv.ndim == 3 else np.abs(sv).mean(0)
    imp = np.ravel(imp)[:len(clf_feats)]
    cls_imp = {label[f]: float(v) for f, v in zip(clf_feats, imp)}
    dfr = df.copy(); dfr["y_reg"] = make_regression_target(dfr)
    tr, te = patient_split(dfr, test_size=0.5, seed=0)
    mr = LGBMRegressor(random_state=0, **LGBM_FULL)
    mr.fit(tr[REG_FEATURES], tr["y_reg"])
    sv = np.abs(shap.TreeExplainer(mr).shap_values(te[REG_FEATURES])).mean(0)
    order = np.argsort(sv)[::-1][:6]
    reg_imp = {REG_FEATURES[i]: float(sv[i]) for i in order}
    return dict(classification=cls_imp, regression=reg_imp)


# ============================================================ main
def _dump(name, obj):
    os.makedirs(RES_DIR, exist_ok=True)
    json.dump(obj, open(os.path.join(RES_DIR, f"{name}.json"), "w"),
              indent=2, default=float)


def main(only=None):
    t0 = time.time()
    steps = [
        ("classification", run_classification),
        ("regression", run_regression),
        ("calibration", run_calibration),
        ("leakage_ablation", run_leakage_ablation),
        ("heterogeneity", run_heterogeneity),
        ("fairness", run_fairness),
        ("conformal", run_conformal),
        ("incremental", lambda: run_incremental()),
        ("shap", run_shap),
    ]
    if only:
        steps = [s for s in steps if s[0] in only]
    for name, fn in steps:
        print(f"=== {name} ===")
        t = time.time()
        try:
            _dump(name, fn())
        except Exception as e:
            print(f"  !! {name} failed: {e}")
        print(f"    [{time.time()-t:.1f}s]")
    if (only is None) or ("decision" in only):
        print("=== decision ===")
        t = time.time()
        _dump("decision", run_decision())
        print(f"    [{time.time()-t:.1f}s]")
    if (only is None) or ("benefit_harm" in only):
        print("=== benefit_harm ===")
        t = time.time()
        _dump("benefit_harm", run_benefit_harm())
        print(f"    [{time.time()-t:.1f}s]")
    print(f"\nDone  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    main(only=only)
