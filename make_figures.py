"""
make_figures.py
===============
Renders every figure from the JSON results produced by ``experiments.py``.

Figures use a white background, black text, and a serif (Times-like) font;
data series are in colour, per the manuscript figure specification. Each figure
is guarded so a single failure does not abort the rest.
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, LBLUE, GREEN, ORANGE, RED, PURPLE, GRAY = \
    "#1B3A6B", "#4A7FC1", "#2E7D32", "#E65100", "#C62828", "#6A1B9A", "#555555"
CONFIGS = ["AR-AB", "AR-HB", "HR-AB", "HR-HB"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
})


def load(name):
    return json.load(open(os.path.join(RES, f"{name}.json")))


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ───────────────────────────────── Fig 1: architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.5); ax.axis("off")

    def box(x, y, w, h, txt, fc=BLUE, tc="white", fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec="black", lw=1.3))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold", multialignment="center")

    def arr(x1, y1, x2, y2, txt=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, txt, ha="center",
                    fontsize=7.5, style="italic", color=GRAY)

    ax.text(6, 8.2, "FEQUA: Data-Local Screening Decision Architecture",
            ha="center", fontsize=13, fontweight="bold", color=BLUE)
    for i, lbl in enumerate(["Hospital 1", "Hospital 2", "Hospital 3", "Hospital K"]):
        box(0.4 + i * 2.9, 6.9, 2.4, 0.9, f"{lbl}\n(local records)", LBLUE, fs=8.5)
    ax.text(6, 7.95, "Raw patient records never leave the institution",
            ha="center", fontsize=8.5, style="italic", color=GRAY)
    for i in range(4):
        arr(1.6 + i * 2.9, 6.9, 6, 6.1)
    box(2.6, 5.1, 6.8, 1.0,
        "Transmit ONLY per-site class probabilities (ensemble)\n"
        "or averaged logistic weights (FedAvg/FedProx)", BLUE, fs=9)
    arr(6, 5.1, 6, 4.4)
    box(3.2, 3.4, 5.6, 1.0,
        "Server aggregation -> global screening-policy surrogate", GREEN, fs=9.5)
    arr(6, 3.4, 6, 2.7)
    box(0.6, 1.5, 3.3, 1.1, "Subgroup error control\n(TPR & FPR parity)", ORANGE, fs=9)
    box(4.35, 1.5, 3.3, 1.1, "Conformal deferral\n(calibrated uncertainty)",
        PURPLE, fs=9)
    box(8.1, 1.5, 3.3, 1.1, "Interpretable\nrecommendation (SHAP)", GRAY, fs=9)
    arr(2.25, 1.5, 4.0, 0.7); arr(6, 1.5, 6, 0.7); arr(9.75, 1.5, 8.0, 0.7)
    box(4.0, 0.0, 4.0, 0.65,
        "Screening action  OR  defer to clinician", RED, fs=9.5)
    _save(fig, "fig1_architecture.png")


# ───────────────────────────────── Fig 2: federated performance
def fig_federated():
    cls = load("classification")
    methods = ["central", "ens_size", "ens_uniform", "fedavg", "fedprox", "local"]
    labels = ["Centralized\n(upper ref.)", "Ensemble\n(size-wt)",
              "Ensemble\n(uniform)", "FedAvg\n(parametric)",
              "FedProx\n(parametric)", "Local-only\n(lower ref.)"]
    colors = [BLUE, GREEN, LBLUE, ORANGE, RED, GRAY]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        means = [cls[cfg][m]["mf1"]["mean"] for m in methods]
        errs = [cls[cfg][m]["mf1"]["mean"] - cls[cfg][m]["mf1"]["lo"]
                for m in methods]
        ax.bar(range(len(methods)), means, yerr=errs, color=colors,
               edgecolor="black", lw=0.8, capsize=3)
        ax.set_title(cfg, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        ax.set_ylim(0.85, 1.005); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Macro F1 (mean, 95% CI)")
    fig.suptitle("Federated policy-surrogate performance across configurations",
                 fontweight="bold", y=1.02)
    _save(fig, "fig2_federated_performance.png")


# ───────────────────────────────── Fig 3: convergence
def fig_convergence():
    cls = load("classification")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [BLUE, GREEN, ORANGE, RED]
    for cfg, c in zip(CONFIGS, colors):
        h = cls[cfg]["convergence"]["fedprox"]
        ax.plot(range(1, len(h) + 1), h, "-o", color=c, ms=3, label=f"{cfg}")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Macro F1 (FedProx)")
    ax.set_title("Genuine FedProx convergence (parametric client)",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, "fig3_convergence.png")


# ───────────────────────────────── Fig 4: heterogeneity
def fig_heterogeneity():
    het = load("heterogeneity")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    # non-IID: performance vs divergence
    alphas = sorted(het["noniid_sweep"].keys(), key=float)
    div = [het["noniid_sweep"][a]["divergence"]["mean"] for a in alphas]
    order = np.argsort(div)
    div = np.array(div)[order]
    for m, c, lbl in [("ensemble", GREEN, "Federated ensemble"),
                      ("fedavg", ORANGE, "FedAvg (parametric)"),
                      ("local", GRAY, "Local-only")]:
        y = np.array([het["noniid_sweep"][a][m]["mean"] for a in alphas])[order]
        e = np.array([het["noniid_sweep"][a][m]["mean"] -
                      het["noniid_sweep"][a][m]["lo"] for a in alphas])[order]
        ax1.errorbar(div, y, yerr=e, marker="o", color=c, label=lbl, capsize=3)
    ax1.set_xlabel("Site label divergence (JS, higher = more non-IID)")
    ax1.set_ylabel("Macro F1"); ax1.set_title("Robustness to institutional heterogeneity",
                                              fontweight="bold")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    # K sweep
    ks = sorted(het["k_sweep"].keys(), key=int)
    for m, c, lbl in [("ensemble", GREEN, "Federated ensemble"),
                      ("fedavg", ORANGE, "FedAvg (parametric)"),
                      ("local", GRAY, "Local-only")]:
        y = [het["k_sweep"][k][m]["mean"] for k in ks]
        ax2.plot([int(k) for k in ks], y, "-o", color=c, label=lbl)
    ax2.set_xlabel("Number of participating sites (K)")
    ax2.set_ylabel("Macro F1"); ax2.set_title("Scaling with site count",
                                              fontweight="bold")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
    _save(fig, "fig4_heterogeneity.png")


# ───────────────────────────────── Fig 5: fairness
def fig_fairness():
    fair = load("fairness")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, cfg in zip(axes, ["AR-HB", "HR-HB"]):
        ga = fair["sweeps"][cfg]["induced_bias|downgrade|group_aware"]
        gb = fair["sweeps"][cfg]["induced_bias|downgrade|group_blind"]
        lam = [r["lam"] for r in ga]
        ax.plot(lam, [r["tpr_gap"] for r in ga], "-o", color=RED,
                label="TPR gap (group-aware)")
        ax.plot(lam, [r["fpr_gap"] for r in ga], "-s", color=ORANGE,
                label="FPR gap (group-aware)")
        ax.plot(lam, [r["tpr_gap"] for r in gb], "--^", color=GRAY,
                label="TPR gap (group-blind)")
        ax2 = ax.twinx()
        ax2.plot(lam, [r["mf1"] for r in ga], "-D", color=GREEN,
                 label="Macro F1 (group-aware)", ms=4)
        ax2.set_ylabel("Macro F1", color=GREEN)
        ax2.tick_params(axis="y", colors=GREEN); ax2.set_ylim(0.7, 1.02)
        ax.set_xlabel("Mitigation strength  λ"); ax.set_ylabel("Between-group gap")
        ax.set_title(f"{cfg}: fairness–utility (induced bias)", fontweight="bold")
        ax.grid(alpha=0.3)
        if cfg == "AR-HB":
            ax.legend(loc="upper right", fontsize=8)
    _save(fig, "fig5_fairness.png")


# ───────────────────────────────── Fig 6: conformal
def fig_conformal():
    conf = load("conformal")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = [BLUE, GREEN, ORANGE, RED]
    for cfg, c in zip(CONFIGS, colors):
        ba = conf[cfg]["by_alpha"]
        al = [r["alpha"] for r in ba]
        cov = [r["coverage"]["mean"] for r in ba]
        lo = [r["coverage"]["mean"] - r["coverage"]["lo"] for r in ba]
        ax1.errorbar(al, cov, yerr=lo, marker="o", color=c, label=cfg, capsize=3)
    ax1.plot([0.01, 0.2], [0.99, 0.80], "k--", alpha=0.6, label="target (1−α)")
    ax1.set_xlabel("α"); ax1.set_ylabel("Empirical coverage (95% CI)")
    ax1.set_title("Marginal coverage vs target", fontweight="bold")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    # class-conditional coverage at alpha 0.05 for HR-HB
    cc = conf["HR-HB"]["detail"]["class_conditional"]
    action_names = {0: "Wait", 1: "Mammo", 2: "US", 3: "MRI"}
    ks = sorted(cc.keys(), key=int)
    covs = [cc[k]["coverage"] for k in ks]
    errs = [cc[k]["coverage"] - cc[k]["lo"] for k in ks]
    ax2.bar(range(len(ks)), covs, yerr=errs, color=[LBLUE, GREEN, ORANGE, PURPLE][:len(ks)],
            edgecolor="black", capsize=3)
    ax2.axhline(0.95, color="k", ls="--", alpha=0.6, label="target 0.95")
    ax2.set_xticks(range(len(ks)))
    ax2.set_xticklabels([action_names.get(int(k), k) for k in ks])
    ax2.set_ylim(0.8, 1.01); ax2.set_ylabel("Coverage")
    ax2.set_title("HR-HB class-conditional coverage\n(rare actions hidden by marginal)",
                  fontweight="bold")
    ax2.grid(axis="y", alpha=0.3); ax2.legend(fontsize=8)
    _save(fig, "fig6_conformal.png")


# ───────────────────────────────── Fig 7: decision regret & capacity
def fig_decision():
    dec = load("decision")
    policies = ["oracle", "annual", "biennial", "risk_stratified", "central", "ensemble"]
    labels = ["Oracle", "Annual", "Biennial", "Risk-strat.", "FEQUA\ncentral", "FEQUA\nensemble"]
    colors = [BLUE, ORANGE, RED, PURPLE, GREEN, LBLUE]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # regret (abs) for HR-HB
    cfg = "HR-HB"
    reg = [abs(dec[cfg][p]["regret_per_1000"]["mean"]) for p in policies]
    axes[0].bar(range(len(policies)), reg, color=colors, edgecolor="black")
    axes[0].set_xticks(range(len(policies)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("|Decision regret| per 1,000 (vs oracle value)")
    axes[0].set_title(f"{cfg}: value loss by policy", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    # capacity: screens/1000 by policy
    scr = [dec[cfg][p]["screens_per_1000"]["mean"] for p in policies]
    mri = [dec[cfg][p]["mri_per_1000"]["mean"] for p in policies]
    us = [dec[cfg][p]["ultrasound_per_1000"]["mean"] for p in policies]
    x = np.arange(len(policies))
    axes[1].bar(x, scr, color=LBLUE, edgecolor="black", label="Any screen")
    axes[1].bar(x, mri, color=RED, edgecolor="black", label="MRI")
    axes[1].bar(x, us, bottom=mri, color=ORANGE, edgecolor="black", label="Ultrasound")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Actions per 1,000 decisions")
    axes[1].set_title(f"{cfg}: capacity & modality demand", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3); axes[1].legend(fontsize=8)
    _save(fig, "fig7_decision_capacity.png")


# ───────────────────────────────── Fig 8: benefit-harm frontier
def fig_benefit_harm():
    bh = load("benefit_harm")
    policies = ["oracle", "annual", "biennial", "risk_stratified", "fequa_model"]
    labels = {"oracle": "Oracle", "annual": "Annual", "biennial": "Biennial",
              "risk_stratified": "Risk-strat.", "fequa_model": "FEQUA"}
    markers = {"oracle": "o", "annual": "s", "biennial": "^",
               "risk_stratified": "D", "fequa_model": "*"}
    colors = {"oracle": BLUE, "annual": ORANGE, "biennial": RED,
              "risk_stratified": PURPLE, "fequa_model": GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, cfg in zip(axes, ["HR-HB", "AR-HB"]):
        for p in policies:
            x = bh[cfg][p]["false_positives_per_patient"]["mean"]
            y = bh[cfg][p]["early_detection_rate"]["mean"]
            s = bh[cfg][p]["total_screens_per_patient"]["mean"]
            ax.scatter(x, y, s=90 + s * 8, marker=markers[p], color=colors[p],
                       edgecolor="black", zorder=3, label=labels[p])
            ax.annotate(labels[p], (x, y), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("False positives per patient (harm)")
        ax.set_ylabel("Early-detection rate (benefit)")
        ax.set_title(f"{cfg}: benefit–harm frontier\n(marker size ∝ screening burden)",
                     fontweight="bold")
        ax.grid(alpha=0.3)
    _save(fig, "fig8_benefit_harm.png")


# ───────────────────────────────── Fig 9: SHAP & incremental
def fig_shap_incremental():
    sh = load("shap"); inc = load("incremental")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    cls = sh["classification"]
    names = list(cls.keys()); vals = list(cls.values())
    order = np.argsort(vals)
    ax1.barh([names[i] for i in order], [vals[i] for i in order],
             color=BLUE, edgecolor="black")
    ax1.set_xlabel("Mean |SHAP| value")
    ax1.set_title("Global feature importance (classification)", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    c = inc["classification"]
    ax2.plot([d["n"] for d in c], [d["mf1"] for d in c], "-o", color=GREEN)
    ax2.set_xscale("log"); ax2.set_xlabel("Training patients (log scale)")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Incremental learning curve", fontweight="bold")
    ax2.grid(alpha=0.3)
    _save(fig, "fig9_shap_incremental.png")


# ───────────────────────────────── Fig 10: coverage-under-shift
def fig_shift():
    conf = load("conformal")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    shifts = ["age", "demographic", "practice"]
    x = np.arange(len(shifts)); w = 0.2
    colors = [BLUE, GREEN, ORANGE, RED]
    for i, cfg in enumerate(CONFIGS):
        det = {s["shift"]: s for s in conf[cfg]["detail"]["shift"]}
        base = [det[s]["base_coverage"] for s in shifts]
        shifted = [det[s]["shift_coverage"] for s in shifts]
        ax.bar(x + i * w - 1.5 * w, shifted, w, color=colors[i],
               edgecolor="black", label=cfg)
    ax.axhline(0.95, color="k", ls="--", alpha=0.6, label="target 0.95")
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in shifts])
    ax.set_ylabel("Coverage under shift"); ax.set_ylim(0.8, 1.02)
    ax.set_title("Conformal coverage under distribution shift", fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig10_coverage_shift.png")


def main():
    for fn in [fig_architecture, fig_federated, fig_convergence, fig_heterogeneity,
               fig_fairness, fig_conformal, fig_decision, fig_benefit_harm,
               fig_shap_incremental, fig_shift]:
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} failed: {e}")
    print("Figures written to figures/")


if __name__ == "__main__":
    main()
