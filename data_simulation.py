"""
data_simulation.py
==================
Synthetic breast-cancer-screening generator for the FEQUA study.

Each row is a (patient-state, recommended-action) pair sampled along a simulated
patient lifetime (ages 40-80). A ``patient_id`` column groups every annual
decision epoch belonging to the same simulated patient so that all downstream
train/calibration/test splits are performed at the *patient* level (no within-
patient leakage). Four risk-budget configurations are produced: AR-AB, AR-HB,
HR-AB, HR-HB.

This generator was rebuilt during revision to address reviewer concerns about
population realism and outcome relevance:

* **Marginals calibrated to BCSC references** (Breast Cancer Surveillance
  Consortium risk-factor tables). First-degree family history ~17 %, prior
  biopsy ~23 %, race/ethnicity ~65 % White over the standard categories, and a
  BI-RADS breast-density variable (a-d) with the widely reported ~10/40/40/10
  split. These replace the near-uniform priors used in the first draft.
* **Breast-density-dependent belief.** Density enters the disease-belief state
  and the (still deterministic) POMDP surrogate policy, so the policy legitimately
  depends on a clinically meaningful covariate rather than only on age/time.
* **Cancer natural-history process.** A latent healthy -> preclinical in-situ ->
  preclinical invasive -> clinical progression is simulated per patient. Screening
  detects preclinical disease with modality- and density-dependent sensitivity;
  undetected preclinical cancer surfaces clinically as an *interval cancer* at a
  worse stage. This yields patient-level cancer *benefit* outcomes (early vs late
  detection, interval cancers, stage at detection) and screening *harm* outcomes
  (false-positive recalls, benign biopsies) for the decision/benefit-harm analysis.
* **Adherence.** A site- and patient-varying probability that a recommended
  screen is actually completed, so downstream analyses can study operational
  imperfection.

The screening *policy* itself remains a deterministic function of the state
(no policy randomness); this is what allows an interpretable model to recover it,
exactly the premise of the base study. All stochasticity comes from the disease /
observation / adherence process.

Author: FEQUA project
License: MIT
"""

from __future__ import annotations
import numpy as np
import pandas as pd

ACTIONS = {0: "Wait", 1: "Mammography", 2: "Ultrasound", 3: "MRI"}
AGE_MIN, AGE_MAX = 40, 80

# ---- BCSC-calibrated categorical marginals -------------------------------
# Race/ethnicity ordering (12 codes used elsewhere in the pipeline); the first
# entry (code 1) is White, weighted to ~65 % to match BCSC screening-population
# composition, remaining probability spread across 11 minority codes.
RACE_CODES = list(range(1, 13))
_RACE_MINOR = np.array([0.11, 0.06, 0.05, 0.035, 0.03, 0.02,
                        0.015, 0.01, 0.01, 0.005, 0.005])   # sums 0.30
RACE_PROBS = np.concatenate([[0.70], _RACE_MINOR])
RACE_PROBS = RACE_PROBS / RACE_PROBS.sum()

# BI-RADS density a/b/c/d -> codes 1..4 ; ~10/40/40/10 (BCSC).
DENSITY_CODES = [1, 2, 3, 4]
DENSITY_PROBS = np.array([0.10, 0.40, 0.40, 0.10])

# Modality detection sensitivity for *preclinical* disease, before density
# modulation. Mammography loses sensitivity in dense breasts; MRI is least
# density-sensitive. (Illustrative, ordered as in the screening literature.)
BASE_SENS = {1: 0.80, 2: 0.86, 3: 0.90}   # mammography, ultrasound, MRI
MODALITY_COST = {0: 0, 1: 100, 2: 150, 3: 500}


def _density_sensitivity(modality, density):
    """Detection sensitivity of `modality` given BI-RADS `density` (1..4)."""
    if modality == 0:
        return 0.0
    base = BASE_SENS[modality]
    # dense breasts (c/d) reduce mammography most, ultrasound moderately, MRI little
    penalty = {1: 0.12, 2: 0.05, 3: 0.02}[modality] * max(0, density - 2)
    return float(np.clip(base - penalty, 0.30, 0.98))


def _belief_state(age, risk_level, t_since_screen, density, latent=1.0):
    """Approximate POMDP belief over (in-situ, invasive), density-aware."""
    base = 0.0008 + 0.00004 * (age - 40)
    risk_mult = (1.0 if risk_level == "AR" else 2.4) * latent
    dens_mult = 1.0 + 0.12 * (density - 2)          # dense breasts -> higher belief
    b1 = base * risk_mult * dens_mult * (1 + 0.15 * t_since_screen)   # in-situ
    b2 = 0.5 * b1 * (1 + 0.10 * t_since_screen)                        # invasive
    return min(b1, 0.012), min(b2, 0.008)


def _pomdp_policy(age, t_since_screen, t_since_wp, b1, b2,
                  risk, bud_level, density):
    """Deterministic surrogate for a constrained-POMDP screening policy.

    Returns an integer action 0-3 (Wait / Mammography / Ultrasound / MRI). The
    policy is a fixed function of the state; density now participates (dense-breast
    high-risk patients escalate to ultrasound / MRI earlier).
    """
    # Wait-positive forces a confirmatory screen
    if t_since_wp <= 2:
        if bud_level == "HB" and age < 55 and risk == "AR":
            return 3
        return 1

    # Recent screen -> wait (dominant behaviour)
    if t_since_screen < 2:
        return 0

    # Younger high-budget AR patients: aggressive early MRI
    if bud_level == "HB" and age < 50 and risk == "AR" and t_since_screen >= 2:
        return 3

    # Dense-breast high-risk high-budget -> supplemental ultrasound
    if risk == "HR" and bud_level == "HB" and density >= 3 and t_since_screen >= 2:
        return 2

    # Mammography when elapsed time crosses an age/budget-dependent threshold
    thr = 3 if bud_level == "AB" else 2
    if risk == "HR":
        thr = max(2, thr - 1)
    if t_since_screen >= thr:
        return 1
    return 0


def simulate_config(risk, bud_level, n_patients=1200, seed=0, site_adherence=None):
    """Simulate `n_patients` lifetimes for one configuration.

    Returns a tuple (rows_df, patients_df). ``rows_df`` is the decision-epoch
    table used for learning; ``patients_df`` is one row per patient carrying the
    realised cancer-outcome / harm summary used by the benefit-harm analysis.
    """
    rng = np.random.default_rng(seed)
    rows, patient_summ = [], []

    for pid in range(n_patients):
        latent = rng.lognormal(0, 0.4)
        race = int(rng.choice(RACE_CODES, p=RACE_PROBS))
        density = int(rng.choice(DENSITY_CODES, p=DENSITY_PROBS))

        # --- BCSC-calibrated risk factors ---
        menarche = int(rng.integers(10, 16))
        first_birth = int(rng.choice([0] + list(range(18, 32))))
        first_deg = 1 if rng.random() < 0.17 else 0            # ~17 % (BCSC)
        had_biopsy = 1 if rng.random() < 0.23 else 0           # ~23 % (BCSC)
        num_biopsy = int(rng.integers(1, 3)) if had_biopsy else 0
        hyperplasia = 1 if (had_biopsy and rng.random() < 0.10) else 0
        adherence = 0.90 if site_adherence is None else site_adherence
        adherence = float(np.clip(adherence + rng.normal(0, 0.03), 0.6, 0.99))

        # --- latent cancer onset (preclinical) ---
        onset_hazard = (0.0016 if risk == "AR" else 0.0040) * latent \
            * (1.0 + 0.15 * (density - 2)) * (1.0 + 0.25 * first_deg)
        cancer_onset_age = None
        for a in range(AGE_MIN, AGE_MAX + 1):
            if rng.random() < onset_hazard * (1 + 0.03 * (a - 40)):
                cancer_onset_age = a
                break
        # preclinical sojourn: time from onset until it would surface clinically
        sojourn = int(np.clip(rng.normal(3.0, 1.2), 1, 7)) if cancer_onset_age else 0
        detected, detect_age, detect_stage, detect_mod = 0, None, None, None
        interval_cancer = 0
        false_positives, benign_biopsies = 0, 0

        budget = 1720 if bud_level == "HB" else 1000
        t_since_screen = 1
        t_since_wp = int(rng.integers(1, 10))
        t_since_sp = int(rng.integers(1, 20))
        wp_tot = 0
        last_fp_age = int(rng.integers(5, 16))
        hist_actions = [0, 0, 0, 0, 0]
        hist_obs = [0, 0, 0, 0, 0]

        age = AGE_MIN
        alive = True
        while age <= AGE_MAX and alive:
            b1, b2 = _belief_state(age, risk, t_since_screen, density, latent)
            action = _pomdp_policy(age, t_since_screen, t_since_wp,
                                   b1, b2, risk, bud_level, density)

            # adherence: a recommended screen may not be completed
            performed = action
            if action != 0 and rng.random() > adherence:
                performed = 0

            budget = max(0, budget - MODALITY_COST[performed])

            # --- cancer present preclinically this year? ---
            preclinical = (cancer_onset_age is not None
                           and cancer_onset_age <= age < cancer_onset_age + sojourn
                           and not detected)

            obs = 0
            if performed == 0:                          # wait -> self-detection chance
                if preclinical and rng.random() < 0.04:
                    obs = 1; wp_tot += 1; t_since_wp = 0
                elif (b1 + b2) * 3 > rng.random():
                    obs = 1; wp_tot += 1; t_since_wp = 0
            else:                                       # screening performed
                if preclinical:
                    sens = _density_sensitivity(performed, density)
                    if rng.random() < sens:
                        detected = 1
                        detect_age = age
                        detect_mod = performed
                        # earlier detection -> earlier stage
                        yrs_in = age - cancer_onset_age
                        detect_stage = 0 if yrs_in <= 1 else (1 if yrs_in <= 3 else 2)
                        obs = 1; t_since_sp = 0
                else:
                    # false-positive recall + possible benign biopsy (harm)
                    if rng.random() < 0.08:
                        false_positives += 1
                        last_fp_age = age
                        if rng.random() < 0.25:
                            benign_biopsies += 1

            rows.append({
                "patient_id": f"{risk}-{bud_level}-{pid:05d}",
                "observation t-5": hist_obs[0], "action t-5": hist_actions[0],
                "observation t-4": hist_obs[1], "action t-4": hist_actions[1],
                "observation t-3": hist_obs[2], "action t-3": hist_actions[2],
                "observation t-2": hist_obs[3], "action t-2": hist_actions[3],
                "observation t-1": hist_obs[4], "action t-1": hist_actions[4],
                "age": age,
                "time_since_last_screening": t_since_screen,
                "time_since_last_wp": t_since_wp,
                "time_since_last_sp": t_since_sp,
                "menarcheAge": menarche, "firstLiveBirthAge": first_birth,
                "firstDegreeRel": first_deg, "hadBiopsy": had_biopsy,
                "numBiopsy": num_biopsy, "hyperPlasia": hyperplasia,
                "race": race, "density": density, "last_fp_age": last_fp_age,
                "wp_tot": wp_tot, "b1": b1, "b2": b2,
                "remaining_budget": budget,
                "risk": risk, "budget_level": bud_level,
                "action": action,               # the POMDP-recommended action (label)
                "performed_action": performed,  # what actually happened (adherence)
            })

            hist_actions = hist_actions[1:] + [action]
            hist_obs = hist_obs[1:] + [obs]
            t_since_screen = 1 if performed != 0 else t_since_screen + 1
            t_since_wp += 1
            t_since_sp += 1
            age += 1
            if rng.random() < 0.002 * (age - 40):
                alive = False

        # if cancer surfaced clinically without detection -> interval cancer (late)
        if cancer_onset_age is not None and not detected:
            interval_cancer = 1
            detect_age = min(cancer_onset_age + sojourn, AGE_MAX)
            detect_stage = 2                    # clinically surfaced -> advanced
            detect_mod = 0

        patient_summ.append({
            "patient_id": f"{risk}-{bud_level}-{pid:05d}",
            "risk": risk, "budget_level": bud_level,
            "race": race, "density": density, "firstDegreeRel": first_deg,
            "cancer": int(cancer_onset_age is not None),
            "cancer_onset_age": cancer_onset_age if cancer_onset_age else -1,
            "detected": detected,
            "detect_age": detect_age if detect_age else -1,
            "detect_stage": detect_stage if detect_stage is not None else -1,
            "detect_modality": detect_mod if detect_mod is not None else -1,
            "interval_cancer": interval_cancer,
            "false_positives": false_positives,
            "benign_biopsies": benign_biopsies,
        })

    return pd.DataFrame(rows), pd.DataFrame(patient_summ)


def build_all(n_patients=1200, out_dir="data"):
    """Build all four configurations, write CSVs, return dict of row DataFrames."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    cfgs = {}
    for i, (risk, bud) in enumerate([("AR", "AB"), ("AR", "HB"),
                                     ("HR", "AB"), ("HR", "HB")]):
        key = f"{risk}-{bud}"
        rows, pat = simulate_config(risk, bud, n_patients=n_patients, seed=i + 1)
        rows.to_csv(f"{out_dir}/{key}.csv", index=False)
        pat.to_csv(f"{out_dir}/{key}_patients.csv", index=False)
        cfgs[key] = rows
    return cfgs


if __name__ == "__main__":
    import os
    _out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")
    data = build_all(n_patients=1200, out_dir=_out)
    for k, df in data.items():
        dist = df["action"].value_counts().sort_index().to_dict()
        print(f"{k}: {len(df):,} epochs | action dist: {dist}")
