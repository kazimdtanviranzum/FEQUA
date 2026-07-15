"""
rollout.py
==========
Counterfactual policy evaluation for cancer benefit-harm outcomes.

The learning tables record outcomes under the *oracle* policy. To compare the
cancer benefit-harm of different policies (oracle vs fixed-interval vs
risk-stratified vs a *learned* model) we must re-simulate the disease/detection
process under each policy on the **same** patient population. This module does
exactly that: patient primitives (latent risk, density, risk factors, cancer
onset age, preclinical sojourn) and all disease/observation random draws are
fixed per patient via a deterministic per-patient seed, so the *only* thing that
changes between policies is the sequence of actions and therefore what gets
detected. This isolates each policy's causal effect on early detection, interval
cancers, stage at detection, and screening harms.

Learned models are rolled out **closed-loop**: at each annual epoch we assemble
exactly the feature vector the model was trained on (including lagged actions and
observations produced by the rollout itself) and take the model's predicted
action. Fixed policies use their state->action rule directly.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from data_simulation import (_belief_state, _density_sensitivity, _pomdp_policy,
                             RACE_CODES, RACE_PROBS, DENSITY_CODES, DENSITY_PROBS,
                             MODALITY_COST, AGE_MIN, AGE_MAX)
from common import CLF_FEATURES


def _draw_primitives(risk, bud_level, pid, seed):
    """Deterministic per-patient primitives (identical across policies)."""
    rng = np.random.default_rng((seed + 1) * 1_000_003 + pid)
    latent = rng.lognormal(0, 0.4)
    race = int(rng.choice(RACE_CODES, p=RACE_PROBS))
    density = int(rng.choice(DENSITY_CODES, p=DENSITY_PROBS))
    menarche = int(rng.integers(10, 16))
    first_birth = int(rng.choice([0] + list(range(18, 32))))
    first_deg = 1 if rng.random() < 0.17 else 0
    had_biopsy = 1 if rng.random() < 0.23 else 0
    num_biopsy = int(rng.integers(1, 3)) if had_biopsy else 0
    hyperplasia = 1 if (had_biopsy and rng.random() < 0.10) else 0
    onset_hazard = (0.0016 if risk == "AR" else 0.0040) * latent \
        * (1.0 + 0.15 * (density - 2)) * (1.0 + 0.25 * first_deg)
    onset = None
    for a in range(AGE_MIN, AGE_MAX + 1):
        if rng.random() < onset_hazard * (1 + 0.03 * (a - 40)):
            onset = a
            break
    sojourn = int(np.clip(rng.normal(3.0, 1.2), 1, 7)) if onset else 0
    # pre-draw per-age uniforms so detection/FP are policy-independent given action
    det_u = rng.random(AGE_MAX - AGE_MIN + 2)
    fp_u = rng.random(AGE_MAX - AGE_MIN + 2)
    adh_u = rng.random(AGE_MAX - AGE_MIN + 2)
    mort_u = rng.random(AGE_MAX - AGE_MIN + 2)
    return dict(latent=latent, race=race, density=density, menarche=menarche,
                first_birth=first_birth, first_deg=first_deg,
                had_biopsy=had_biopsy, num_biopsy=num_biopsy,
                hyperplasia=hyperplasia, onset=onset, sojourn=sojourn,
                det_u=det_u, fp_u=fp_u, adh_u=adh_u, mort_u=mort_u,
                adherence=0.90)


def _feature_row(state, prim):
    """Assemble a CLF_FEATURES-ordered vector from the running rollout state."""
    ha, ho = state["hist_actions"], state["hist_obs"]
    d = {
        "observation t-5": ho[0], "action t-5": ha[0],
        "observation t-4": ho[1], "action t-4": ha[1],
        "observation t-3": ho[2], "action t-3": ha[2],
        "observation t-2": ho[3], "action t-2": ha[3],
        "observation t-1": ho[4], "action t-1": ha[4],
        "age": state["age"],
        "time_since_last_screening": state["tss"],
        "time_since_last_wp": state["twp"],
        "time_since_last_sp": state["tsp"],
        "menarcheAge": prim["menarche"], "firstLiveBirthAge": prim["first_birth"],
        "firstDegreeRel": prim["first_deg"], "hadBiopsy": prim["had_biopsy"],
        "numBiopsy": prim["num_biopsy"], "hyperPlasia": prim["hyperplasia"],
        "race": prim["race"], "density": prim["density"],
        "last_fp_age": state["last_fp_age"],
    }
    return np.array([d[f] for f in CLF_FEATURES], float)


def rollout_policy(risk, bud_level, policy, n_patients=800, seed=0, model=None):
    """Roll a policy over a patient cohort; return per-patient benefit-harm rows.

    `policy` is either a string in {'oracle','annual','biennial','risk_stratified'}
    or 'model' (then `model` must be a fitted classifier taking CLF_FEATURES).
    """
    summ = []
    for pid in range(n_patients):
        prim = _draw_primitives(risk, bud_level, pid, seed)
        state = dict(age=AGE_MIN, tss=1, twp=int(2 + pid % 8),
                     tsp=int(3 + pid % 15),
                     hist_actions=[0, 0, 0, 0, 0], hist_obs=[0, 0, 0, 0, 0],
                     last_fp_age=int(5 + pid % 11), wp_tot=0)
        detected, detect_age, detect_stage, detect_mod = 0, None, None, None
        interval_cancer = 0
        fp, biopsy = 0, 0
        n_screens = {1: 0, 2: 0, 3: 0}
        budget = 1720 if bud_level == "HB" else 1000
        step = 0
        alive = True
        while state["age"] <= AGE_MAX and alive:
            b1, b2 = _belief_state(state["age"], risk, state["tss"],
                                   prim["density"], prim["latent"])
            # choose action
            if policy == "oracle":
                action = _pomdp_policy(state["age"], state["tss"], state["twp"],
                                       b1, b2, risk, bud_level, prim["density"])
            elif policy == "annual":
                action = 1 if (state["tss"] >= 1 or state["twp"] <= 2) else 0
            elif policy == "biennial":
                action = 1 if (state["tss"] >= 2 or state["twp"] <= 2) else 0
            elif policy == "risk_stratified":
                if state["twp"] <= 2:
                    action = 1
                elif risk == "HR" and state["age"] < 50 and state["tss"] >= 1:
                    action = 3
                elif risk == "HR" and prim["density"] >= 3 and state["tss"] >= 1:
                    action = 2
                elif risk == "HR" and state["tss"] >= 1:
                    action = 1
                elif state["tss"] >= 2:
                    action = 1
                else:
                    action = 0
            elif policy == "model":
                x = _feature_row(state, prim).reshape(1, -1)
                action = int(model.predict(x)[0])
            else:
                raise ValueError(policy)

            # adherence
            performed = action
            if action != 0 and prim["adh_u"][step] > prim["adherence"]:
                performed = 0
            budget = max(0, budget - MODALITY_COST[performed])
            if performed in n_screens:
                n_screens[performed] += 1

            preclinical = (prim["onset"] is not None
                           and prim["onset"] <= state["age"] < prim["onset"] + prim["sojourn"]
                           and not detected)
            obs = 0
            if performed == 0:
                if preclinical and prim["det_u"][step] < 0.04:
                    obs = 1; state["wp_tot"] += 1; state["twp"] = 0
                elif (b1 + b2) * 3 > prim["fp_u"][step]:
                    obs = 1; state["wp_tot"] += 1; state["twp"] = 0
            else:
                if preclinical:
                    sens = _density_sensitivity(performed, prim["density"])
                    if prim["det_u"][step] < sens:
                        detected = 1
                        detect_age = state["age"]; detect_mod = performed
                        yrs = state["age"] - prim["onset"]
                        detect_stage = 0 if yrs <= 1 else (1 if yrs <= 3 else 2)
                        obs = 1; state["tsp"] = 0
                else:
                    if prim["fp_u"][step] < 0.08:
                        fp += 1; state["last_fp_age"] = state["age"]
                        if prim["det_u"][step] < 0.25:
                            biopsy += 1

            state["hist_actions"] = state["hist_actions"][1:] + [action]
            state["hist_obs"] = state["hist_obs"][1:] + [obs]
            state["tss"] = 1 if performed != 0 else state["tss"] + 1
            state["twp"] += 1
            state["tsp"] += 1
            state["age"] += 1
            step += 1
            if prim["mort_u"][step] < 0.002 * (state["age"] - 40):
                alive = False

        if prim["onset"] is not None and not detected:
            interval_cancer = 1
            detect_stage = 2
        summ.append(dict(
            patient_id=f"{risk}-{bud_level}-{pid:05d}",
            race=prim["race"], density=prim["density"],
            firstDegreeRel=prim["first_deg"],
            cancer=int(prim["onset"] is not None),
            detected=detected, interval_cancer=interval_cancer,
            detect_stage=detect_stage if detect_stage is not None else -1,
            false_positives=fp, benign_biopsies=biopsy,
            total_screens=sum(n_screens.values()),
            mammography=n_screens[1], ultrasound=n_screens[2], mri=n_screens[3],
            budget_used=(1720 if bud_level == "HB" else 1000) - budget,
        ))
    return pd.DataFrame(summ)
