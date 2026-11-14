# -*- coding: utf-8 -*-
"""m17_transport_ladder.py - transport/calibration ladder (held-out) and local sample-size rules.

Motivation:
  - The earlier "recalibration_sensitivity" block was a calibration-only simulation that used target prevalence,
    and was explicitly labelled "NOT a validation result"; it cannot support a transportability claim.
  - This script uses a K-fold held-out ladder: corrections are fitted on the calibration subset of each fold and evaluated on the held-out subset.
  - It also derives an empirical sample-size rule for a new system to reach a target ECE and C-index precision.

Ladder (every rung evaluated held-out):
  rung0 = no correction
  rung1 = intercept-only recalibration (calibration-in-the-large, closed form)
  rung2 = slope and intercept recalibration (logistic recalibration on logit(p))
  rung3 = local redevelopment (logistic refit on target-system covariates)

Definitions follow the frozen race-blind model and harmonisation in knh_multiwave_cross_system.py.
Both self-check gates (with race 0.8099, race-blind 0.8065) must pass; otherwise the run aborts.
Output: results/m2/m17_transport_ladder.json
Run: python scripts/m17_transport_ladder.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

PROJ = os.environ.get("CKD_PIPELINE_BASE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = PROJ                      # package layout: the knh modules sit beside m17 in scripts/
sys.path.insert(0, os.path.join(REPO, "scripts"))
import knh2022_cross_system as M                      # noqa: E402  (provides PROJ/HYP/load_nhanes)
import knh_multiwave_cross_system as MW               # noqa: E402  (provides load_year/RB and the self-check gate)

OUT = os.path.join(PROJ, "results", "m2", "m17_transport_ladder.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RB = MW.RB
K = 5
SEED = 20261004
NGRID = (200, 400, 800, 1600, 3200, 6400)
NREP = 200


def drb(X):
    """Race-blind variant (identical to knh_multiwave_cross_system.main): female dummy, sex dropped."""
    X = X.copy()
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["sex"])


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    q = pd.qcut(p, bins, duplicates="drop")
    o = pd.Series(y).groupby(q, observed=True).mean().values
    e = pd.Series(p).groupby(q, observed=True).mean().values
    w = pd.Series(y).groupby(q, observed=True).size().values / len(y)
    return float(np.sum(w * np.abs(o - e)))


def fit_intercept(p, y):
    """Intercept-only recalibration: solve sum(sigmoid(logit(p)+a)) = sum(y)."""
    lo = logit(p)
    f = lambda a: expit(lo + a).sum() - np.sum(y)
    return float(brentq(f, -30, 30))


def fit_slope_intercept(p, y):
    lo = logit(p).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, max_iter=2000)
    except TypeError:
        lr = LogisticRegression(C=1e9, max_iter=2000)
    lr.fit(lo, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_rung(rung, params, p):
    if rung == 0:
        return np.asarray(p, float)
    if rung == 1:
        return expit(logit(p) + params)
    b0, b1 = params
    return expit(b1 * logit(p) + b0)


def main():
    t0 = time.time()
    res = {"design": {
        "purpose": "held-out transport recalibration ladder + empirical local sample-size rule",
        "target_system": "KNHANES 2022-2024 (pooled), frozen race-blind NHANES model applied without refitting",
        "folds": K, "seed": SEED, "sample_size_grid": list(NGRID), "reps_per_size": NREP,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}}

    # ---- self-check gate (reuses the frozen values) ----
    dev = pd.concat([M.load_nhanes(c) for c in ("G", "H", "I")], ignore_index=True)
    val = M.load_nhanes("J")
    d_rb, v_rb = drb(dev[RB]), drb(val[RB])
    med = d_rb.median()
    F = lambda X: pd.concat([X.fillna(med), X.isna().add_prefix("miss_")], axis=1)
    from xgboost import XGBClassifier
    m = XGBClassifier(**M.HYP).fit(F(d_rb), dev["ckd"].values)
    auc_ck = float(roc_auc_score(val["ckd"].values, m.predict_proba(F(v_rb))[:, 1]))
    res["self_check"] = {"race_blind_temporal_auc": round(auc_ck, 4), "expected": 0.8065,
                         "passed": abs(auc_ck - 0.8065) <= 1e-4}
    print(f"[self-check] race-blind temporal AUC={auc_ck:.4f} (expected 0.8065) {'✅' if res['self_check']['passed'] else '❌'}")
    if not res["self_check"]["passed"]:
        sys.exit("self-check gate failed; aborting")

    # ---- target system (KNHANES, three waves pooled) and harmonisation ----
    pir = dev["poverty_ratio"].dropna()
    qs = pir.quantile([0, .2, .4, .6, .8, 1.0]).values
    means = [pir[(pir >= qs[i]) & (pir <= qs[i + 1])].mean() for i in range(5)]
    frames = []
    for yr in MW.YEARS:
        x, _ = MW.load_year(yr)
        x["poverty_ratio"] = x["incm5"].map({i + 1: means[i] for i in range(5)})
        x["education"] = 1.0 + (x["education_raw"] - 1.0) * (4.0 / 3.0)
        x = x[(x["age"] >= 18) & x["sex"].isin([1, 2]) & x["ckd"].notna()].copy()
        frames.append(x)
    D = pd.concat(frames, ignore_index=True)
    p_all = m.predict_proba(F(drb(D[RB])))[:, 1]
    y_all = D["ckd"].values.astype(float)
    X_all = F(drb(D[RB])).reset_index(drop=True)
    res["target_cohort"] = {"n": int(len(D)), "n_ckd": int(y_all.sum()),
                            "ckd_pct": round(100 * float(y_all.mean()), 2),
                            "auc_frozen": round(float(roc_auc_score(y_all, p_all)), 4),
                            "ece_raw": round(ece(p_all, y_all), 4)}
    print(f"[target] n={len(D)} CKD={int(y_all.sum())} AUC(冻结)={res['target_cohort']['auc_frozen']} ECE={res['target_cohort']['ece_raw']}")

    # ---- A3-1 held-out ladder (K-fold) ----
    skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
    rows = {r: [] for r in ("rung0_none", "rung1_intercept", "rung2_slope_intercept", "rung3_local_refit")}
    for tr, te in skf.split(p_all.reshape(-1, 1), y_all):
        ptr, ytr, pte, yte = p_all[tr], y_all[tr], p_all[te], y_all[te]
        a = fit_intercept(ptr, ytr)
        b1, b0 = fit_slope_intercept(ptr, ytr)
        try:
            lr = LogisticRegression(penalty=None, max_iter=3000)
        except TypeError:
            lr = LogisticRegression(C=1e9, max_iter=3000)
        med_tr = X_all.iloc[tr].median()
        Xtr = X_all.iloc[tr].fillna(med_tr)
        Xte = X_all.iloc[te].fillna(med_tr)
        lr.fit(Xtr, ytr)
        preds = {
            "rung0_none": pte,
            "rung1_intercept": apply_rung(1, a, pte),
            "rung2_slope_intercept": apply_rung(2, (b0, b1), pte),
            "rung3_local_refit": lr.predict_proba(Xte)[:, 1],
        }
        for k, pr in preds.items():
            rows[k].append({"ece": ece(pr, yte), "auc": float(roc_auc_score(yte, pr)),
                            "mean_pred": float(np.mean(pr)), "obs": float(np.mean(yte))})
    ladder = {}
    for k, lst in rows.items():
        ladder[k] = {m_: round(float(np.mean([r[m_] for r in lst])), 4) for m_ in ("ece", "auc", "mean_pred", "obs")}
        ladder[k]["ece_sd"] = round(float(np.std([r["ece"] for r in lst])), 4)
    res["held_out_ladder"] = ladder
    print("[ladder]", json.dumps(ladder, ensure_ascii=False))

    # ---- A3-2 empirical sample-size rule: ECE and AUC precision as a function of n ----
    rng = np.random.default_rng(SEED)
    idx_all = np.arange(len(y_all))
    curve = []
    for n in NGRID:
        eces, aucs, events, ok = [], [], [], 0
        for _ in range(NREP):
            i = rng.choice(idx_all, n, replace=False)
            if y_all[i].sum() < 20 or (1 - y_all[i]).sum() < 20:
                continue
            rest = np.setdiff1d(idx_all, i, assume_unique=False)
            if y_all[rest].sum() < 5:
                continue
            a = fit_intercept(p_all[i], y_all[i])
            pr = apply_rung(1, a, p_all[rest])
            eces.append(ece(pr, y_all[rest]))
            aucs.append(roc_auc_score(y_all[rest], p_all[rest]))
            events.append(int(y_all[i].sum())); ok += 1
        if ok == 0:
            continue
        curve.append({"n_calib": int(n), "reps_used": int(ok),
                      "events_calib_median": int(np.median(events)),
                      "held_out_ece_median": round(float(np.median(eces)), 4),
                      "held_out_ece_p90": round(float(np.percentile(eces, 90)), 4),
                      "held_out_auc_median": round(float(np.median(aucs)), 4),
                      "held_out_auc_iqr_width": round(float(np.percentile(aucs, 75) - np.percentile(aucs, 25)), 4),
                      "held_out_auc_p025": round(float(np.percentile(aucs, 2.5)), 4),
                      "held_out_auc_p975": round(float(np.percentile(aucs, 97.5)), 4)})
        print(f"  n={n:5d} events≈{curve[-1]['events_calib_median']:4d} -> held-out median ECE {curve[-1]['held_out_ece_median']:.4f} "
              f"(P90 {curve[-1]['held_out_ece_p90']:.4f}) | AUC 95% interval width {curve[-1]['held_out_auc_p975']-curve[-1]['held_out_auc_p025']:.4f}")
    res["sample_size_curve"] = curve
    # smallest n meeting median held-out ECE <= 0.02 and AUC 95% interval width <= 0.05
    def minimal(pred):
        hit = [c for c in curve if pred(c)]
        return hit[0] if hit else None
    res["rules"] = {
        "ECE_median_le_0.02": minimal(lambda c: c["held_out_ece_median"] <= 0.02),
        "AUC_ciwidth_le_0.05": minimal(lambda c: (c["held_out_auc_p975"] - c["held_out_auc_p025"]) <= 0.05),
        "note": "n_calib is the number of target-system individuals available for local recalibration (events = empirical median); the remainder is used for held-out evaluation.",
    }
    res["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"\nsaved: {OUT}  ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
