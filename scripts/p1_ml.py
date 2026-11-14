# -*- coding: utf-8 -*-
"""Stage A, wave 2: machine-learning analyses.
P1-13 in-repository attribution using the exported XGBoost model (pred_contribs = SHAP)
P1-14 feature ladder (1, 3, 5, 8, 12, and 12 plus missingness flags) with delta-AUC confidence intervals
P1-15 extreme-missingness perturbation (all missing, 50% missing at random, single-feature ablation)
Self-check: the exported model must reach AUC 0.8099 (+/-1e-4) on the test set; otherwise abort.
Output: results/p1_ml.json (optionally figures/fig_shap.pdf).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, DMatrix

t0 = time.time(); RNG = 2026
RES = r"results"
REPO = r"."
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
cohort = pd.read_parquet(os.path.join(RES, "cohort.parquet"))
train = cohort[cohort["year"].isin([2011, 2013, 2015])].copy()
test = cohort[cohort["year"] == 2017].copy().reset_index(drop=True)

def enc(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float); X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float); X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

med = enc(train[FEATURES]).median()
f = lambda D: pd.concat([D.fillna(med), D.isna().add_prefix("miss_")], axis=1)
Xtr, Xte = f(enc(train[FEATURES])), f(enc(test[FEATURES]))
ytr, yte = train["ckd"].values, test["ckd"].values
out = {}

# ---------- P1-13 attribution with the exported model (SHAP) ----------
print("P1-13 attribution (exported model, pred_contribs) ...", flush=True)
mp = os.path.join(REPO, "models", "xgb_model.json")
pp = os.path.join(REPO, "models", "preprocessing.json")
if not (os.path.exists(mp) and os.path.exists(pp)):
    out["shap"] = {"note": "exported model or preprocessing file missing; cannot attribute with the published artifacts"}
else:
    ppj = json.load(open(pp, encoding="utf-8"))
    cols = ppj.get("columns") or ppj.get("feature_order") or ppj.get("final_columns")
    booster = XGBClassifier(); booster.load_model(mp)
    Xuse = Xte[cols] if cols else Xte
    p_export = booster.predict_proba(Xuse.astype(float))[:, 1]
    a_exp = roc_auc_score(yte, p_export)
    print(f"   exported model AUC={a_exp:.4f}(target 0.8099)", flush=True)
    if abs(a_exp - 0.8099) > 1e-4:
        out["shap"] = {"note": "exported-model self-check failed; attribution not reported", "auc_export": round(a_exp, 4)}
    else:
        dm = DMatrix(Xuse, feature_names=list(Xuse.columns))
        contribs = booster.get_booster().predict(dm, pred_contribs=True)   # last column is the bias term
        vals = contribs[:, :-1]
        mean_abs = np.abs(vals).mean(axis=0)
        idx = np.argsort(-mean_abs)[:8]
        out["shap"] = {"model": "models/xgb_model.json (published artifact)", "auc_export": round(a_exp, 4),
                       "top_features": [{"feature": list(Xuse.columns)[i], "mean_abs_shap": round(float(mean_abs[i]), 4),
                                         "share_pct": round(float(100 * mean_abs[i] / mean_abs.sum()), 1)} for i in idx],
                       "n_features": int(len(Xuse.columns))}
        print("   Top-5:", [(d["feature"], d["mean_abs_shap"]) for d in out["shap"]["top_features"][:5]], flush=True)
        json.dump({"note": "P1 ML addendum (partial)", "results": out}, open(os.path.join(RES, "p1_ml.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        # optional vector figure for repository use (not part of the supplement layout)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            d = out["shap"]["top_features"][::-1]
            fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)
            ax.barh([x["feature"] for x in d], [x["mean_abs_shap"] for x in d], color="#3F3F3F")
            ax.set_xlabel("mean |SHAP| (log-odds)")
            ax.spines[["top", "right"]].set_visible(False)
            fdir = os.path.join(REPO, "figures"); os.makedirs(fdir, exist_ok=True)
            fp = os.path.join(fdir, "fig_shap_attribution.pdf")
            fig.tight_layout(); fig.savefig(fp, format="pdf"); plt.close(fig)
            out["shap"]["figure"] = "figures/fig_shap_attribution.pdf"
            print("   figure written:", fp, flush=True)
        except Exception as e:
            print("   figure skipped:", e, flush=True)

# ---------- P1-14 feature ladder ----------
print("P1-14 feature ladder ...", flush=True)
XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=4, random_state=42)
LADDER = [("age only", ["age"]), ("3 variables (age, diabetes, SBP)", ["age", "diabetes", "sbp"]),
          ("5 variables (+ BMI, HbA1c)", ["age", "diabetes", "sbp", "bmi", "hba1c"]),
          ("8 variables (+ DBP, total cholesterol, HDL)", ["age", "diabetes", "sbp", "bmi", "hba1c", "dbp", "total_cholesterol", "hdl"]),
          ("12 variables (no missingness indicators)", [c for c in Xtr.columns if not c.startswith("miss_")]),
          ("12 variables + missingness indicators (primary model)", list(Xtr.columns))]
rng = np.random.default_rng(RNG); out["feature_ladder"] = {"items": []}
full = XGBClassifier(**XGBP).fit(Xtr, ytr); p_full = full.predict_proba(Xte)[:, 1]
a_full = roc_auc_score(yte, p_full); print(f"   full model AUC={a_full:.4f}", flush=True)
for tag, cols in LADDER:
    m = XGBClassifier(**XGBP).fit(Xtr[cols], ytr); p = m.predict_proba(Xte[cols])[:, 1]
    a = roc_auc_score(yte, p)
    bs = []
    for _ in range(2000):
        i = rng.choice(np.arange(len(yte)), len(yte), replace=True)
        if len(np.unique(yte[i])) > 1: bs.append(roc_auc_score(yte[i], p[i]) - roc_auc_score(yte[i], p_full[i]))
    out["feature_ladder"]["items"].append({"model": tag, "n_inputs": len(cols), "auc": round(float(a), 4),
                                           "delta_vs_full": round(float(a - a_full), 4),
                                           "delta_vs_full_ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]})
    print(f"   {tag:<52} AUC={a:.4f} Δ={a - a_full:+.4f}", flush=True)
    json.dump({"note": "P1 ML addendum (partial)", "results": out}, open(os.path.join(RES, "p1_ml.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---------- P1-15 extreme-missingness perturbation ----------
print("P1-15 extreme-missingness perturbation ...", flush=True)
def score(Xq):
    return roc_auc_score(yte, full.predict_proba(Xq[list(Xtr.columns)].astype(float))[:, 1])
base = score(Xte)
Xall = Xte.copy()
clin = [c for c in Xte.columns if not c.startswith("miss_")]
Xall[clin] = np.nan; Xall[[c for c in Xte.columns if c.startswith("miss_")]] = np.nan
Xall = f(enc(test[FEATURES]))
a_all = score(Xall)
rng2 = np.random.default_rng(RNG); X50 = Xte.copy()
mask = rng2.random(X50.shape) < 0.5
X50 = X50.mask(mask).astype(float)
# masked missingness flags differ from true missingness: this simulates the worst case of unavailable inputs
a_half = score(X50)
single = {}
for c in clin:
    Xs = Xte.copy(); Xs[c] = np.nan
    if f"miss_{c}" in Xs.columns: Xs[f"miss_{c}"] = 1.0
    single[c] = round(float(base - score(Xs)), 4)
out["missingness_perturbation"] = {"baseline_auc": round(float(base), 4), "all_clinical_inputs_missing_auc": round(float(a_all), 4),
                                   "random_50pct_cells_missing_auc": round(float(a_half), 4),
                                   "single_feature_ablation_delta_auc": dict(sorted(single.items(), key=lambda kv: -kv[1]))}
print("   all-missing AUC=", round(a_all, 4), "| 50% missing-at-random AUC=", round(a_half, 4), flush=True)
print("   largest single-feature delta AUC:", list(out["missingness_perturbation"]["single_feature_ablation_delta_auc"].items())[:3], flush=True)

json.dump({"note": "P1 machine-learning addendum (expert-panel items A-phase)", "results": out},
          open(os.path.join(RES, "p1_ml.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("✅ p1_ml.json written (elapsed %.0fs）" % (time.time() - t0), flush=True)
