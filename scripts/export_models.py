# -*- coding: utf-8 -*-
"""Export trained model artifacts for reuse.

Design
- Reproduces the preprocessing and hyperparameters of modeling.py (train 2011/2013/2015; test 2017).
- Exports the fitted artifacts, imputation statistics and the exact column order required for inference.
- Self-check (aborts rather than continuing silently): predictions from the exported artifacts must match
  the recorded test AUCs in results/model_results.json (|delta| <= 1e-4), and are compared with
  the saved probabilities for maximum absolute difference and label agreement.
- Output: models/ (xgb_model.json, lr_model.joblib, mlp_model.joblib,
  preprocessing.json / model_metadata.json / README.md）
"""
import hashlib, json, os, platform, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # repository root
RES = os.environ.get("CKD-detection_RESULTS_DIR", r"results")
OUT = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)

# ---- precondition check: give actionable guidance when gitignored intermediates are absent ----
_REQUIRED = [os.path.join(RES, "cohort.parquet"), os.path.join(RES, "model_results.json")]
_missing = [p for p in _REQUIRED if not os.path.exists(p)]
if _missing:
    import sys as _sys
    print("❌ export_models.py required inputs are missing(gitignored intermediates, not distributed with the repository):")
    for p in _missing:
        print(f"   - {os.path.basename(p)}")
    print("  They are rebuilt from public NHANES data (see README, Track 2 - full pipeline):")
    print("      python scripts/download_nhanes.py   # four clinical-cycle files")
    print("      python scripts/build_cohort.py     # cohort.parquet")
    print("      python scripts/modeling.py         # model_results.json + test_proba_*.npy")
    print("  Then re-run this script to reproduce every artifact in models/ and its self-check.")
    _sys.exit(1)

FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi",
            "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
TARGET = "ckd"

cohort = pd.read_parquet(os.path.join(RES, "cohort.parquet"))
train = cohort[cohort["year"].isin([2011, 2013, 2015])]
test = cohort[cohort["year"] == 2017]

def add_dummies(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
med = Xtr_raw.median()                       # fitted on the training set only, as in modeling.py

def transform(X_):
    X_ = X_.copy()
    miss = X_.isna()
    X_ = X_.fillna(med)
    return pd.concat([X_, miss.add_prefix("miss_")], axis=1)

Xtr, Xte = transform(Xtr_raw), transform(Xte_raw)
ytr, yte = train[TARGET].values, test[TARGET].values
print(f"train {Xtr.shape} | test {Xte.shape} | encoded columns={Xte.shape[1]}(12 raw variables expanded into dummies and indicators)")
print(f"  encoded column order: {list(Xte.columns)}")

MODELS = {
    "LR": LogisticRegression(max_iter=2000, C=0.1),
    "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         n_jobs=4, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500,
                         early_stopping=True, random_state=42),
}

ref = json.load(open(os.path.join(RES, "model_results.json"), encoding="utf-8"))
probas, checks = {}, {}
for name, model in MODELS.items():
    model.fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    probas[name] = p_te
    auc = float(roc_auc_score(yte, p_te))
    ref_auc = float(ref[name]["auc_test"])
    ok = abs(auc - ref_auc) <= 1e-4
    arch = os.path.join(RES, f"test_proba_{name.lower()}.npy")
    maxdiff, agree = None, None
    if os.path.exists(arch):
        p_arch = np.load(arch)
        maxdiff = float(np.max(np.abs(p_te - p_arch)))
        agree = float(np.mean((p_te >= 0.5) == (p_arch >= 0.5)))
    checks[name] = {"auc_recomputed": round(auc, 6), "auc_published": ref_auc,
                    "abs_diff": round(abs(auc - ref_auc), 8), "pass": ok,
                    "max_prob_diff_vs_archive": maxdiff, "label_agreement": agree}
    print(f"[{name}] AUC recomputed={auc:.6f} 已发表={ref_auc:.6f} |delta|= {abs(auc-ref_auc):.2e} {'✅' if ok else '❌'}"
          + (f" | max difference vs saved probabilities={maxdiff:.2e} label agreement={agree:.1%}" if maxdiff is not None else ""))

if not all(c["pass"] for c in checks.values()):
    sys.exit("self-check failed: exported model disagrees with the published results; aborting without writing artifacts")

# ---------- export artifacts ----------
MODELS["XGB"].save_model(os.path.join(OUT, "xgb_model.json"))
import joblib
joblib.dump(MODELS["LR"], os.path.join(OUT, "lr_model.joblib"))
joblib.dump(MODELS["MLP"], os.path.join(OUT, "mlp_model.joblib"))

prep = {
    "features_raw": FEATURES,
    "encoding": {
        "race_black": "race == 4",
        "race_hisp": "race in (1, 2)",
        "race_other": "race not in (3, 4, 1, 2)",
        "female": "sex == 2",
        "dropped": ["race", "sex"],
    },
    "imputation": {"strategy": "median (fit on development set only)",
                   "median_values": {k: (None if pd.isna(v) else float(v)) for k, v in med.items()}},
    "final_columns": list(Xte.columns),
    "final_n_features": int(Xte.shape[1]),
    "note": ("Apply: select features_raw → one-hot as above → fill NaN with median_values → "
             "append miss_<column> indicators in final_columns order → predict_proba."),
}
json.dump(prep, open(os.path.join(OUT, "preprocessing.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

arts = {f: {"sha256": sha(os.path.join(OUT, f)), "bytes": os.path.getsize(os.path.join(OUT, f))}
        for f in ["xgb_model.json", "lr_model.joblib", "mlp_model.joblib", "preprocessing.json"]}
meta = {
    "generated_by": "scripts/export_models.py",
    "source_cohort": "results/cohort.parquet (NHANES 2011–2018)",
    "train": "NHANES 2011–2012, 2013–2014, 2015–2016", "test": "NHANES 2017–2018",
    "hyperparameters": {
        "LR": "LogisticRegression(max_iter=2000, C=0.1)",
        "XGB": "XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric=logloss, random_state=42)",
        "MLP": "MLPClassifier(hidden_layer_sizes=(64,32), alpha=1e-3, max_iter=500, early_stopping=True, random_state=42)",
    },
    "library_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": __import__("sklearn").__version__, "xgboost": __import__("xgboost").__version__},
    "reproduction_check": checks,
    "artifacts": arts,
}
json.dump(meta, open(os.path.join(OUT, "model_metadata.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

readme = f"""# Trained model artifacts (CKD-detection CKD detection)

These artifacts are the **exact models** whose external-validation performance is reported in the
manuscript (XGBoost AUC 0.8099; logistic regression 0.7943; multilayer perceptron 0.7695 on
NHANES 2017–2018). They were regenerated by `scripts/export_models.py` and verified against
`results/model_results.json` (|Δ AUC| ≤ 1e-4) and `results/test_proba_*.npy`.

| file | model | sha256 (first 16) |
|---|---|---|
| `xgb_model.json` | XGBoost (300 trees, depth 4, lr 0.05) | `{arts['xgb_model.json']['sha256'][:16]}` |
| `lr_model.joblib` | L2 logistic regression (C=0.1) | `{arts['lr_model.joblib']['sha256'][:16]}` |
| `mlp_model.joblib` | MLP (64–32, L2 1e-3, early stopping) | `{arts['mlp_model.joblib']['sha256'][:16]}` |
| `preprocessing.json` | imputation medians + exact column order | `{arts['preprocessing.json']['sha256'][:16]}` |

## How to apply a model

```python
import json, joblib, pandas as pd
prep = json.load(open("models/preprocessing.json"))
model = joblib.load("models/lr_model.joblib")          # or xgboost.XGBClassifier(); booster.load_model(...)

def encode(df):                                        # df: raw 12 features, same names as NHANES cohort
    X = df[prep["features_raw"]].copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"]  = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"]     = (X["sex"] == 2).astype(float)
    X = X.drop(columns=["race", "sex"])
    miss = X.isna()
    X = X.fillna(pd.Series(prep["imputation"]["median_values"]))
    X = pd.concat([X, miss.add_prefix("miss_")], axis=1)
    return X[prep["final_columns"]]                    # column order matters

p = model.predict_proba(encode(df))[:, 1]
```

**Note on model inputs**: {Xte.shape[1]} encoded columns are passed to the models
(12 raw variables → race and sex expanded into 4 indicator columns = 14; plus 14 missingness
indicators). Four indicators (`miss_race_*`, `miss_female`) are constant zero because race and sex
had no missing values in these NHANES cycles.

No protected health information is included; all inputs derive from public NHANES files.
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)

print("\n=== export complete ===")
for f, d in arts.items():
    print(f"  {f:<22} {d['bytes']:>9,} B  sha256 {d['sha256'][:16]}")
print("  README.md / model_metadata.json")
print(f"  自检: 全部通过（{', '.join(k for k, v in checks.items() if v['pass'])}）")
