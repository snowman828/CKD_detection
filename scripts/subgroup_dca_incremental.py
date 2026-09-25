# -*- coding: utf-8 -*-
"""A1+A2 完整版：Table 3 全部 10 亚组的 ΔAUC+CI 与 DCA；另补头条基准差的 CI（vs age-only / vs 临床规则）
自检：XGB 0.8099 / LR 0.7943 / age-only 0.7393 / 临床规则 0.7708（±1e-4），否则中止
输出：results/subgroup_dca_incremental.json
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RES = os.environ.get("MACKI_RESULTS_DIR", r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results")
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
t0 = time.time()
cohort = pd.read_parquet(os.path.join(RES, "cohort.parquet"))
train, test = cohort[cohort["year"].isin([2011, 2013, 2015])], cohort[cohort["year"] == 2017]

def enc(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float); X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float); X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

med = enc(train[FEATURES]).median()
f = lambda D: pd.concat([D.fillna(med), D.isna().add_prefix("miss_")], axis=1)
Xtr, Xte = f(enc(train[FEATURES])), f(enc(test[FEATURES]))
ytr, yte = train["ckd"].values, test["ckd"].values
XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=4, random_state=42)

xgb = XGBClassifier(**XGBP).fit(Xtr, ytr); lr = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
# 基准：age-only XGB 与临床规则 LR（age + diabetes + sbp）
age_tr = enc(train[["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]])[["age"]]
age_te = enc(test[FEATURES])[["age"]]
age_only = XGBClassifier(**XGBP).fit(age_tr, ytr)
rule_cols = ["age", "diabetes", "sbp"]
rule_tr, rule_te = Xtr[rule_cols].fillna(Xtr[rule_cols].median()), Xte[rule_cols].fillna(Xtr[rule_cols].median())
rule = LogisticRegression(max_iter=2000, C=0.1).fit(rule_tr, ytr)

p_x, p_l = xgb.predict_proba(Xte)[:, 1], lr.predict_proba(Xte)[:, 1]
p_a, p_r = age_only.predict_proba(age_te)[:, 1], rule.predict_proba(rule_te)[:, 1]
ref = json.load(open(os.path.join(RES, "model_results.json"), encoding="utf-8"))["XGB"]["auc_test"]
sup = json.load(open(os.path.join(RES, "supplementary_results.json"), encoding="utf-8"))
checks = {"XGB": (roc_auc_score(yte, p_x), ref), "LR": (roc_auc_score(yte, p_l), 0.7943),
          "age_only": (roc_auc_score(yte, p_a), sup["age_only_xgb_auc"]), "clinical_rule": (roc_auc_score(yte, p_r), sup["clinical_rule_lr_auc"])}
print("自检：", {k: f"{v[0]:.4f} vs {v[1]}" for k, v in checks.items()})
if any(abs(a - b) > 1e-4 for a, b in checks.values()):
    sys.exit("❌ 自检未通过：中止")

def boot(y, p1, p2, n=5000, seed=2026):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    return {"delta_auc": round(float(roc_auc_score(y, p1) - roc_auc_score(y, p2)), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_boot": round(float((d <= 0).mean()), 4)}

def dca(y, p, ths):
    n = len(y); ev = int(y.sum()); o = {}
    for pt in ths:
        pred = p >= pt
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        nb = tp / n - (fp / n) * (pt / (1 - pt)); nb_all = ev / n - ((n - ev) / n) * (pt / (1 - pt))
        o[f"{pt:.2f}"] = {"nb_model": round(float(nb), 4), "nb_screen_all": round(float(nb_all), 4), "delta_nb": round(float(nb - nb_all), 4)}
    win = [pt for pt in ths if o[f"{pt:.2f}"]["delta_nb"] > 0]
    o["window_model_gt_all"] = [min(win), max(win)] if win else None
    return o

THS = [round(0.05 + 0.01 * i, 2) for i in range(21)]
te = test.reset_index(drop=True)
SUBS = [("age_18_44", "Age 18–44 y", (te["age"] < 45).values), ("age_45_64", "Age 45–64 y", ((te["age"] >= 45) & (te["age"] < 65)).values),
        ("age_65plus", "Age ≥65 y", (te["age"] >= 65).values), ("sex_M", "Male", (te["sex"] == 1).values),
        ("sex_F", "Female", (te["sex"] == 2).values), ("race_white", "White", (te["race"] == 3).values),
        ("race_black", "Black", (te["race"] == 4).values), ("race_hisp", "Hispanic", (te["race"].isin([1, 2])).values),
        ("pov_lt1.3", "Poverty-income ratio <1.3", (te["poverty_ratio"] < 1.3).values),
        ("pov_ge1.3", "Poverty-income ratio ≥1.3", (te["poverty_ratio"] >= 1.3).values)]

out = {"_note": "Pre-submission addendum (expert-panel items A1/A2): within-subgroup ΔAUC (XGBoost vs logistic "
                "regression) with paired-bootstrap 95% CIs, benchmark ΔAUC CIs, and subgroup decision-curve analysis. "
                "Exploratory; not adjusted for multiplicity. Seed 2026, 5,000 resamples (same framework as the primary analysis).",
       "overall": {"n": int(len(te)), "events": int(yte.sum()), "auc_xgb": round(float(roc_auc_score(yte, p_x)), 4),
                   "auc_lr": round(float(roc_auc_score(yte, p_l)), 4),
                   "delta_xgb_vs_lr": boot(yte, p_x, p_l), "delta_xgb_vs_age_only": boot(yte, p_x, p_a),
                   "delta_xgb_vs_clinical_rule": boot(yte, p_x, p_r),
                   "dca": dca(yte, p_x, THS)}, "subgroups": {}}
print("总体 ΔAUC vs LR:", out["overall"]["delta_xgb_vs_lr"], "| vs age-only:", out["overall"]["delta_xgb_vs_age_only"],
      "| vs 临床规则:", out["overall"]["delta_xgb_vs_clinical_rule"])
for key, label, mask in SUBS:
    y = yte[mask]
    if len(np.unique(y)) < 2: continue
    out["subgroups"][key] = {"label": label, "n": int(mask.sum()), "events": int(y.sum()),
                             "auc_xgb": round(float(roc_auc_score(y, p_x[mask])), 4),
                             "auc_lr": round(float(roc_auc_score(y, p_l[mask])), 4),
                             "delta_auc_ci": boot(y, p_x[mask], p_l[mask]), "dca": dca(y, p_x[mask], THS)}
    d = out["subgroups"][key]["delta_auc_ci"]
    print(f"  {label:<28} ΔAUC={d['delta_auc']:+.4f} CI=[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] P={d['p_boot']} | 优于全筛区间={out['subgroups'][key]['dca']['window_model_gt_all']}")
out["thresholds"] = THS
dst = os.path.join(RES, "subgroup_dca_incremental.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n写出 {dst} ({os.path.getsize(dst):,} B) 用时 {time.time()-t0:.0f}s")
