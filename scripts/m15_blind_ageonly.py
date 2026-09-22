# -*- coding: utf-8 -*-
"""M15 红队复审补算 —— 盲区(uACR<30)层内 age-only 对照
红队指控：盲区反超 0.865 缺层内 age-only 对照（若 age-only 已 0.8+，则反超多为年龄解释）
输出：盲区内 age-only / age+sex / full AUC + review_fixes2.json
"""
import pandas as pd, numpy as np, os, json
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
cohort = pd.read_parquet(os.path.join(BASE, "results/cohort.parquet"))
FEATURES = ["age","sex","race","poverty_ratio","education","bmi","sbp","dbp",
            "hba1c","diabetes","total_cholesterol","hdl"]
def prep_fit(X):
    med = X.median()
    def tf(X_):
        X_ = X_.copy(); m = X_.isna()
        return pd.concat([X_.fillna(med), m.add_prefix("miss_")], axis=1)
    return tf
def ad(X):
    X = X.copy()
    if "race" in X.columns:
        X["race_black"]=(X["race"]==4).astype(float); X["race_hisp"]=(X["race"].isin([1,2])).astype(float)
        X["race_other"]=(~X["race"].isin([3,4,1,2])).astype(float); X=X.drop(columns=["race"])
    if "sex" in X.columns:
        X["female"]=(X["sex"]==2).astype(float); X=X.drop(columns=["sex"])
    return X

tr = cohort[cohort["year"].isin([2011,2013,2015])]; te = cohort[cohort["year"]==2017]
tr_b = tr[tr["uacr"]<30].reset_index(drop=True)
te_b = te[te["uacr"]<30].reset_index(drop=True)
print(f"盲区子集: train n={len(tr_b)} (CKD {int(tr_b['ckd'].sum())}) | test n={len(te_b)} (CKD {int(te_b['ckd'].sum())})", flush=True)

res = {}
for name, feats in {"age-only": ["age"], "age+sex": ["age","sex"], "age+sex+race": ["age","sex","race"],
                    "clinical-rule": ["age","diabetes","sbp"], "full": FEATURES}.items():
    Xtr = ad(tr_b[feats]); Xte = ad(te_b[feats])
    fit = prep_fit(Xtr); Xtr2, Xte2 = fit(Xtr), fit(Xte)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
    m.fit(Xtr2, tr_b["ckd"].values)
    a = roc_auc_score(te_b["ckd"].values, m.predict_proba(Xte2)[:,1])
    res[name] = round(float(a), 4)
    print(f"  盲区内 {name:<16} AUC={a:.3f}", flush=True)

with open(os.path.join(BASE, "results/m2/review_fixes2.json"), "w", encoding="utf-8") as f:
    json.dump({"blind_stratum_ageonly": res}, f, ensure_ascii=False, indent=2)
print("已保存: review_fixes2.json")
