"""S1b: does the within-stratum increment follow the age exponent?

For each age exponent, rebuild the CKD label, retrain the full model and an
age-only model *within the albuminuria-blind stratum*, and report the AUC gap.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

BASE = r"G:\项目文件\CKD多模态AI早诊_本项目专属归档\09_无湿实验执行"
RES = os.path.join(BASE, "results")
FE = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp",
      "hba1c", "diabetes", "total_cholesterol", "hdl"]

c = pd.read_parquet(os.path.join(RES, "cohort.parquet"))
tr = c[c["year"].isin([2011, 2013, 2015])].copy()
te = c[c["year"] == 2017].copy().reset_index(drop=True)


def enc(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])


def egfr(exp_age, D):
    fem = (D["sex"] == 2).values
    k = np.where(fem, 0.7, 0.9)
    a = np.where(fem, -0.241, -0.302)
    r = D["creatinine"].values.astype(float) / k
    return (142.0 * np.minimum(r, 1.0) ** a * np.maximum(r, 1.0) ** -1.200
            * (exp_age ** D["age"].values.astype(float)) * np.where(fem, 1.012, 1.0))


def relabel(D, exp_age):
    e = egfr(exp_age, D)
    return ((e < 60) | (D["uacr"].values >= 30)).astype(int)


med = enc(tr[FE]).median()
f = lambda D: pd.concat([D.fillna(med), D.isna().add_prefix("miss_")], axis=1)

rows = []
for exp_age in (1.0000, 0.9938, 0.9900):
    ytr = relabel(tr, exp_age)
    yte = relabel(te, exp_age)
    blind = (te["uacr"] < 30).values
    # fit BOTH models inside the albuminuria-blind stratum (manuscript design)
    trb = (tr["uacr"].values < 30)
    Xtr_b, Xte_b = f(enc(tr[FE])).loc[trb], f(enc(te[FE]))
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
    m.fit(Xtr_b, ytr[trb])
    p = m.predict_proba(Xte_b)[:, 1]
    age_only = LogisticRegression(max_iter=2000, C=0.1).fit(
        tr.loc[trb, ["age"]].values, ytr[trb])
    pa = age_only.predict_proba(te[["age"]].values)[:, 1]

    sub = blind & (np.isin(yte, [0, 1]))
    yy = yte[sub]
    if len(np.unique(yy)) < 2:
        continue
    auc_full = roc_auc_score(yy, p[sub])
    auc_age = roc_auc_score(yy, pa[sub])
    rows.append({"age_exponent": exp_age, "n_blind": int(sub.sum()),
                 "n_cases": int(yy.sum()), "auc_age_only": round(float(auc_age), 4),
                 "auc_full": round(float(auc_full), 4),
                 "delta": round(float(auc_full - auc_age), 4)})
    print(rows[-1])

out = {"design": "labels rebuilt per age exponent; models retrained; blind stratum only",
       "rows": rows}
path = os.path.join(RES, "m2", "age_term_dose_response_auc.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written:", path)
