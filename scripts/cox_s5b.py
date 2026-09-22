# -*- coding: utf-8 -*-
"""S5-B 补真正的 Cox 生存分析（lifelines），消除二分类 Harrell C 的「偏离声明」
全因死亡 + CVD 死亡，full 12 特征 vs age-only 对照，测试集 C-index。
"""
import pandas as pd, numpy as np, json, os
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
cohort = pd.read_parquet(os.path.join(BASE, "results/cohort.parquet"))

FEATURES = ["age","sex","race","poverty_ratio","education","bmi","sbp","dbp",
            "hba1c","diabetes","total_cholesterol","hdl"]

def add_dummies(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"]  = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

def prep_fit(X):
    med = X.median()
    def transform(X_):
        X_ = X_.copy(); miss = X_.isna()
        return pd.concat([X_.fillna(med), miss.add_prefix("miss_")], axis=1)
    return transform

def read_lmf(path):
    widths = [(1,6),(15,15),(16,16),(17,19),(20,20),(21,21),(22,22),(23,26),(43,45),(46,48)]
    cols = ["seqn","eligstat","mortstat","ucod_leading","diabetes","hyperten","dodqtr","dodyear","permth_int","permth_exm"]
    recs = []
    for line in open(path):
        if len(line) < 48: continue
        vals = {}
        for (a,b), c in zip(widths, cols):
            s = line[a-1:b].strip()
            vals[c] = float(s) if s not in ("", ".") else np.nan
        recs.append(vals)
    return pd.DataFrame(recs)

lmf_dir = os.path.join(BASE, "data/lmf")
parts = []
for cyc, fn in [("2011_2012","NHANES_2011_2012_MORT_2019_PUBLIC.dat"),
                ("2013_2014","NHANES_2013_2014_MORT_2019_PUBLIC.dat"),
                ("2015_2016","NHANES_2015_2016_MORT_2019_PUBLIC.dat"),
                ("2017_2018","NHANES_2017_2018_MORT_2019_PUBLIC.dat")]:
    p = read_lmf(os.path.join(lmf_dir, fn)); p["cycle"] = cyc; parts.append(p)
lmf = pd.concat(parts, ignore_index=True)

df = cohort.merge(lmf[["seqn","mortstat","ucod_leading","permth_int"]],
                  left_on="SEQN", right_on="seqn", how="left")
df = df[df["permth_int"].notna()].reset_index(drop=True)  # drop 未匹配 LMF 者
df["fu_years"] = df["permth_int"] / 12.0
df["mort"] = (df["mortstat"] == 1).astype(float)
df["cvd"] = ((df["mortstat"] == 1) & (df["ucod_leading"] == 1)).astype(float)

X_all_raw = add_dummies(df[FEATURES])
tr_mask = df["year"].isin([2011, 2013, 2015])
te_mask = df["year"] == 2017
fit = prep_fit(X_all_raw[tr_mask])
X_all = fit(X_all_raw)
# drop 训练集上方差为 0（完全常数）的列，避免 Cox 共线/NaN
tr_idx0 = df.index[tr_mask]
const_cols = X_all.loc[tr_idx0].std() == 0
if const_cols.any():
    print(f"  drop 常数列: {list(X_all.columns[const_cols])}", flush=True)
    X_all = X_all.loc[:, ~const_cols]

results = {}
for oname, ev in [("all_cause","mort"), ("cvd","cvd")]:
    tr_idx = df.index[tr_mask]; te_idx = df.index[te_mask]
    tr_df = pd.DataFrame(X_all.loc[tr_idx].values, columns=X_all.columns)
    tr_df["T"] = df.loc[tr_idx, "fu_years"].values
    tr_df["E"] = df.loc[tr_idx, ev].values
    te_df = pd.DataFrame(X_all.loc[te_idx].values, columns=X_all.columns)

    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(tr_df, duration_col="T", event_col="E")
    full_score = cph.predict_partial_hazard(te_df)
    c_full = concordance_index(df.loc[te_idx, "fu_years"].values,
                               -full_score.values, df.loc[te_idx, ev].values)

    age_cols = [c for c in X_all.columns if c.startswith("age") or c.startswith("miss_age")]
    tr_age = tr_df[["T", "E"] + age_cols]
    cph_age = CoxPHFitter(penalizer=0.5)
    cph_age.fit(tr_age, duration_col="T", event_col="E")
    age_score = cph_age.predict_partial_hazard(te_df[age_cols])
    c_age = concordance_index(df.loc[te_idx, "fu_years"].values,
                              -age_score.values, df.loc[te_idx, ev].values)

    results[oname] = {
        "c_index_full": round(float(c_full), 4),
        "c_index_ageonly": round(float(c_age), 4),
        "delta": round(float(c_full - c_age), 4),
        "n_train_events": int(df.loc[tr_idx, ev].sum()),
        "n_test_events": int(df.loc[te_idx, ev].sum()),
    }
    print(f"[{oname}] full C={c_full:.4f} vs age-only C={c_age:.4f} (Δ{c_full-c_age:+.4f}) | "
          f"train events {results[oname]['n_train_events']}, test events {results[oname]['n_test_events']}", flush=True)

os.makedirs(os.path.join(BASE, "results/m2"), exist_ok=True)
with open(os.path.join(BASE, "results/m2/cox_s5b.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved: results/m2/cox_s5b.json", flush=True)
