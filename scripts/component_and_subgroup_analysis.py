# -*- coding: utf-8 -*-
"""分量 AUC 与亚组校准（投稿前审计补强分析；对应审计报告 M4/M5）

输出 `results/component_subgroup_results.json`，供 SI Table S4 引用。
- 分量 AUC：模型分数对 (a) eGFR<60（仅 eGFR 可测者）与 (b) uACR≥30（仅 uACR 可测者）分别求 AUC
- 亚组校准：按 age/sex/race/PIR 分层，十分位拟合校准斜率/截距 + O:E 比（探索性，不做多重校正）
- 自检：重训 XGB 的 test AUC 必须等于 `results/model_results.json` 的 0.8099（±1e-4），否则中止
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("MACKI_RESULTS_DIR", r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results")

FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi",
            "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
cohort = pd.read_parquet(os.path.join(RES, "cohort.parquet"))
train, test = cohort[cohort["year"].isin([2011, 2013, 2015])], cohort[cohort["year"] == 2017]

def enc(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float); X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float); X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

def prep(Xtr, Xte):
    m = Xtr.median()
    f = lambda D: pd.concat([D.fillna(m), D.isna().add_prefix("miss_")], axis=1)
    return f(Xtr), f(Xte)

Xtr, Xte = prep(enc(train[FEATURES]), enc(test[FEATURES]))
ytr, yte = train["ckd"].values, test["ckd"].values
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42).fit(Xtr, ytr)
p = xgb.predict_proba(Xte)[:, 1]
auc = float(roc_auc_score(yte, p))
ref = json.load(open(os.path.join(RES, "model_results.json"), encoding="utf-8"))["XGB"]["auc_test"]
print(f"自检: 重训 test AUC={auc:.6f} vs 已发表 {ref} |Δ|={abs(auc-ref):.2e}")
if abs(auc - ref) > 1e-4:
    sys.exit("❌ 自检未通过：中止（不产出不一致的数字）")

te = test.reset_index(drop=True)
out = {"_note": "Exploratory addendum analyses (component discrimination and subgroup calibration) "
                "for the detection manuscript; not adjusted for multiplicity.",
       "model": "XGBoost (300 trees, depth 4, lr 0.05, subsample/colsample 0.8, seed 42)",
       "overall_auc_composite_ckd": round(auc, 4), "n_test": int(len(te)), "n_ckd": int(yte.sum())}

# ---------- 1. 分量 AUC ----------
comp = {}
egfr_ok = te["egfr"].notna().values
uacr_ok = te["uacr"].notna().values
for label, ok, cond in [("egfr_lt60", egfr_ok, (te["egfr"] < 60).values),
                        ("uacr_ge30", uacr_ok, (te["uacr"] >= 30).values)]:
    yy = cond[ok].astype(int)
    if len(np.unique(yy)) == 2:
        comp[label] = {"n_assessable": int(ok.sum()), "n_positive": int(yy.sum()),
                       "auc": round(float(roc_auc_score(yy, p[ok])), 4)}
        print(f"  分量 {label}: n={ok.sum()} 阳性={yy.sum()} AUC={comp[label]['auc']}")
out["component_auc"] = comp

# ---------- 2. 亚组校准 ----------
def calib(y, pp):
    d = pd.DataFrame({"y": y, "p": pp})
    d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(mp=("p", "mean"), my=("y", "mean"), n=("y", "size"))
    slope, intercept = np.polyfit(g["mp"], g["my"], 1)
    oe = float(d["y"].mean() / d["p"].mean()) if d["p"].mean() > 0 else None
    return round(float(slope), 3), round(float(intercept), 4), round(oe, 3), int(len(d)), int(d["y"].sum())

subs = [("age_18_44", te["age"] < 45), ("age_45_64", (te["age"] >= 45) & (te["age"] < 65)),
        ("age_65plus", te["age"] >= 65), ("sex_M", te["sex"] == 1), ("sex_F", te["sex"] == 2),
        ("race_white", te["race"] == 3), ("race_black", te["race"] == 4), ("race_hisp", te["race"].isin([1, 2])),
        ("pov_lt1.3", te["poverty_ratio"] < 1.3), ("pov_ge1.3", te["poverty_ratio"] >= 1.3)]
sg = {}
for label, mask in subs:
    m = mask.values
    if m.sum() >= 50 and len(np.unique(yte[m])) == 2:
        s, i, oe, n, ev = calib(yte[m], p[m])
        sg[label] = {"n": n, "events": ev, "calibration_slope": s, "calibration_intercept": i, "observed_expected_ratio": oe}
        print(f"  亚组校准 {label:<12} n={n:<5} 事件={ev:<4} slope={s} intercept={i} O:E={oe}")
out["subgroup_calibration"] = sg

dst = os.path.join(RES, "component_subgroup_results.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n写出: {dst} ({os.path.getsize(dst):,} B)")
