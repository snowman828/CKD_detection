# -*- coding: utf-8 -*-
"""阶段 B：跨时代独立外部验证（NHANES 2007-2010，E+F 周期）
- 开发集与主分析完全一致（2011/2013/2015 训练，2017-2018 自检：XGB 0.8099 / LR 0.7943）
- 冻结同一模型，评估于独立时代队列（2007-2008 + 2009-2010，未被任何环节使用）
输出：results/p2_independent_era.json
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

t0 = time.time(); RNG = 2026
DATA = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data"
OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi", "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]

def load(f):
    p = os.path.join(DATA, f + ".XPT")
    return pd.read_sas(p, format="xport") if os.path.exists(p) else None

def build(cyc, year):
    dem, alb, bio = load(f"DEMO_{cyc}"), load(f"ALB_CR_{cyc}"), load(f"BIOPRO_{cyc}")
    bpx, bmx, diq, ghb = load(f"BPX_{cyc}"), load(f"BMX_{cyc}"), load(f"DIQ_{cyc}"), load(f"GHB_{cyc}")
    if dem is None:
        print(f"  ⚠️ DEMO_{cyc} 缺失"); return None
    need = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "WTMEC2YR"]
    miss = [c for c in need if c not in dem.columns]
    if miss:
        print(f"  ⚠️ {cyc} DEMO 缺列: {miss}"); return None
    df = dem[need + ([c for c in ["RIDEXPRG"] if c in dem.columns])].copy()
    df["cycle"], df["year"] = cyc, year
    df = df.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH1": "race",
                            "INDFMPIR": "poverty_ratio", "DMDEDUC2": "education", "WTMEC2YR": "wt"})
    df["pregnant"] = df["RIDEXPRG"] if "RIDEXPRG" in df.columns else np.nan
    m = lambda s, cols, pre: (df.merge(s[["SEQN"] + cols].rename(columns={c: pre + c for c in cols}), on="SEQN", how="left") if s is not None else df)
    df = m(alb, ["URXUMA", "URXUCR"], "a_"); df = m(bio, ["LBXSCR"], "b_")
    df = m(bpx, ["BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2"], "c_"); df = m(bmx, ["BMXBMI"], "d_")
    df = m(diq, ["DIQ010"], "e_"); df = m(ghb, ["LBXGH"], "f_")
    df = m(load(f"TCHOL_{cyc}"), ["LBXTC"], "g_"); df = m(load(f"HDL_{cyc}"), ["LBDHDD"], "g_")
    df["uacr"] = (df["a_URXUMA"] * 100.0 / df["a_URXUCR"]).where(df["a_URXUMA"].notna() & df["a_URXUCR"].notna() & (df["a_URXUCR"] > 0))
    df["creatinine"] = df["b_LBXSCR"]
    k = np.where(df["sex"] == 2, 0.7, 0.9); al = np.where(df["sex"] == 2, -0.241, -0.302)
    s = df["creatinine"] / k
    df["egfr"] = (142.0 * np.minimum(s, 1.0) ** al * np.maximum(s, 1.0) ** (-1.200) * 0.9938 ** df["age"] * np.where(df["sex"] == 2, 1.012, 1.0)).where(df["creatinine"].notna())
    df["sbp"] = df["c_BPXSY1"].fillna(df["c_BPXSY2"]); df["dbp"] = df["c_BPXDI1"].fillna(df["c_BPXDI2"])
    df["bmi"] = df["d_BMXBMI"]; df["hba1c"] = df["f_LBXGH"]
    df["total_cholesterol"] = df["g_LBXTC"]; df["hdl"] = df["g_LBDHDD"]
    df["diabetes"] = np.where(df["e_DIQ010"] == 1, 1.0, np.where(df["e_DIQ010"] == 2, 0.0, np.nan))
    df["ckd"] = ((df["egfr"] < 60) | (df["uacr"] >= 30)).astype(float)
    df = df[(df["age"] >= 18) & (df["pregnant"] != 1) & df["ckd"].notna()]
    print(f"  {cyc}({year}): n={len(df)} CKD={int(df['ckd'].sum())}")
    return df

print("构建队列 ...", flush=True)
dev = pd.concat([build(c, y) for c, y in (("G", 2011), ("H", 2013), ("I", 2015))], ignore_index=True)
val = build("J", 2017)
eraF = pd.concat([x for x in (build("E", 2007), build("F", 2009)) if x is not None], ignore_index=True)
print(f"  开发 n={len(dev)} | 主验证 n={len(val)} | 跨时代外部队列 n={len(eraF)} CKD={int(eraF['ckd'].sum())}", flush=True)

def enc(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float); X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float); X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

med = enc(dev[FEATURES]).median()
f = lambda D: pd.concat([D.fillna(med), D.isna().add_prefix("miss_")], axis=1)
Xtr, Xv, Xe = f(enc(dev[FEATURES])), f(enc(val[FEATURES])), f(enc(eraF[FEATURES]))
ytr, yv, ye = dev["ckd"].values, val["ckd"].values, eraF["ckd"].values
XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
xgb = XGBClassifier(**XGBP).fit(Xtr, ytr); lr = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
age_only = XGBClassifier(**XGBP).fit(enc(dev[FEATURES])[["age"]], ytr)
rc = ["age", "diabetes", "sbp"]; rule = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr[rc], ytr)
sup = json.load(open(os.path.join(OUT, "supplementary_results.json"), encoding="utf-8"))
print("自检(2017-2018):", f"XGB={roc_auc_score(yv, xgb.predict_proba(Xv)[:,1]):.4f}(0.8099) LR={roc_auc_score(yv, lr.predict_proba(Xv)[:,1]):.4f}(0.7943) rule={roc_auc_score(yv, rule.predict_proba(Xv[rc])[:,1]):.4f}({sup['clinical_rule_lr_auc']})", flush=True)
if abs(roc_auc_score(yv, xgb.predict_proba(Xv)[:, 1]) - 0.8099) > 1e-4: sys.exit("❌ 自检未通过")

p_x, p_l = xgb.predict_proba(Xe)[:, 1], lr.predict_proba(Xe)[:, 1]
p_a, p_r = age_only.predict_proba(enc(eraF[FEATURES])[["age"]])[:, 1], rule.predict_proba(Xe[rc])[:, 1]
YTHR = 0.1602

def boot(y, p, n=5000, seed=RNG):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); o = []
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
    a = np.array(o); return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

def calib(y, p, nd=10):
    q = np.quantile(p, np.linspace(0, 1, nd + 1)); q[0], q[-1] = -np.inf, np.inf
    mp, ob = [], []
    for i in range(nd):
        m = (p > q[i]) & (p <= q[i + 1])
        if m.sum() == 0: continue
        mp.append(p[m].mean()); ob.append(y[m].mean())
    mp, ob = np.array(mp), np.array(ob)
    c = np.polyfit(mp, ob, 1)
    return float(c[0]), float(c[1]), float(np.mean(np.abs(mp - ob)))

def ops(y, p, thr):
    pred = p >= thr; n = len(y)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn_ = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"sensitivity": round(tp / max(tp + fn_, 1e-9), 4), "specificity": round(tn / max(tn + fp, 1e-9), 4),
            "ppv": round(tp / max(tp + fp, 1e-9), 4), "npv": round(tn / max(tn + fn_, 1e-9), 4),
            "flagged_per1000": round(1000 * (tp + fp) / n, 1), "tp_per1000": round(1000 * tp / n, 1)}

def nb(y, p, thr):
    pred = p >= thr; n = len(y)
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    return float(tp / n - (fp / n) * (thr / (1 - thr)))

THS = [round(0.05 + 0.01 * i, 2) for i in range(21)]
w = eraF["wt"].fillna(0).values
def w_auc(y, p, ww):
    o = np.argsort(p); y, ww = y[o], ww[o]
    tw = np.cumsum(ww * y); fw = np.cumsum(ww * (1 - y))
    tp = np.concatenate([[0], tw / max(tw[-1], 1e-12)]); fp = np.concatenate([[0], fw / max(fw[-1], 1e-12)])
    return float(1.0 - np.trapezoid(tp, fp))

sl, ic, ec = calib(ye, p_x)
win = [t for t in THS if nb(ye, p_x, t) - (ye.mean() - (1 - ye.mean()) * (t / (1 - t))) > 0]
out = {
 "_note": "Independent-era external validation (NHANES 2007-2008 and 2009-2010; never used for development, threshold derivation or model selection). Models are identical to the primary analysis (trained on 2011/2013/2015); the 2017-2018 external validation reproduces the published AUC (0.8099) as an internal consistency check.",
 "cohort": {"n": int(len(eraF)), "events": int(ye.sum()), "prevalence_pct": round(float(100 * ye.mean()), 2),
            "cycles": {c: int((eraF["cycle"] == c).sum()) for c in ("E", "F")},
            "age_mean": round(float(eraF["age"].mean()), 1), "female_pct": round(float(100 * (eraF["sex"] == 2).mean()), 1),
            "diabetes_pct": round(float(100 * np.nanmean(eraF["diabetes"])), 1),
            "egfr_median": round(float(eraF["egfr"].median()), 1), "uacr_median": round(float(eraF["uacr"].median()), 1)},
 "discrimination": {"xgb": round(float(roc_auc_score(ye, p_x)), 4), "xgb_ci": boot(ye, p_x),
                    "lr": round(float(roc_auc_score(ye, p_l)), 4), "lr_ci": boot(ye, p_l),
                    "age_only": round(float(roc_auc_score(ye, p_a)), 4), "age_only_ci": boot(ye, p_a),
                    "clinical_rule": round(float(roc_auc_score(ye, p_r)), 4), "clinical_rule_ci": boot(ye, p_r),
                    "weighted_xgb": round(w_auc(ye, p_x, w), 4)},
 "calibration": {"slope": round(sl, 3), "intercept": round(ic, 4), "ece": round(ec, 4)},
 "operating": {"threshold": YTHR, "xgb": ops(ye, p_x, YTHR)},
 "decision_curve": {"at_10pct": {"nb_model": round(nb(ye, p_x, 0.10), 4), "nb_screen_all": round(float(ye.mean() - (1 - ye.mean()) * (0.10 / 0.90)), 4)},
                    "at_20pct": {"nb_model": round(nb(ye, p_x, 0.20), 4), "nb_screen_all": round(float(ye.mean() - (1 - ye.mean()) * (0.20 / 0.80)), 4)},
                    "window_model_gt_all": [min(win), max(win)] if win else None},
}
json.dump(out, open(os.path.join(OUT, "p2_independent_era.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1)[:1800], flush=True)
print(f"\n✅ 已写 results/p2_independent_era.json（用时 {time.time()-t0:.0f}s）", flush=True)
