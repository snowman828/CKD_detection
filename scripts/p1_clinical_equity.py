# -*- coding: utf-8 -*-
"""阶段 A · Wave 2 · 临床/公平批：
P1-05 重校准实证(K折CV) / P1-06 BH-FDR / P1-07 标签不确定性量化(概率偏差分析) /
P1-08 临床影响量化 / P1-09 1,233 例画像 / P1-10 加权校准与DCA / P1-11 亚组PPV / P1-12 分组阈值与等赔率 / P1-17 交叉性
自检：XGB 0.8099 / LR 0.7943 / 临床规则 0.7708；两标志缺失数=1,233；否则中止
输出：results/p1_clinical_equity.json
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

t0 = time.time(); RNG = 2026
RES = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
DATA = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data"
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
XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=4, random_state=42)
xgb = XGBClassifier(**XGBP).fit(Xtr, ytr); lr = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
rc = ["age", "diabetes", "sbp"]
rule = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr[rc], ytr)
p_x, p_l = xgb.predict_proba(Xte)[:, 1], lr.predict_proba(Xte)[:, 1]
p_r = rule.predict_proba(Xte[rc])[:, 1]
p_x_tr = xgb.predict_proba(Xtr)[:, 1]
sup = json.load(open(os.path.join(RES, "supplementary_results.json"), encoding="utf-8"))
chk = {"XGB": (roc_auc_score(yte, p_x), 0.8099), "LR": (roc_auc_score(yte, p_l), 0.7943), "rule": (roc_auc_score(yte, p_r), sup["clinical_rule_lr_auc"])}
print("自检:", {k: f"{v[0]:.4f} vs {v[1]}" for k, v in chk.items()}, flush=True)
if any(abs(a - b) > 1e-4 for a, b in chk.values()): sys.exit("❌ 自检未通过")
YTHR = 0.1602

# 抽样设计变量
ds = []
for fn, yr in (("DEMO_G.XPT", 2011), ("DEMO_H.XPT", 2013), ("DEMO_I.XPT", 2015), ("DEMO_J.XPT", 2017)):
    p = os.path.join(DATA, fn)
    if os.path.exists(p):
        d = pd.read_sas(p, format="xport")
        d = d[[c for c in ("SEQN", "SDMVPSU", "SDMVSTRA", "WTMEC2YR") if c in d.columns]].copy(); d["year"] = yr
        ds.append(d)
test = test.merge(pd.concat(ds, ignore_index=True), on=["SEQN", "year"], how="left")
w = test["WTMEC2YR"].fillna(test["wt"]).values.astype(float) if "WTMEC2YR" in test.columns else test["wt"].values.astype(float)

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def calib(y, p, ww=None, nd=10):
    ww = np.ones(len(y)) if ww is None else np.asarray(ww, float)
    q = np.quantile(p, np.linspace(0, 1, nd + 1)); q[0], q[-1] = -np.inf, np.inf
    mp, ob, wv = [], [], []
    for i in range(nd):
        m = (p > q[i]) & (p <= q[i + 1])
        if m.sum() == 0: continue
        mp.append(np.average(p[m], weights=ww[m])); ob.append(np.average(y[m], weights=ww[m])); wv.append(ww[m].sum())
    mp, ob, wv = np.array(mp), np.array(ob), np.array(wv)
    A = np.vstack([mp, np.ones_like(mp)]).T; W = np.diag(wv)
    c = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ ob, rcond=None)[0]
    return float(c[0]), float(c[1]), float(np.sum(wv * np.abs(mp - ob)) / np.sum(wv))

def ops(y, p, thr, ww=None):
    ww = np.ones(len(y)) if ww is None else np.asarray(ww, float)
    pred = p >= thr
    tp = ww[(pred) & (y == 1)].sum(); fp = ww[(pred) & (y == 0)].sum()
    fn_ = ww[(~pred) & (y == 1)].sum(); tn = ww[(~pred) & (y == 0)].sum()
    return {"sens": round(float(tp / max(tp + fn_, 1e-9)), 4), "spec": round(float(tn / max(tn + fp, 1e-9)), 4),
            "ppv": round(float(tp / max(tp + fp, 1e-9)), 4), "npv": round(float(tn / max(tn + fn_, 1e-9)), 4),
            "flagged_per1000": round(float(1000 * (tp + fp) / ww.sum()), 1), "tp_per1000": round(float(1000 * tp / ww.sum()), 1),
            "fp_per1000": round(float(1000 * fp / ww.sum()), 1)}

def nb(y, p, thr, ww=None):
    ww = np.ones(len(y)) if ww is None else np.asarray(ww, float); pred = p >= thr
    tp = ww[pred & (y == 1)].sum(); fp = ww[pred & (y == 0)].sum(); N = ww.sum()
    return float(tp / N - (fp / N) * (thr / (1 - thr)))

out = {}

# ---------- P1-05 重校准实证（验证集内 5 折 CV，不自欺） ----------
print("P1-05 重校准实证 ...", flush=True)
def recalib_cv(p, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); pr = np.zeros(len(y))
    for tr, va in skf.split(p, y):
        m = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p[tr]).reshape(-1, 1), y[tr])
        pr[va] = m.predict_proba(logit(p[va]).reshape(-1, 1))[:, 1]
    return pr
XL = logit(p_x)
out["recalibration"] = {"before": {"slope": round(calib(yte, p_x)[0], 3), "intercept": round(calib(yte, p_x)[1], 4), "ece": round(calib(yte, p_x)[2], 4)},
                        "after_cv": {"slope": round(calib(yte, recalib_cv(p_x, yte))[0], 3), "intercept": round(calib(yte, recalib_cv(p_x, yte))[1], 4),
                                     "ece": round(calib(yte, recalib_cv(p_x, yte))[2], 4), "auc_unchanged": round(roc_auc_score(yte, recalib_cv(p_x, yte)), 4)}}
# MLP（本稿唯一"需重校准"的模型）：按稿件规格复现并自检
try:
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=500, random_state=42).fit(Xtr, ytr)
    p_m = mlp.predict_proba(Xte)[:, 1]; a_m = roc_auc_score(yte, p_m)
    print(f"   MLP 复现 AUC={a_m:.4f}（目标 0.7695）", flush=True)
    if abs(a_m - 0.7695) < 0.005:
        pmr = recalib_cv(p_m, yte)
        out["recalibration"]["mlp"] = {"auc": round(a_m, 4), "before_slope": round(calib(yte, p_m)[0], 3), "before_ece": round(calib(yte, p_m)[2], 4),
                                       "after_cv_slope": round(calib(yte, pmr)[0], 3), "after_cv_ece": round(calib(yte, pmr)[2], 4)}
    else:
        out["recalibration"]["mlp"] = {"note": "MLP 复现 AUC 偏离目标 >0.005，按诚实原则不报告其重校准结果", "reproduced_auc": round(a_m, 4)}
except Exception as e:
    out["recalibration"]["mlp"] = {"note": f"MLP 复现失败: {e}"}
print("   ", out["recalibration"], flush=True)

# ---------- P1-06 BH-FDR（10 亚组 ΔAUC 的 bootstrap P） ----------
print("P1-06 BH-FDR ...", flush=True)
sd = json.load(open(os.path.join(RES, "subgroup_dca_incremental.json"), encoding="utf-8"))
items = [(k, v["delta_auc_ci"]["p_boot"], v["delta_auc_ci"]["delta_auc"]) for k, v in sd["subgroups"].items()]
ps = np.array([max(x[1], 1e-6) for x in items]); order = np.argsort(ps); m = len(ps)
adj = np.empty(m)
for i, o in enumerate(order):
    adj[o] = min(1.0, ps[o] * m / (i + 1))
out["fdr"] = {"method": "Benjamini-Hochberg on 10 subgroup bootstrap P values (one-sided P(delta<=0))",
              "items": [{"subgroup": items[i][0], "delta_auc": items[i][2], "p_raw": round(float(ps[i]), 4), "p_bh": round(float(adj[i]), 4),
                         "significant_0.05_after_bh": bool(adj[i] < 0.05)} for i in order]}
out["fdr"]["n_significant_raw"] = int((ps < 0.05).sum()); out["fdr"]["n_significant_bh"] = int((adj < 0.05).sum())
print("   原始显著", out["fdr"]["n_significant_raw"], "| BH 后显著", out["fdr"]["n_significant_bh"], flush=True)

# ---------- P1-07 标签不确定性量化（概率偏差分析） ----------
print("P1-07 概率偏差分析 ...", flush=True)
uacr = test["uacr"].values; egfr = test["egfr"].values
pos = uacr >= 30; near = pos & (uacr < 50)
frac_near = float(near.sum() / max(pos.sum(), 1))
def scenario(reclass_mask, tag):
    y2 = yte.copy(); y2[reclass_mask & (yte == 1)] = 0          # 视为非持续 -> 改判阴性
    o = ops(y2, p_x, YTHR); a2 = roc_auc_score(y2, p_x) if len(np.unique(y2)) > 1 else float("nan")
    return {"scenario": tag, "reclassified_n": int((reclass_mask & (yte == 1)).sum()), "auc": round(float(a2), 4), "ppv": o["ppv"], "npv": o["npv"],
            "tp_per1000": o["tp_per1000"], "fp_per1000": o["fp_per1000"], "nns_per_case": round(float(o["flagged_per1000"] / max(o["tp_per1000"], 1e-9)), 1)}
rng = np.random.default_rng(RNG); half = np.zeros(len(yte), bool)
idxn = np.where(near)[0]; half[rng.choice(idxn, int(round(len(idxn) / 2)), replace=False)] = True
out["bias_analysis"] = {"basis": "uACR-positive participants at 30-49 mg/g (nearest the KDIGO threshold) treated as non-persistent in scenario analyses",
                        "uacr_positive_n": int(pos.sum()), "near_threshold_n": int(near.sum()), "near_threshold_fraction": round(frac_near, 4),
                        "base": {"auc": round(float(roc_auc_score(yte, p_x)), 4), **ops(yte, p_x, YTHR)},
                        "scenarios": [scenario(near, "all near-threshold uACR positives treated as transient (upper-bound)"),
                                      scenario(half, "half of near-threshold uACR positives treated as transient")]}
print("   近阈占比", round(frac_near, 4), "| 情景 AUC:", [s["auc"] for s in out["bias_analysis"]["scenarios"]], flush=True)

# ---------- P1-08 临床影响量化（每 1,000 人） ----------
print("P1-08 临床影响 ...", flush=True)
pain = {"screened_per_1000": 1000}
o_base = ops(yte, p_x, YTHR)
pain["model_triage"] = o_base
det = (yte == 1) & (p_x >= YTHR)
pain["detected_with_albuminuria_preserved_egfr_per1000"] = round(float(1000 * ((det) & (uacr >= 30) & (egfr >= 60)).sum() / len(yte)), 1)
elig = (uacr >= 200) | ((test["diabetes"].values == 1) & (uacr >= 30))
pain["sglt2i_eligibility_proxy_per1000"] = round(float(1000 * ((det) & elig & (egfr >= 20)).sum() / len(yte)), 1)
pain["sglt2i_proxy_definition"] = "uACR >=200 mg/g, or diabetes with uACR >=30 mg/g, both with eGFR >=20 mL/min/1.73 m2 (proxy, not a formal eligibility assessment)"
age = test["age"].values; m60 = age >= 60
pain["screen_all_age60plus"] = ops(yte[m60], np.ones(m60.sum()), 0.5)
pain["screen_all_age60plus"]["n_per1000"] = int(round(1000 * m60.mean()))
pain["efficiency_ratio_flagged_per_detected_case"] = {
    "model": round(float(o_base["flagged_per1000"] / max(o_base["tp_per1000"], 1e-9)), 1),
    "screen_all_age60plus": round(float(pain["screen_all_age60plus"]["flagged_per1000"] / max(pain["screen_all_age60plus"]["tp_per1000"], 1e-9)), 1)}
print("   模型每 1,000 人：检出", o_base["tp_per1000"], "假阳", o_base["fp_per1000"], "| 干预窗", pain["detected_with_albuminuria_preserved_egfr_per1000"], flush=True)

# ---------- P1-09 1,233 例不可评估者画像 ----------
print("P1-09 1,233 画像 ...", flush=True)
both = cohort["egfr"].isna() & cohort["uacr"].isna()
print(f"   两标志均缺: {int(both.sum())}（台账 1,233）", flush=True)
if int(both.sum()) != 1233:
    sys.exit(f"❌ 自检失败：两标志均缺 {int(both.sum())} != 1233")
def prof(mask):
    d = cohort[mask]
    return {"n": int(mask.sum()), "age_mean": round(float(d["age"].mean()), 1), "female_pct": round(float(100 * (d["sex"] == 2).mean()), 1),
            "diabetes_pct": round(float(100 * (d["diabetes"] == 1).mean()), 1), "bmi_mean": round(float(d["bmi"].mean(skipna=True)), 1),
            "sbp_mean": round(float(d["sbp"].mean(skipna=True)), 1), "poverty_lt1.3_pct": round(float(100 * (d["poverty_ratio"] < 1.3).mean()), 1)}
out["non_assessable_profile"] = {"cohort": prof(pd.Series(True, index=cohort.index)), "non_assessable": prof(both), "assessable": prof(~both),
                                 "note": "Participants without an assessable eGFR or uACR cannot have CKD status determined; they were retained as CKD-negative in the primary analysis."}
print("   ", out["non_assessable_profile"]["non_assessable"], flush=True)

# ---------- P1-10 加权校准与 DCA ----------
print("P1-10 加权校准/DCA ...", flush=True)
ws, wi, we = calib(yte, p_x, w)
THS = [round(0.05 + 0.01 * i, 2) for i in range(21)]
wd = {}
for t in (0.10, 0.20):
    wd[f"{t:.2f}"] = {"nb_model": round(nb(yte, p_x, t, w), 4), "nb_screen_all": round((np.average(yte, weights=w)) - (1 - np.average(yte, weights=w)) * (t / (1 - t)), 4),
                      "delta_nb": round(nb(yte, p_x, t, w) - ((np.average(yte, weights=w)) - (1 - np.average(yte, weights=w)) * (t / (1 - t))), 4)}
win = [t for t in THS if (nb(yte, p_x, t, w) - ((np.average(yte, weights=w)) - (1 - np.average(yte, weights=w)) * (t / (1 - t)))) > 0]
out["weighted_calibration_dca"] = {"slope": round(ws, 3), "intercept": round(wi, 4), "ece": round(we, 4), "dca": wd,
                                   "window_model_gt_screen_all": [min(win), max(win)] if win else None,
                                   "unweighted_reference": {"slope": round(calib(yte, p_x)[0], 3), "ece": round(calib(yte, p_x)[2], 4)}}
print("   加权 slope", round(ws, 3), "| 加权 DCA@20%", wd["0.20"], "| 窗口", out["weighted_calibration_dca"]["window_model_gt_screen_all"], flush=True)

# ---------- P1-11 亚组 PPV/NPV ----------
print("P1-11 亚组 PPV/NPV ...", flush=True)
te = test
SUBS = [("Age 18–44 y", (te["age"] < 45).values), ("Age 45–64 y", ((te["age"] >= 45) & (te["age"] < 65)).values),
        ("Age ≥65 y", (te["age"] >= 65).values), ("Male", (te["sex"] == 1).values), ("Female", (te["sex"] == 2).values),
        ("White", (te["race"] == 3).values), ("Black", (te["race"] == 4).values), ("Hispanic", (te["race"].isin([1, 2])).values),
        ("Poverty-income ratio <1.3", (te["poverty_ratio"] < 1.3).values), ("Poverty-income ratio ≥1.3", (te["poverty_ratio"] >= 1.3).values)]
out["subgroup_ppv"] = {"threshold": YTHR, "items": []}
for lab, m in SUBS:
    o = ops(yte[m], p_x[m], YTHR)
    out["subgroup_ppv"]["items"].append({"subgroup": lab, "n": int(m.sum()), "events": int(yte[m].sum()),
                                         "ppv": o["ppv"], "npv": o["npv"], "sens": o["sens"], "spec": o["spec"], "flagged_per1000": o["flagged_per1000"]})
    print(f"   {lab:<28} PPV={o['ppv']:.3f} NPV={o['npv']:.3f} 每千标记={o['flagged_per1000']}", flush=True)

# ---------- P1-12 分组阈值与等赔率演示 ----------
print("P1-12 分组阈值/等赔率 ...", flush=True)
bands = [("18–44", 18, 45), ("45–64", 45, 65), ("65+", 65, 200)]
dev_age = train["age"].values
band_thr = {}
for lab, lo, hi in bands:
    m = (dev_age >= lo) & (dev_age < hi)
    y_d, p_d = ytr[m], p_x_tr[m]
    best, bt = -1, YTHR
    for t in np.arange(0.05, 0.51, 0.005):
        pred = p_d >= t
        if pred.sum() == 0: continue
        tp = ((pred) & (y_d == 1)).sum(); fp = ((pred) & (y_d == 0)).sum(); fn_ = ((~pred) & (y_d == 1)).sum(); tn = ((~pred) & (y_d == 0)).sum()
        j = tp / max(tp + fn_, 1e-9) + tn / max(tn + fp, 1e-9) - 1
        if j > best: best, bt = j, float(t)
    band_thr[lab] = round(bt, 3)
glob_dev = YTHR
glob_sens_dev = ops(ytr, p_x_tr, glob_dev)["sens"]
eq_thr = {}
for lab, lo, hi in bands:
    m = (dev_age >= lo) & (dev_age < hi)
    y_d, p_d = ytr[m], p_x_tr[m]
    best, bt = 1e9, band_thr[lab]
    for t in np.arange(0.02, 0.61, 0.002):
        s = ops(y_d, p_d, float(t))["sens"]
        if abs(s - glob_sens_dev) < best: best, bt = abs(s - glob_sens_dev), float(t)
    eq_thr[lab] = round(bt, 3)
def eval_bands(thr_map, tag):
    pred = np.zeros(len(yte), bool)
    per = []
    for lab, lo, hi in bands:
        m = (test["age"].values >= lo) & (test["age"].values < hi)
        t = thr_map[lab] if isinstance(thr_map, dict) else thr_map
        pred[m] = p_x[m] >= t
        o = ops(yte[m], p_x[m], t); per.append({"band": lab, **o})
    gsens = float(((pred) & (yte == 1)).sum() / max((yte == 1).sum(), 1)); gspec = float(((~pred) & (yte == 0)).sum() / max((yte == 0).sum(), 1))
    return {"strategy": tag, "per_band": per, "global_sens": round(gsens, 4), "global_spec": round(gspec, 4),
            "global_flagged_per1000": round(float(1000 * pred.mean()), 1),
            "nb_at_10pct": round(float(nb(yte, pred.astype(float), 0.10)), 4), "nb_at_20pct": round(float(nb(yte, pred.astype(float), 0.20)), 4)}
out["fairness_thresholds"] = {"development_derived_thresholds": band_thr, "global_threshold": glob_dev,
                              "equalized_sensitivity_thresholds": eq_thr,
                              "evaluation": [eval_bands(glob_dev, "single global threshold"),
                                             eval_bands(band_thr, "age-band-specific Youden thresholds (development-derived)"),
                                             eval_bands(eq_thr, "age-band thresholds equalising development-set sensitivity")]}
print("   分组阈值", band_thr, "| 等敏感度阈值", eq_thr, flush=True)

# ---------- P1-17 交叉性 ----------
print("P1-17 交叉性 ...", flush=True)
cells = [("Age 18–44 y & PIR <1.3", (te["age"] < 45) & (te["poverty_ratio"] < 1.3)), ("Age 18–44 y & PIR ≥1.3", (te["age"] < 45) & (te["poverty_ratio"] >= 1.3)),
         ("Age ≥45 y & PIR <1.3", (te["age"] >= 45) & (te["poverty_ratio"] < 1.3)), ("Age ≥45 y & PIR ≥1.3", (te["age"] >= 45) & (te["poverty_ratio"] >= 1.3))]
rng = np.random.default_rng(RNG); out["intersectional"] = {"items": []}
for lab, m in cells:
    y = yte[m.values]; p = p_x[m.values]
    a = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ci = None
    if len(np.unique(y)) > 1:
        bs = []
        for _ in range(1000):
            i = rng.choice(np.arange(len(y)), len(y), replace=True)
            if len(np.unique(y[i])) > 1: bs.append(roc_auc_score(y[i], p[i]))
        ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    out["intersectional"]["items"].append({"cell": lab, "n": int(m.sum()), "events": int(y.sum()), "auc": round(float(a), 4), "ci95": ci})
    print(f"   {lab:<26} n={int(m.sum()):<5} 事件={int(y.sum()):<4} AUC={a:.4f} CI={ci}", flush=True)

json.dump({"note": "P1 clinical/equity addendum (expert-panel items A-phase)", "results": out},
          open(os.path.join(RES, "p1_clinical_equity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("✅ p1_clinical_equity.json 已写（用时 %.0fs）" % (time.time() - t0), flush=True)
