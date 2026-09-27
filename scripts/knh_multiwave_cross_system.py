# -*- coding: utf-8 -*-
"""knh_multiwave_cross_system.py — 三波（2022/2023/2024）跨体系外部验证 + 波次一致性。

沿用 knh2022_cross_system.py 的冻结流程与预指定 race-blind 口径，仅把 KNHANES 侧扩为三波。
"""
import os, sys, json, zipfile, glob, numpy as np, pandas as pd, pyreadstat
sys.path.insert(0, r"scripts")
import knh2022_cross_system as M
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

BASE = os.path.join(M.PROJ, r"data\knh")
OUT = os.path.join(M.PROJ, r"results\knh_multiwave_cross_system.json")
YEARS = ("2022", "2023", "2024")
RB = ["age", "sex", "poverty_ratio", "education", "bmi", "sbp", "dbp",
      "hba1c", "diabetes", "total_cholesterol", "hdl"]


def load_year(yr):
    d = os.path.join(BASE, yr)
    for z in glob.glob(os.path.join(d, "*.zip")):
        with zipfile.ZipFile(z) as zz:
            for i in zz.infolist():
                if i.filename.lower().endswith(".sas7bdat"):
                    zz.extract(i.filename, d)
    f = sorted(glob.glob(os.path.join(d, "**", "*.sas7bdat"), recursive=True))[-1]
    df, meta = pyreadstat.read_sas7bdat(f)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    x = pd.DataFrame({
        "year": yr, "age": num("age"), "sex": num("sex"), "education_raw": num("edu"),
        "incm5": num("incm5"), "bmi": num("HE_BMI"), "sbp": num("HE_sbp"), "dbp": num("HE_dbp"),
        "hba1c": num("HE_HbA1c"), "total_cholesterol": num("HE_chol"), "hdl": num("HE_HDL_st2"),
        "creatinine": num("HE_crea"), "ualb": num("HE_Ualb"), "ucrea": num("HE_Ucrea"),
        "de1_dg": num("DE1_dg")})
    x["diabetes"] = np.where(x["de1_dg"] == 1, 1.0, np.where(x["de1_dg"] == 0, 0.0, np.nan))
    x["uacr"] = (x["ualb"] * 100.0 / x["ucrea"]).where(x["ualb"].notna() & x["ucrea"].notna() & (x["ucrea"] > 0))
    k = np.where(x["sex"] == 2, 0.7, 0.9); al = np.where(x["sex"] == 2, -0.241, -0.302)
    s = x["creatinine"] / k
    x["egfr"] = (142.0 * np.minimum(s, 1.0) ** al * np.maximum(s, 1.0) ** (-1.200)
                 * 0.9938 ** x["age"] * np.where(x["sex"] == 2, 1.012, 1.0))
    x["ckd"] = np.where(x["egfr"].notna() | x["uacr"].notna(),
                        ((x["egfr"] < 60) | (x["uacr"] >= 30)).astype(float), np.nan)
    return x, os.path.basename(f)


def boot_auc(y, p, n=1000, seed=20260927):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if 0 < y[s].sum() < len(s):
            out.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def ece(p, y, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    o = pd.Series(y).groupby(q, observed=True).mean().values
    e = pd.Series(p).groupby(q, observed=True).mean().values
    w = pd.Series(y).groupby(q, observed=True).size().values / len(y)
    return float(np.sum(w * np.abs(o - e)))


def dca(y, p, t):
    f = p >= t
    tp = int(((f) & (y == 1)).sum()); fp = int(((f) & (y == 0)).sum())
    return {"model_nb": float(tp / len(y) - (fp / len(y)) * (t / (1 - t))),
            "treat_all_nb": float(y.mean() - (1 - y.mean()) * (t / (1 - t)))}


def main():
    # 1) NHANES 自检门 + 冻结 race-blind 模型
    dev = pd.concat([M.load_nhanes(c) for c in ("G", "H", "I")], ignore_index=True)
    val = M.load_nhanes("J")
    def drb(X):
        X = X.copy(); X["female"] = (X["sex"] == 2).astype(float); return X.drop(columns=["sex"])
    # --- 双自检门：含 race 必须=0.8099；race-blind 必须=0.8065（本课题预指定跨体系口径的冻结值）---
    FE = RB + ["race"]  # RB 已含全部 12 特征（无 race），此处补上 race 以复现含 race 的冻结值
    def dfull(X):
        X = X.copy()
        X["race_black"] = (X["race"] == 4).astype(float)
        X["race_hisp"] = X["race"].isin([1, 2]).astype(float)
        X["race_other"] = (~X["race"].isin([1, 2, 3, 4])).astype(float)
        X["female"] = (X["sex"] == 2).astype(float)
        return X.drop(columns=["race", "sex"])
    d_f, v_f = dfull(dev[FE]), dfull(val[FE])
    med_f = d_f.median()
    Ff = lambda X: pd.concat([X.fillna(med_f), X.isna().add_prefix("miss_")], axis=1)
    m_f = XGBClassifier(**M.HYP).fit(Ff(d_f), dev["ckd"].values)
    auc_full = roc_auc_score(val["ckd"].values, m_f.predict_proba(Ff(v_f))[:, 1])
    d_rb, v_rb = drb(dev[RB]), drb(val[RB])
    med = d_rb.median()
    F = lambda X: pd.concat([X.fillna(med), X.isna().add_prefix("miss_")], axis=1)
    m = XGBClassifier(**M.HYP).fit(F(d_rb), dev["ckd"].values)
    auc_ck = roc_auc_score(val["ckd"].values, m.predict_proba(F(v_rb))[:, 1])
    print("自检门①（含 race）AUC=%.4f %s | 自检门②（race-blind）AUC=%.4f %s"
          % (auc_full, "✅" if abs(auc_full - 0.8099) <= 1e-4 else "❌",
             auc_ck, "✅" if abs(auc_ck - 0.8065) <= 1e-4 else "❌"))
    if abs(auc_full - 0.8099) > 1e-4 or abs(auc_ck - 0.8065) > 1e-4:
        sys.exit("❌ 自检门未过，中止")

    # 2) 三波 KNHANES + 协调化
    pir = dev["poverty_ratio"].dropna(); qs = pir.quantile([0, .2, .4, .6, .8, 1.0]).values
    means = [pir[(pir >= qs[i]) & (pir <= qs[i + 1])].mean() for i in range(5)]
    frames = []
    per_wave = {}
    for yr in YEARS:
        x, fn = load_year(yr)
        raw_n = len(x)
        x["poverty_ratio"] = x["incm5"].map({i + 1: means[i] for i in range(5)})
        x["education"] = 1.0 + (x["education_raw"] - 1.0) * (4.0 / 3.0)
        x = x[(x["age"] >= 18) & x["sex"].isin([1, 2]) & x["ckd"].notna()].copy()
        p = m.predict_proba(F(drb(x[RB])))[:, 1]
        per_wave[yr] = {"file": fn, "n_raw": int(raw_n), "n": int(len(x)), "n_ckd": int(x.ckd.sum()),
                        "ckd_pct": round(100 * float(x.ckd.mean()), 2),
                        "uacr_median_mg_g": round(float(np.nanmedian(x["uacr"])), 3),
                        "auc": round(float(roc_auc_score(x.ckd.values, p)), 4)}
        print("  %s: n=%d CKD=%d (%.2f%%) uACR中位=%.3f AUC=%.4f"
              % (yr, len(x), int(x.ckd.sum()), 100 * x.ckd.mean(),
                 np.nanmedian(x["uacr"]), per_wave[yr]["auc"]))
        x["_p"] = p
        frames.append(x)
    kd = pd.concat(frames, ignore_index=True)
    y = kd["ckd"].values; p = kd["_p"].values
    med_u = float(np.nanmedian(kd["uacr"]))
    print("\n合并队列 n=%d CKD=%d (%.2f%%) uACR中位=%.3f（门 2–20）%s"
          % (len(kd), int(kd.ckd.sum()), 100 * kd.ckd.mean(), med_u, "✅" if 2 <= med_u <= 20 else "❌"))
    if not (2 <= med_u <= 20):
        sys.exit("❌ 单位门未过")
    auc = roc_auc_score(y, p); lo, hi = boot_auc(y, p)
    print("跨体系 AUC = %.4f (95%% CI %.4f–%.4f)" % (auc, lo, hi))

    eps = 1e-9
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    b1, b0 = np.polyfit(lp, y, 1)
    g = lambda a: float(np.mean(1 / (1 + np.exp(-(a + lp))))) - float(y.mean())
    a_hat = brentq(g, -20, 20)
    p_i = 1 / (1 + np.exp(-(a_hat + lp)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    p_s = lr.predict_proba(lp.reshape(-1, 1))[:, 1]
    print("校准 slope=%.3f intercept=%.3f ECE=%.4f | 重校准 ECE: 截距 %.4f / 斜率+截距 %.4f"
          % (b1, b0, ece(p, y), ece(p_i, y), ece(p_s, y)))

    fl = p >= M.THR
    tp = int(((fl) & (y == 1)).sum()); fp = int(((fl) & (y == 0)).sum())
    fn = int(((~fl) & (y == 1)).sum()); tn = int(((~fl) & (y == 0)).sum())
    metrics = {"sensitivity": round(tp / (tp + fn), 4), "specificity": round(tn / (tn + fp), 4),
               "ppv": round(tp / max(tp + fp, 1), 4), "flagged_per_1000": round(1000 * float(fl.mean()), 1),
               "true_cases_per_1000": round(1000 * float(y.mean()), 1)}
    print("阈值 %.2f：%s" % (M.THR, metrics))
    dc = {"%.2f" % t: dca(y, p, t) for t in (0.05, 0.10, 0.15, 0.20, 0.25)}
    print("DCA: " + " | ".join("%s %.4f/%.4f" % (k, v["model_nb"], v["treat_all_nb"]) for k, v in dc.items()))

    res = {
        "design": "frozen NHANES 2011-2016 XGBoost, race-blind variant (pre-specified)",
        "self_check_nhanes_2017_2018_auc": round(float(auc_ck), 4),
        "harmonization": {"income": "incm5 quintile -> NHANES dev PIR quintile means",
                          "income_quintile_means": [round(float(v), 4) for v in means],
                          "education": "edu 1-4 linearly rescaled to DMDEDUC2 1-5"},
        "per_wave": per_wave,
        "pooled": {"n": int(len(kd)), "n_ckd": int(kd.ckd.sum()), "ckd_pct": round(100 * float(y.mean()), 2),
                   "uacr_median_mg_g": round(med_u, 3),
                   "auc": {"value": round(float(auc), 4), "ci": [round(lo, 4), round(hi, 4)]},
                   "calibration": {"slope": round(float(b1), 4), "intercept": round(float(b0), 4),
                                   "ece": round(float(ece(p, y)), 4)},
                   "at_frozen_threshold_0.16": metrics, "dca": dc},
        "recalibration_sensitivity": {"note": "calibration-only simulation; NOT a validation result",
                                      "ece_raw": round(float(ece(p, y)), 4),
                                      "ece_intercept_only": round(float(ece(p_i, y)), 4),
                                      "intercept_correction_a": round(float(a_hat), 4),
                                      "ece_slope_intercept": round(float(ece(p_s, y)), 4),
                                      "slope_intercept_fit": [round(float(lr.coef_[0][0]), 4), round(float(lr.intercept_[0]), 4)]},
        "mean_predicted_risk": round(float(p.mean()), 4),
        "nhanes_reference": {"dev_n": int(len(dev)), "dev_ckd_pct": round(100 * float(dev.ckd.mean()), 2),
                             "val_n": int(len(val)), "val_ckd_pct": round(100 * float(val.ckd.mean()), 2)},
    }
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n✅ 写出", OUT)


if __name__ == "__main__":
    main()
