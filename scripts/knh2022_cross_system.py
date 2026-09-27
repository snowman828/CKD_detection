# -*- coding: utf-8 -*-
"""knh2022_cross_system.py — 阶段 C：KNHANES 2022 跨医疗体系外部验证（冻结模型 + race-blind 预指定口径）

设计（全部显式、可复核）：
  1) 自检门：用 NHANES 2011–2016 开发集重训 XGBoost → 必须在 2017–2018 复现 AUC=0.8099（±1e-4），否则中止。
  2) 预指定口径：跨体系应用走 **race-blind** 变体（去 race 三个指示列及其缺失指示列 → 22 输入）。
  3) 协调化（各国量表不同，全部显式声明）：
       - education：KNHANES edu 1–4 → 单调线性映射到 NHANES DMDEDUC2 的 1–5 尺度
       - income   ：KNHANES incm5（1–5 五分位）→ NHANES 开发集 PIR 的**五分位均值**（等百分位映射）
  4) 单位门：uACR 中位数须落在 2–20 mg/g（KNHANES 尿白蛋白 mg/L、尿肌酐 mg/dL）
  5) 输出：AUC + bootstrap CI、校准（斜率/截距/ECE）、冻结阈值 0.16 下的敏感性/特异性/PPV、DCA(5–25%)
  6) 敏感性：income 用队列中位数（映射不敏感检查）、education 不重标（原始 1–4）

列名依据：2022 官方文件 hn22_all.sas7bdat 的**实测列名与官方标签**（DI1_dg=고혈압❌不用于糖尿病；DE1_dg=당뇨병✅）
"""
import os, sys, json, numpy as np, pandas as pd, pyreadstat
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

PROJ = r"G:\项目文件\CKD多模态AI早诊_本项目专属归档"
KNH = os.path.join(PROJ, r"09_无湿实验执行\data\knh\2022")
DATA = os.path.join(PROJ, r"09_无湿实验执行\data")
OUT = os.path.join(PROJ, r"09_无湿实验执行\results\knh2022_cross_system.json")
HYP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
THR = 0.16  # 冻结的部署阈值（开发集 Youden）


def load_nhanes(cyc):
    L = lambda n: pd.read_sas(os.path.join(DATA, f"{n}_{cyc}.XPT"), format="xport")
    dem, alb, bio = L("DEMO"), L("ALB_CR"), L("BIOPRO")
    bpx, bmx, diq, ghb, tc, hd = L("BPX"), L("BMX"), L("DIQ"), L("GHB"), L("TCHOL"), L("HDL")
    x = dem[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "RIDEXPRG"]].copy()
    x = x.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH1": "race",
                          "INDFMPIR": "poverty_ratio", "DMDEDUC2": "education", "RIDEXPRG": "pregnant"})
    M = lambda s, c, p: x.merge(s[["SEQN"] + c].rename(columns={v: p + v for v in c}), on="SEQN", how="left")
    x = M(alb, ["URXUMA", "URXUCR"], "a_"); x = M(bio, ["LBXSCR"], "b_")
    x = M(bpx, ["BPXSY1", "BPXDI1"], "c_"); x = M(bmx, ["BMXBMI"], "d_")
    x = M(diq, ["DIQ010"], "e_"); x = M(ghb, ["LBXGH"], "f_")
    x = M(tc, ["LBXTC"], "g_"); x = M(hd, ["LBDHDD"], "g_")
    x["uacr"] = (x["a_URXUMA"] * 100.0 / x["a_URXUCR"]).where(x["a_URXUCR"] > 0)
    x["creatinine"] = x["b_LBXSCR"]; x["sbp"] = x["c_BPXSY1"]; x["dbp"] = x["c_BPXDI1"]
    x["bmi"] = x["d_BMXBMI"]; x["hba1c"] = x["f_LBXGH"]
    x["total_cholesterol"] = x["g_LBXTC"]; x["hdl"] = x["g_LBDHDD"]
    x["diabetes"] = np.where(x["e_DIQ010"] == 1, 1.0, np.where(x["e_DIQ010"] == 2, 0.0, np.nan))
    k = np.where(x["sex"] == 2, 0.7, 0.9); al = np.where(x["sex"] == 2, -0.241, -0.302)
    s = x["creatinine"] / k
    x["egfr"] = (142.0 * np.minimum(s, 1.0) ** al * np.maximum(s, 1.0) ** (-1.200)
                 * 0.9938 ** x["age"] * np.where(x["sex"] == 2, 1.012, 1.0))
    x["ckd"] = ((x["egfr"] < 60) | (x["uacr"] >= 30)).astype(float)
    return x[(x["age"] >= 18) & (x["pregnant"] != 1) & x["ckd"].notna()]


def load_knh():
    z = os.path.join(KNH, "HN22_ALL(SAS).zip")
    import zipfile
    with zipfile.ZipFile(z) as zz:
        zz.extract("hn22_all.sas7bdat", KNH)
    df, meta = pyreadstat.read_sas7bdat(os.path.join(KNH, "hn22_all.sas7bdat"))
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    d = pd.DataFrame({
        "age": num("age"), "sex": num("sex"),
        "education_raw": num("edu"), "incm5": num("incm5"),
        "bmi": num("HE_BMI"), "sbp": num("HE_sbp"), "dbp": num("HE_dbp"),
        "hba1c": num("HE_HbA1c"), "total_cholesterol": num("HE_chol"), "hdl": num("HE_HDL_st2"),
        "creatinine": num("HE_crea"), "ualb": num("HE_Ualb"), "ucrea": num("HE_Ucrea"),
        "de1_dg": num("DE1_dg"), "di1_dg_hypertension": num("DI1_dg"),
    })
    d["diabetes"] = np.where(d["de1_dg"] == 1, 1.0, np.where(d["de1_dg"] == 0, 0.0, np.nan))
    d["uacr"] = (d["ualb"] * 100.0 / d["ucrea"]).where(d["ualb"].notna() & d["ucrea"].notna() & (d["ucrea"] > 0))
    k = np.where(d["sex"] == 2, 0.7, 0.9); al = np.where(d["sex"] == 2, -0.241, -0.302)
    s = d["creatinine"] / k
    d["egfr"] = (142.0 * np.minimum(s, 1.0) ** al * np.maximum(s, 1.0) ** (-1.200)
                 * 0.9938 ** d["age"] * np.where(d["sex"] == 2, 1.012, 1.0))
    d["ckd"] = np.where(d["egfr"].notna() | d["uacr"].notna(),
                        ((d["egfr"] < 60) | (d["uacr"] >= 30)).astype(float), np.nan)
    return d


def boot_auc(y, p, n=1000, seed=20260927):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if 0 < y[s].sum() < len(s):
            out.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    print("== 1) NHANES 自检门 ==")
    dev = pd.concat([load_nhanes(c) for c in ("G", "H", "I")], ignore_index=True)
    val = load_nhanes("J")
    print("   dev n=%d CKD=%d | val n=%d CKD=%d" % (len(dev), int(dev.ckd.sum()), len(val), int(val.ckd.sum())))

    FE = ["age", "sex", "poverty_ratio", "education", "bmi", "sbp", "dbp",
          "hba1c", "diabetes", "total_cholesterol", "hdl"]

    def design_full(X):  # 含 race（用于自检门，必须复现 0.8099）
        X = X.copy()
        X["race_black"] = (X["race"] == 4).astype(float)
        X["race_hisp"] = X["race"].isin([1, 2]).astype(float)
        X["race_other"] = (~X["race"].isin([1, 2, 3, 4])).astype(float)
        X["female"] = (X["sex"] == 2).astype(float)
        return X.drop(columns=["race", "sex"])

    d_full = design_full(dev[FE + ["race"]]); v_full = design_full(val[FE + ["race"]])
    med_full = d_full.median()
    F = lambda X, m: pd.concat([X.fillna(m), X.isna().add_prefix("miss_")], axis=1)
    m_full = XGBClassifier(**HYP).fit(F(d_full, med_full), dev["ckd"].values)
    auc_ck = roc_auc_score(val["ckd"].values, m_full.predict_proba(F(v_full, med_full))[:, 1])
    print("   自检门 NHANES 2017–2018 AUC=%.4f（须=0.8099）%s" % (auc_ck, "✅" if abs(auc_ck - 0.8099) <= 1e-4 else "❌"))
    if abs(auc_ck - 0.8099) > 1e-4:
        sys.exit("❌ 自检门未通过 → 中止（不产出跨体系结果）")

    print("== 2) 冻结 race-blind 模型（预指定口径）==")
    RB = ["age", "sex", "poverty_ratio", "education", "bmi", "sbp", "dbp",
          "hba1c", "diabetes", "total_cholesterol", "hdl"]

    def design_rb(X):
        X = X.copy(); X["female"] = (X["sex"] == 2).astype(float); return X.drop(columns=["sex"])

    d_rb = design_rb(dev[RB]); v_rb = design_rb(val[RB])
    med_rb = d_rb.median()
    m_rb = XGBClassifier(**HYP).fit(F(d_rb, med_rb), dev["ckd"].values)
    p_val = m_rb.predict_proba(F(v_rb, med_rb))[:, 1]
    auc_rb_val = roc_auc_score(val["ckd"].values, p_val)
    print("   race-blind 在 NHANES 2017–2018 AUC=%.4f（与含 race 的 0.8099 对比）" % auc_rb_val)

    print("== 3) KNHANES 协调化 ==")
    kd = load_knh()
    print("   原始 n=%d（有效标签 %d）" % (len(kd), int(kd.ckd.notna().sum())))
    # income：incm5(1–5) → NHANES 开发集 PIR 的五分位均值
    pir = dev["poverty_ratio"].dropna()
    qs = pir.quantile([0, .2, .4, .6, .8, 1.0]).values
    means = [pir[(pir >= qs[i]) & (pir <= qs[i + 1])].mean() for i in range(5)]
    print("   NHANES PIR 五分位均值 = %s" % [round(float(x), 3) for x in means])
    kd["poverty_ratio"] = kd["incm5"].map({i + 1: means[i] for i in range(5)})
    # education：edu 1–4 → 1–5（单调线性）
    kd["education"] = 1.0 + (kd["education_raw"] - 1.0) * (4.0 / 3.0)

    kd = kd[kd["age"] >= 18].copy()
    kd = kd[kd["sex"].isin([1, 2])]
    kd = kd[kd["ckd"].notna()].copy()
    med_u = float(np.nanmedian(kd["uacr"]))
    print("   KNHANES uACR 中位数 = %.2f mg/g（门 2–20）%s" % (med_u, "✅" if 2 <= med_u <= 20 else "❌"))
    if not (2 <= med_u <= 20):
        sys.exit("❌ 单位门未过 → 中止")
    print("   KNHANES 分析队列 n=%d | CKD=%d (%.2f%%)" % (len(kd), int(kd.ckd.sum()), 100 * kd.ckd.mean()))

    print("== 4) 跨体系性能 ==")
    Xk = design_rb(kd[RB])
    p_k = m_rb.predict_proba(F(Xk, med_rb))[:, 1]
    y_k = kd["ckd"].values
    auc_k = roc_auc_score(y_k, p_k); lo, hi = boot_auc(y_k, p_k)
    print("   AUC = %.4f (95%% CI %.4f–%.4f)  n=%d  CKD=%d" % (auc_k, lo, hi, len(y_k), int(y_k.sum())))

    # 校准
    eps = 1e-6; lp = np.log(np.clip(p_k, eps, 1 - eps) / (1 - np.clip(p_k, eps, 1 - eps)))
    b1, b0 = np.polyfit(lp, y_k, 1)
    dec = pd.qcut(p_k, 10, duplicates="drop")
    obs = pd.Series(y_k).groupby(dec, observed=True).mean().values
    exp = pd.Series(p_k).groupby(dec, observed=True).mean().values
    wt = pd.Series(y_k).groupby(dec, observed=True).size().values / len(y_k)
    ece = float(np.sum(wt * np.abs(obs - exp)))
    print("   校准 slope=%.3f intercept=%.3f ECE=%.4f" % (b1, b0, ece))

    # 冻结阈值
    fl = p_k >= THR
    tp = int(((fl == 1) & (y_k == 1)).sum()); fp = int(((fl == 1) & (y_k == 0)).sum())
    fn = int(((fl == 0) & (y_k == 1)).sum()); tn = int(((fl == 0) & (y_k == 0)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / max(tp + fp, 1)
    print("   阈值 %.2f：sens=%.4f spec=%.4f PPV=%.4f | 标记/千=%.1f 真阳/千=%.1f"
          % (THR, sens, spec, ppv, 1000 * fl.mean(), 1000 * y_k.mean()))

    # DCA
    dca = {}
    for t in (0.05, 0.10, 0.15, 0.20, 0.25):
        nb = (tp / len(y_k)) - (fp / len(y_k)) * (t / (1 - t)) if True else None
        # 用阈值 t 重算（标准 DCA）
        f2 = p_k >= t
        tp2 = int(((f2 == 1) & (y_k == 1)).sum()); fp2 = int(((f2 == 1) & (y_k == 0)).sum())
        dca["%.2f" % t] = {"model_nb": float(tp2 / len(y_k) - (fp2 / len(y_k)) * (t / (1 - t))),
                           "treat_all_nb": float(y_k.mean() - (1 - y_k.mean()) * (t / (1 - t)))}
    print("   DCA(model / treat-all): " + " | ".join(
        "%s %.4f/%.4f" % (k, v["model_nb"], v["treat_all_nb"]) for k, v in dca.items()))

    # 敏感性
    pit = []
    kd2 = kd.copy(); kd2["poverty_ratio"] = float(pir.median())
    p2 = m_rb.predict_proba(F(design_rb(kd2[RB]), med_rb))[:, 1]
    pit.append({"label": "income=队列中位数（映射不敏感）", "auc": round(float(roc_auc_score(y_k, p2)), 4)})
    kd3 = kd.copy(); kd3["education"] = kd3["education_raw"]  # 不重标
    p3 = m_rb.predict_proba(F(design_rb(kd3[RB]), med_rb))[:, 1]
    pit.append({"label": "education 原始 1–4（不重标）", "auc": round(float(roc_auc_score(y_k, p3)), 4)})
    for x in pit: print("   敏感性 %-32s AUC=%.4f" % (x["label"], x["auc"]))

    res = {
        "design": "frozen NHANES 2011-2016 XGBoost; race-blind variant (pre-specified for cross-system use)",
        "self_check_nhanes_2017_2018_auc": round(float(auc_ck), 4),
        "nhanes_race_blind_2017_2018_auc": round(float(auc_rb_val), 4),
        "harmonization": {"income": "KNHANES incm5 (quintile) -> NHANES dev PIR quintile means",
                          "income_quintile_means": [round(float(x), 4) for x in means],
                          "education": "KNHANES edu (1-4) linearly rescaled to NHANES DMDEDUC2 scale (1-5)"},
        "knh_cohort": {"n": int(len(kd)), "n_ckd": int(kd.ckd.sum()), "ckd_pct": round(float(100 * kd.ckd.mean()), 2),
                       "uacr_median_mg_g": round(med_u, 3)},
        "auc": {"value": round(float(auc_k), 4), "ci": [round(lo, 4), round(hi, 4)]},
        "calibration": {"slope": round(float(b1), 4), "intercept": round(float(b0), 4), "ece": round(ece, 4)},
        "at_frozen_threshold_0.16": {"sensitivity": round(sens, 4), "specificity": round(spec, 4),
                                     "ppv": round(ppv, 4), "flagged_per_1000": round(1000 * float(fl.mean()), 1),
                                     "true_cases_per_1000": round(1000 * float(y_k.mean()), 1)},
        "dca": dca,
        "sensitivity_analyses": pit,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n✅ 写出 %s" % OUT)


if __name__ == "__main__":
    main()
