"""NHANES CKD 队列构建 v2（按 SEQN 左连接合并，修复广播错误）
- 3 周期：2011-2012(G) / 2015-2016(I) / 2017-2018(J)（2019-2020 因 COVID 暂停，并入 P_ 系列，后补）
- 金标准标签（KDIGO）：CKD = eGFR<60 或 uACR≥30（CKD-EPI 2021 肌酐方程，无种族系数）
- 特征（防泄漏）：人口学/体格/糖化血红蛋白/尿酸/糖尿病史 —— 严禁肌酐/胱抑素C/尿白蛋白
"""
import pandas as pd, numpy as np, os, json

DATA = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data"
OUT  = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
os.makedirs(OUT, exist_ok=True)

CYCLES = {"G": 2011, "H": 2013, "I": 2015, "J": 2017}   # H=2013-2014（审稿修复：消除 cherry-picking 质疑）

def load(fname):
    p = os.path.join(DATA, fname + ".XPT")
    if not os.path.exists(p):
        return None
    return pd.read_sas(p, format="xport")

def merge_cols(df, sub, cols, prefix=""):
    """按 SEQN 左连接，带前缀防冲突"""
    if sub is None: return df
    rename = {c: f"{prefix}{c}" for c in cols}
    s = sub[["SEQN"] + cols].rename(columns=rename)
    return df.merge(s, on="SEQN", how="left")

frames = []
for cyc, year in CYCLES.items():
    dem = load(f"DEMO_{cyc}"); alb = load(f"ALB_CR_{cyc}"); bio = load(f"BIOPRO_{cyc}")
    bpx = load(f"BPX_{cyc}"); bmx = load(f"BMX_{cyc}"); diq = load(f"DIQ_{cyc}"); ghb = load(f"GHB_{cyc}")
    if dem is None:
        print(f"[WARN] DEMO_{cyc} missing, skip cycle"); continue

    df = dem[["SEQN"]].copy()
    df["cycle"] = cyc; df["year"] = year
    df["age"] = dem["RIDAGEYR"]; df["sex"] = dem["RIAGENDR"]
    df["race"] = dem["RIDRETH1"]
    df["poverty_ratio"] = dem["INDFMPIR"]; df["education"] = dem["DMDEDUC2"]
    df["pregnant"] = dem["RIDEXPRG"]
    df["wt"] = dem["WTMEC2YR"] if "WTMEC2YR" in dem.columns else np.nan

    # 尿白蛋白/肌酐 → uACR (mg/g) = URXUMA(mg/L)/10 / URXUCR(mg/dL)/100
    df = merge_cols(df, alb, ["URXUMA", "URXUCR"], "alb_")
    df["uacr"] = (df["alb_URXUMA"] * 100.0 / df["alb_URXUCR"]).where(
        df["alb_URXUMA"].notna() & df["alb_URXUCR"].notna() & (df["alb_URXUCR"] > 0))

    # 血清生化（仅肌酐用于 eGFR 标签；胱抑素C/尿酸本周期未测，特征集不含）
    df = merge_cols(df, bio, ["LBXSCR"], "bio_")
    df.rename(columns={"bio_LBXSCR": "creatinine"}, inplace=True)

    # CKD-EPI 2021 肌酐方程（无种族系数）
    scr = df["creatinine"]
    kappa = np.where(df["sex"] == 2, 0.7, 0.9)
    alpha = np.where(df["sex"] == 2, -0.241, -0.302)
    s = scr / kappa
    egfr = 142.0 * np.minimum(s, 1.0) ** alpha * np.maximum(s, 1.0) ** (-1.200) \
           * 0.9938 ** df["age"] * np.where(df["sex"] == 2, 1.012, 1.0)
    df["egfr"] = egfr.where(scr.notna() & df["age"].notna())

    # 血压 / 体格 / 糖尿病 / 糖化血红蛋白
    df = merge_cols(df, bpx, ["BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2"], "bpx_")
    df["sbp"] = df["bpx_BPXSY1"].fillna(df["bpx_BPXSY2"])
    df["dbp"] = df["bpx_BPXDI1"].fillna(df["bpx_BPXDI2"])
    df = merge_cols(df, bmx, ["BMXBMI"], "bmx_")
    df.rename(columns={"bmx_BMXBMI": "bmi"}, inplace=True)
    df = merge_cols(df, diq, ["DIQ010"], "diq_")
    df["diabetes_q"] = df["diq_DIQ010"]
    df = merge_cols(df, ghb, ["LBXGH"], "ghb_")
    df.rename(columns={"ghb_LBXGH": "hba1c"}, inplace=True)

    # 血脂（总胆固醇 LBXTC / HDL LBDHDD）
    df = merge_cols(df, load(f"TCHOL_{cyc}"), ["LBXTC"], "lip_")
    df = merge_cols(df, load(f"HDL_{cyc}"), ["LBDHDD"], "lip_")
    df.rename(columns={"lip_LBXTC": "total_cholesterol", "lip_LBDHDD": "hdl"}, inplace=True)

    frames.append(df)

cohort = pd.concat(frames, ignore_index=True)

# --- 标签 ---
cohort["ckd"] = ((cohort["egfr"] < 60) | (cohort["uacr"] >= 30)).astype(float)
cohort["ckd_g3plus"] = (cohort["egfr"] < 60).astype(float)
cohort["albuminuria"] = (cohort["uacr"] >= 30).astype(float)
cohort["diabetes"] = np.where(cohort["diabetes_q"] == 1, 1.0,
                              np.where(cohort["diabetes_q"] == 2, 0.0, np.nan))  # 缺失保留 NaN → 插补层处理（审稿 P1-6）
cohort["nadkd_candidate"] = ((cohort["diabetes"] == 1) & (cohort["egfr"] < 60) & (cohort["uacr"] < 30)).astype(float)

# --- 排除 ---
n0 = len(cohort)
cohort = cohort[cohort["age"] >= 18]
cohort = cohort[cohort["pregnant"] != 1]
cohort = cohort[cohort["ckd"].notna()]
n1 = len(cohort)
print(f"raw={n0}  after excl(age18+/non-preg/labelable)={n1}")

cohort.to_parquet(os.path.join(OUT, "cohort.parquet"), index=False)

wt = cohort["wt"].fillna(0)
def wprev(mask):
    return (cohort.loc[mask, "wt"].sum() / wt.sum()) * 100

summary = {
    "n_total": int(n1),
    "ckd_prevalence_weighted_pct": round(wprev(cohort["ckd"] == 1), 2),
    "ckd_g3plus_weighted_pct": round(wprev(cohort["ckd_g3plus"] == 1), 2),
    "albuminuria_weighted_pct": round(wprev(cohort["albuminuria"] == 1), 2),
    "nadkd_candidate_weighted_pct": round(wprev(cohort["nadkd_candidate"] == 1), 2),
    "diabetes_weighted_pct": round(wprev(cohort["diabetes"] == 1), 2),
    "by_cycle": {c: {"n": int((cohort["cycle"] == c).sum()),
                     "ckd_pct": round((cohort.loc[(cohort["cycle"] == c) & (cohort["ckd"] == 1), "wt"].sum()
                                       / cohort.loc[cohort["cycle"] == c, "wt"].sum()) * 100, 2)}
                 for c in CYCLES},
    "egfr_median": float(cohort["egfr"].median()),
    "uacr_median": float(cohort["uacr"].median()),
    "age_mean": float(cohort["age"].mean()),
    "female_pct": round(float((cohort["sex"] == 2).mean() * 100), 1),
}
with open(os.path.join(OUT, "cohort_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
