# -*- coding: utf-8 -*-
"""A2 硬结局（NHANES 联动死亡）预后验证 — 阶段 1 新增分析（2026-10-04）

回答三个问题（供论文 A2 轴入库）：
  Q1【锚点】KDIGO 定义的 CKD 标签本身是否与全因/CVD 死亡相关？（证明联动管道有效、样本有功效）
  Q2【主问题】常规数据模型评分，是否在**年龄之上**提供预后判别增量？（OOF 预测，充分功效）
  Q3【次要】冻结模型在时间外验证集（2017–2018）的表现（事件数少 ⇒ 探索性）

设计与纪律：
  · 联动资格：仅 eligstat==1（NHANES 官方规定）
  · OOF 预测：在开发期（2011–2016）内部 StratifiedKFold(5, seed=42) 交叉拟合，复用 modeling.py
    的同一特征管道（dummies + 训练集中位数插补 + 缺失指示符）与同一 XGB 超参
  · 权重：合并 4 个 2 年周期 ⇒ 权重 WTMEC2YR/4（NHANES 官方合并规则）
  · 本分析为 **post-hoc/探索性**（不在预注册方案内），须在稿件如实标注；不声称因果
  · ΔC 的 95% CI = 配对 bootstrap（两个 C-index 在同一重抽样上重算）
输出：results/m2/m16_mortality_a2.json
运行：D:/hermes/scienv/Scripts/python.exe scripts/m16_mortality_a2.py
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

BASE = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
OUT = os.path.join(BASE, "results", "m2")
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi",
            "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
TARGET = "ckd"
RNG = 42
NBOOT = 500

os.makedirs(OUT, exist_ok=True)
cohort = pd.read_parquet(os.path.join(BASE, "results/cohort.parquet"))


# ---------- 特征管道（与 modeling.py 完全一致） ----------
def prep_fit(X):
    med = X.median()

    def transform(X_):
        X_ = X_.copy()
        miss = X_.isna()
        return pd.concat([X_.fillna(med), miss.add_prefix("miss_")], axis=1)

    return transform


def add_dummies(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])


# ---------- LMF 解析（官方固定宽度） ----------
def read_lmf(path):
    widths = [(1, 6), (15, 15), (16, 16), (17, 19), (20, 20), (21, 21),
              (22, 22), (23, 26), (43, 45), (46, 48)]
    cols = ["seqn", "eligstat", "mortstat", "ucod_leading", "diabetes", "hyperten",
            "dodqtr", "dodyear", "permth_int", "permth_exm"]
    recs = []
    for line in open(path, encoding="latin-1"):
        if len(line) < 48:
            continue
        vals = {}
        for (a, b), c in zip(widths, cols):
            s = line[a - 1:b].strip()
            vals[c] = float(s) if s not in ("", ".") else np.nan
        recs.append(vals)
    return pd.DataFrame(recs)


lmf_dir = os.path.join(BASE, "data/lmf")
parts = []
for cyc, fn in [("2011_2012", "NHANES_2011_2012_MORT_2019_PUBLIC.dat"),
                ("2013_2014", "NHANES_2013_2014_MORT_2019_PUBLIC.dat"),
                ("2015_2016", "NHANES_2015_2016_MORT_2019_PUBLIC.dat"),
                ("2017_2018", "NHANES_2017_2018_MORT_2019_PUBLIC.dat")]:
    p = read_lmf(os.path.join(lmf_dir, fn))
    p["lmf_cycle"] = cyc
    parts.append(p)
lmf = pd.concat(parts, ignore_index=True)

df = cohort.merge(lmf[["seqn", "eligstat", "mortstat", "ucod_leading", "permth_int", "lmf_cycle"]],
                  left_on="SEQN", right_on="seqn", how="left")
n_linked_pre = len(df)
df = df[(df["eligstat"] == 1) & df["permth_int"].notna()].reset_index(drop=True)
df["fu_years"] = df["permth_int"] / 12.0
df["mort"] = (df["mortstat"] == 1).astype(float)
df["cvd"] = ((df["mortstat"] == 1) & (df["ucod_leading"] == 1)).astype(float)
df["w"] = df["wt"].fillna(0) / 4.0          # 4 个 2 年周期合并（NHANES 规则）

res = {"design": {
    "linkage": "NHANES Linked Mortality Files 2019 (public), eligstat==1",
    "sample": "cohort (2011-2018, 4 cycles) linked to mortality, adults",
    "n_cohort": int(n_linked_pre), "n_linked_eligible": int(len(df)),
    "weights": "WTMEC2YR/4 (4 two-year cycles combined)",
    "post_hoc": True, "causal_claim": False,
    "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}}

# ---------- 描述 ----------
by_ckd = {}
for k, m in [("ckd", df["ckd"] == 1), ("no_ckd", df["ckd"] == 0)]:
    g = df[m]
    py_w = float(g["w"].sum() * g["fu_years"].mean()) if len(g) else 0.0
    by_ckd[k] = {
        "n": int(len(g)),
        "all_cause_deaths": int(g["mort"].sum()),
        "cvd_deaths": int(g["cvd"].sum()),
        "person_years": round(float(g["fu_years"].sum()), 1),
        "median_followup_years": round(float(g["fu_years"].median()), 2),
        "weighted_mortality_per_1000py": round(float((g["w"] * g["mort"]).sum() / (g["w"] * g["fu_years"]).sum() * 1000), 2) if g["w"].sum() else None,
        "weighted_cvd_per_1000py": round(float((g["w"] * g["cvd"]).sum() / (g["w"] * g["fu_years"]).sum() * 1000), 2) if g["w"].sum() else None,
    }
res["descriptives"] = {
    "by_ckd_status": by_ckd,
    "overall": {
        "n": int(len(df)), "all_cause_deaths": int(df["mort"].sum()), "cvd_deaths": int(df["cvd"].sum()),
        "person_years": round(float(df["fu_years"].sum()), 1),
        "median_followup_years": round(float(df["fu_years"].median()), 2),
    },
}
print(f"[desc] n={len(df)} all-cause deaths={int(df['mort'].sum())} CVD deaths={int(df['cvd'].sum())} "
      f"median FU={df['fu_years'].median():.2f} y", flush=True)


# ---------- Q1 锚点：CKD 标签 ↔ 死亡（校正 HR） ----------
def cox_hr_labels(ev, weighted=False):
    d = add_dummies(df[FEATURES]).assign(T=df["fu_years"].values, E=df[ev].values)
    d["ckd"] = df["ckd"].values
    cols = ["T", "E", "ckd", "age", "female", "race_black", "race_hisp", "race_other"]
    d = d[cols + (["w"] if weighted else [])].dropna(subset=["T", "E"])
    cph = CoxPHFitter(penalizer=0.1)
    kw = {"weights_col": "w"} if weighted else {}
    cph.fit(d, duration_col="T", event_col="E", **kw)
    s = cph.summary.loc["ckd"]
    out = {"hr": round(float(np.exp(s["coef"])), 3),
           "ci95": [round(float(np.exp(s["coef"] - 1.96 * s["se(coef)"])), 3),
                    round(float(np.exp(s["coef"] + 1.96 * s["se(coef)"])), 3)],
           "p": float(s["p"]), "n": int(len(d)), "events": int(d["E"].sum()),
           "adjusted_for": ["age", "female", "race_black", "race_hisp", "race_other"],
           "weighted": weighted}
    try:
        ph = proportional_hazard_test(cph, d, time_transform="rank")
        pv = {k: round(float(v), 4) for k, v in ph.summary["p"].items()}
        out["ph_test_p"] = pv.get("ckd")
    except Exception as e:
        out["ph_test_p"] = f"failed: {type(e).__name__}"
    return out


res["Q1_ckd_label_vs_death"] = {}
for ev in ("mort", "cvd"):
    res["Q1_ckd_label_vs_death"][ev] = cox_hr_labels(ev)
    try:
        res["Q1_ckd_label_vs_death"][ev + "_weighted_sensitivity"] = cox_hr_labels(ev, weighted=True)
    except Exception as e:
        res["Q1_ckd_label_vs_death"][ev + "_weighted_sensitivity"] = f"unavailable: {type(e).__name__}"
    print(f"[Q1] {ev}: HR(ckd)={res['Q1_ckd_label_vs_death'][ev]['hr']} "
          f"{res['Q1_ckd_label_vs_death'][ev]['ci95']}", flush=True)


# ---------- Q2 主问题：模型评分在年龄之上的预后增量（OOF，充分功效） ----------
X_raw = add_dummies(df[FEATURES])
y = df[TARGET].values
is_dev = df["year"].isin([2011, 2013, 2015]).values
is_test = (df["year"] == 2017).values

fit = prep_fit(X_raw[is_dev])
X = fit(X_raw)

xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=RNG)

# OOF（开发期内部 5 折）
oof = np.full(len(df), np.nan)
skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
Xdev, ydev = X[is_dev].reset_index(drop=True), y[is_dev]
oof_dev = np.zeros(is_dev.sum())
for tr, va in skf.split(Xdev, ydev):
    xgb.fit(Xdev.iloc[tr], ydev[tr])
    oof_dev[va] = xgb.predict_proba(Xdev.iloc[va])[:, 1]
oof[is_dev] = oof_dev
# 冻结模型 → 时间外验证集
xgb.fit(Xdev, ydev)
oof[is_test] = xgb.predict_proba(X[is_test])[:, 1]
score = oof
print(f"[Q2] OOF 预测已生成：dev {is_dev.sum()} / test {is_test.sum()}", flush=True)


def c_harrell(t, s, e):
    return float(concordance_index(t, -s, e))


def age_only_c(t, e, age):
    return c_harrell(t, age, e)


def delta_c_with_boot(t, s, e, age, nboot=NBOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    c_s, c_a = c_harrell(t, s, e), age_only_c(t, e, age)
    d = c_s - c_a
    deltas = []
    n = len(t)
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if len(np.unique(e[i])) < 2:
            continue
        try:
            deltas.append(c_harrell(t[i], s[i], e[i]) - age_only_c(t[i], e[i], age[i]))
        except Exception:
            continue
    lo, hi = (np.percentile(deltas, [2.5, 97.5]) if deltas else (np.nan, np.nan))
    return {"c_index_score": round(c_s, 4), "c_index_age_only": round(c_a, 4),
            "delta_c": round(d, 4),
            "delta_c_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "boot_reps_used": len(deltas), "n": int(n), "events": int(e.sum())}


def hr_per_sd(ev, mask, adj_age=True):
    s = score[mask]
    mu, sd = np.nanmean(s), np.nanstd(s)
    z = (s - mu) / sd
    d = pd.DataFrame({"T": df["fu_years"].values[mask], "E": df[ev].values[mask],
                      "z_score": z, "age": df["age"].values[mask],
                      "female": (df["sex"].values[mask] == 2).astype(float),
                      "race_black": (df["race"].values[mask] == 4).astype(float),
                      "race_hisp": np.isin(df["race"].values[mask], [1, 2]).astype(float),
                      "race_other": (~np.isin(df["race"].values[mask], [1, 2, 3, 4])).astype(float)})
    cols = ["T", "E", "z_score"] + (["age", "female", "race_black", "race_hisp", "race_other"] if adj_age else [])
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(d[cols], duration_col="T", event_col="E")
    s_ = cph.summary.loc["z_score"]
    return {"hr_per_sd": round(float(np.exp(s_["coef"])), 3),
            "ci95": [round(float(np.exp(s_["coef"] - 1.96 * s_["se(coef)"])), 3),
                     round(float(np.exp(s_["coef"] + 1.96 * s_["se(coef)"])), 3)],
            "p": float(s_["p"]), "adjusted_for": cols[3:], "n": int(len(d)), "events": int(d["E"].sum())}


res["Q2_score_prognostic_value"] = {}
for ev in ("mort", "cvd"):
    m_dev = is_dev & np.isfinite(score)
    m_test = is_test & np.isfinite(score)
    res["Q2_score_prognostic_value"][ev] = {
        "development_OOF": delta_c_with_boot(df["fu_years"].values[m_dev], score[m_dev],
                                             df[ev].values[m_dev], df["age"].values[m_dev]),
        "temporal_validation_frozen": delta_c_with_boot(df["fu_years"].values[m_test], score[m_test],
                                                        df[ev].values[m_test], df["age"].values[m_test]),
        "hr_per_sd_development": hr_per_sd(ev, m_dev),
    }
    d_ = res["Q2_score_prognostic_value"][ev]["development_OOF"]
    print(f"[Q2] {ev}: dev ΔC={d_['delta_c']:+.4f} {d_['delta_c_ci95']} "
          f"(C_score {d_['c_index_score']} vs C_age {d_['c_index_age_only']}, events {d_['events']})", flush=True)

# ---------- 功效/披露 ----------
res["disclosures"] = {
    "post_hoc": "该分析不在预注册方案内，属投稿后探索性分析",
    "power": "开发期联动样本人-年与事件数见各条目；时间外验证集事件数少（见 n_test_events），结论标为探索性",
    "ph_assumption": "对 CKD 标签模型做了比例风险检验（rank 变换），结果见 Q1_*_ph_test_p",
    "no_causal_claim": "本分析仅报告关联，不声称因果；不作为部署或替代肾脏化验的依据",
    "bootstrap_reps": NBOOT,
}

with open(os.path.join(OUT, "m16_mortality_a2.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=str)
print("saved:", os.path.join(OUT, "m16_mortality_a2.json"), flush=True)
