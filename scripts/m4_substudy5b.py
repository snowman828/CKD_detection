"""M4 子研究 H5-B —— 系统性足迹（NHANES LMF 全因/CVD 死亡）
预注册判定（06 v1.1 §7.5）：全因死亡 C-index ≥ 0.70 且 CVD 死亡 C-index ≥ 0.70 → 系统性足迹支持
两者 < 0.60 → 肾特异共现（足迹非系统性）
方法：XGBoost（同 MACKI 参数）基线常规特征 → 死亡风险分数 → Harrell C（含删失，向量化）
数据：NHANES 2011-2018 cohort（SEQN 匹配 LMF 2019 public-use，随访至 2019-12-31）
"""
import pandas as pd, numpy as np, json, os
from xgboost import XGBClassifier

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
LMF = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data/lmf"
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

# ---- 解析 LMF（fixed-width, NHANES 版）----
def read_lmf(path):
    widths = [(1,6),(15,15),(16,16),(17,19),(20,20),(21,21),(22,22),(23,26),(43,45),(46,48)]
    cols = ["seqn","eligstat","mortstat","ucod_leading","diabetes","hyperten","dodqtr","dodyear","permth_int","permth_exm"]
    recs = []
    with open(path) as f:
        for line in f:
            if len(line) < 48: continue
            vals = {}
            for (a, b), c in zip(widths, cols):
                s = line[a-1:b].strip()
                vals[c] = float(s) if s not in ("", ".") else np.nan
            recs.append(vals)
    return pd.DataFrame(recs)

lmf_parts = []
for cyc, fn in [("2011_2012","NHANES_2011_2012_MORT_2019_PUBLIC.dat"),
                ("2013_2014","NHANES_2013_2014_MORT_2019_PUBLIC.dat"),
                ("2015_2016","NHANES_2015_2016_MORT_2019_PUBLIC.dat"),
                ("2017_2018","NHANES_2017_2018_MORT_2019_PUBLIC.dat")]:
    p = read_lmf(os.path.join(LMF, fn))
    p["cycle"] = cyc
    lmf_parts.append(p)
lmf = pd.concat(lmf_parts, ignore_index=True)
print(f"LMF 合并: {len(lmf)} 条 | 死亡 {int(lmf['mortstat'].fillna(0).sum())} | 匹配前")

df = cohort.merge(lmf[["seqn","mortstat","ucod_leading","permth_int","permth_exm"]],
                  left_on="SEQN", right_on="seqn", how="left")
print(f"Cohort 合并: {len(df)} | 有死亡状态: {df['mortstat'].notna().sum()} | 死亡: {int(df['mortstat'].fillna(0).sum())}")

FEATURES = ["age","sex","race","poverty_ratio","education","bmi","sbp","dbp",
            "hba1c","diabetes","total_cholesterol","hdl"]
def prep_fit(X):
    med = X.median()
    def transform(X_):
        X_ = X_.copy(); miss = X_.isna()
        return pd.concat([X_.fillna(med), miss.add_prefix("miss_")], axis=1)
    return transform
def add_dummies(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"]  = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

# ---- 结局定义（先于 train/test 划分）----
df["fu_years"] = df["permth_int"] / 12.0
df["mort"] = (df["mortstat"] == 1)
df["cvd_mort"] = df["mort"] & (df["ucod_leading"] == 1)   # ucod_leading=1 心脏病
df["other_mort"] = df["mort"] & ~df["cvd_mort"]

train = df[df["year"].isin([2011, 2013, 2015])]
test  = df[df["year"] == 2017]
Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
fit = prep_fit(Xtr_raw)
Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)

def harrell_c(time, event, risk):
    """向量化 Harrell's C（含并列）"""
    time = np.asarray(time, float); event = np.asarray(event, bool); risk = np.asarray(risk, float)
    comp = conc = ties = 0
    ev_idx = np.where(event)[0]
    for i in ev_idx:
        later = time > time[i]
        if later.sum() == 0: continue
        comp += later.sum()
        conc += (risk[i] > risk[later]).sum()
        ties += (risk[i] == risk[later]).sum()
    return (conc + 0.5 * ties) / comp if comp else np.nan

def eval_outcome(outcome_name, train_ev, test_ev, test_time, label):
    """outcome_name: all-cause / cvd"""
    ytr = train_ev.astype(int)
    if ytr.sum() < 100:
        print(f"  [{label}] 事件过少 ({ytr.sum()})，跳过"); return None
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=42)
    m.fit(Xtr, ytr)
    risk = m.predict_proba(Xte)[:, 1]
    c = harrell_c(test_time, test_ev, risk)
    n_ev = int(test_ev.sum())
    print(f"  [{label}] 训练事件={ytr.sum()} | 测试事件={n_ev} | Harrell C = {c:.4f}")
    return {"n_train_events": int(ytr.sum()), "n_test_events": n_ev, "harrell_c": round(float(c), 4)}

# 随访时间（月→年，从访谈起）；死亡者 permth 为死亡时间，存活者 permth 为随访终点

# 评估（test = 2017-18，外部时间验证，与主分析一致）
res = {}
res["all_cause"] = eval_outcome("all", train["mort"], test["mort"], test["fu_years"], "全因死亡")
# CVD 死亡：cause-specific（非 CVD 死亡视为删失）
res["cvd"] = eval_outcome("cvd", train["cvd_mort"], test["cvd_mort"], test["fu_years"], "CVD 死亡")

if res["all_cause"] and res["cvd"]:
    c_all, c_cvd = res["all_cause"]["harrell_c"], res["cvd"]["harrell_c"]
    if c_all >= 0.70 and c_cvd >= 0.70:
        res["decision"] = "H5b 系统性足迹支持（全因+CVD 均 ≥0.70）"
    elif c_all < 0.60 and c_cvd < 0.60:
        res["decision"] = "肾特异共现（足迹非系统性）"
    else:
        res["decision"] = f"中间状态（全因 {c_all:.2f} / CVD {c_cvd:.2f}）"
    print(f"\n[判决] {res['decision']}")

res["n_test"] = int(len(test))
res["test_deaths"] = int(test["mort"].sum())
with open(os.path.join(OUT, "m2", "substudy5b_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"已保存: substudy5b_results.json")
