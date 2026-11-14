"""M4 sub-study H5-B: systemic footprint using all-cause and cardiovascular mortality (NHANES LMF).
Pre-specified rule: all-cause C-index >= 0.70 and CVD C-index >= 0.70 supports the systemic-footprint hypothesis.
Both below 0.60 indicates kidney-specific co-occurrence rather than a systemic footprint.
Method: XGBoost with the same hyperparameters on routine features, giving a mortality risk score evaluated by Harrell C (censoring-aware).
Data: NHANES 2011-2018 cohort linked by SEQN to the 2019 public-use LMF (follow-up to 2019-12-31).
"""
import pandas as pd, numpy as np, json, os
from xgboost import XGBClassifier

OUT = r"results"
LMF = r"data/lmf"
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

# ---- parse the fixed-width LMF file ----
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
print(f"LMF merged: {len(lmf)} 条 | deaths {int(lmf['mortstat'].fillna(0).sum())} | before linkage")

df = cohort.merge(lmf[["seqn","mortstat","ucod_leading","permth_int","permth_exm"]],
                  left_on="SEQN", right_on="seqn", how="left")
print(f"cohort merged: {len(df)} | with mortality status: {df['mortstat'].notna().sum()} | 死亡: {int(df['mortstat'].fillna(0).sum())}")

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

# ---- outcome definitions (fixed before the train/test split) ----
df["fu_years"] = df["permth_int"] / 12.0
df["mort"] = (df["mortstat"] == 1)
df["cvd_mort"] = df["mort"] & (df["ucod_leading"] == 1)   # ucod_leading = 1 denotes heart disease
df["other_mort"] = df["mort"] & ~df["cvd_mort"]

train = df[df["year"].isin([2011, 2013, 2015])]
test  = df[df["year"] == 2017]
Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
fit = prep_fit(Xtr_raw)
Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)

def harrell_c(time, event, risk):
    """Vectorised Harrell C-index with tie handling."""
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
        print(f"  [{label}] too few events ({ytr.sum()}); skipped"); return None
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=42)
    m.fit(Xtr, ytr)
    risk = m.predict_proba(Xte)[:, 1]
    c = harrell_c(test_time, test_ev, risk)
    n_ev = int(test_ev.sum())
    print(f"  [{label}] training events={ytr.sum()} | test events={n_ev} | Harrell C = {c:.4f}")
    return {"n_train_events": int(ytr.sum()), "n_test_events": n_ev, "harrell_c": round(float(c), 4)}

# follow-up in years from interview; permth holds the death time for decedents and the censoring time otherwise

# evaluation on the 2017-2018 temporal test set, consistent with the main analysis
res = {}
res["all_cause"] = eval_outcome("all", train["mort"], test["mort"], test["fu_years"], "all-cause mortality")
# CVD mortality is cause-specific; non-CVD deaths are censored
res["cvd"] = eval_outcome("cvd", train["cvd_mort"], test["cvd_mort"], test["fu_years"], "CVD mortality")

if res["all_cause"] and res["cvd"]:
    c_all, c_cvd = res["all_cause"]["harrell_c"], res["cvd"]["harrell_c"]
    if c_all >= 0.70 and c_cvd >= 0.70:
        res["decision"] = "H5b systemic footprint supported (both C-indices >= 0.70)"
    elif c_all < 0.60 and c_cvd < 0.60:
        res["decision"] = "kidney-specific co-occurrence (not a systemic footprint)"
    else:
        res["decision"] = f"intermediate (all-cause {c_all:.2f}, CVD {c_cvd:.2f})"
    print(f"\n[decision] {res['decision']}")

res["n_test"] = int(len(test))
res["test_deaths"] = int(test["mort"].sum())
with open(os.path.join(OUT, "m2", "substudy5b_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"saved: substudy5b_results.json")
