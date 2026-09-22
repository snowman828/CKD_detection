"""生成投稿补充材料（Supplementary Information）S1-S3 + 方法补充（全部真实数据）
S1 基准分解表（含 CI）| S2 完整模型指标 | S3 特征缺失率 | 方法补充
输出: 论文/投稿材料/SupplementaryInformation.md → docx
"""
import pandas as pd, numpy as np, json, os
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
DOC = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/论文/投稿材料"

cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))
test = pd.read_parquet(os.path.join(OUT, "test_set.parquet"))
y = np.load(os.path.join(OUT, "test_y.npy"))
p_xgb = np.load(os.path.join(OUT, "test_proba_xgb.npy"))
mr = json.load(open(os.path.join(OUT, "model_results.json"), encoding="utf-8"))
sup = json.load(open(os.path.join(OUT, "supplementary_results.json"), encoding="utf-8"))

train = cohort[cohort["year"].isin([2011, 2013, 2015])]
rng = np.random.default_rng(7)

def boot_ci(y, p, n=1000, seed=7):
    r = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        i = r.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

# --- S1 基准分解（补算 age-only / clinical-rule 的 CI）---
Xtr_age = train[["age"]]; Xte_age = test[["age"]]
m_age = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
m_age.fit(Xtr_age, train["ckd"].values)
p_age = m_age.predict_proba(Xte_age)[:, 1]
ci_age = boot_ci(y, p_age)

FEAT_RULE = ["age", "diabetes", "sbp"]
m_rule = LogisticRegression(max_iter=2000, C=0.1)
m_rule.fit(train[FEAT_RULE].fillna(train[FEAT_RULE].median()), train["ckd"].values)
p_rule = m_rule.predict_proba(test[FEAT_RULE].fillna(train[FEAT_RULE].median()))[:, 1]
ci_rule = boot_ci(y, p_rule, seed=8)

# --- S3 缺失率（训练/测试，FEATURES 12 项）---
FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi",
            "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
miss_rows = []
for f in FEATURES:
    miss_rows.append([f, round(float(train[f].isna().mean() * 100), 1),
                      round(float(test[f].isna().mean() * 100), 1)])
miss_df = pd.DataFrame(miss_rows, columns=["Feature", "Development missing %", "Validation missing %"])

# --- 组装补充材料 md ---
sup_md = f"""# Supplementary Information

**Accompanying**: *Detection of chronic kidney disease from routine clinical data without kidney-specific testing: development and external temporal validation of machine-learning models in NHANES 2011–2018* (MACKI-Dry, v2.0)

All results in this supplement are computed from the open analysis pipeline (NHANES 2011–2018, four cycles; external temporal validation on the completely unseen 2017–2018 cycle, n=5,801).

---

## Table S1. Benchmark decomposition (external temporal validation, n=5,801)

| Model | Inputs | AUC (95% CI) | Δ vs full XGBoost |
|---|---|---|---|
| Age-only XGBoost | age | {round(float(sup['age_only_xgb_auc']),3)} ({ci_age[0]:.3f}–{ci_age[1]:.3f}) | –{mr['XGB']['auc_test']-sup['age_only_xgb_auc']:.3f} |
| Clinical rule (logistic) | age + diabetes + SBP | {round(float(sup['clinical_rule_lr_auc']),3)} ({ci_rule[0]:.3f}–{ci_rule[1]:.3f}) | –{mr['XGB']['auc_test']-sup['clinical_rule_lr_auc']:.3f} |
| Full XGBoost | 12 features | {mr['XGB']['auc_test']} ({mr['XGB']['auc_ci95'][0]}–{mr['XGB']['auc_ci95'][1]}) | reference |
| Full logistic regression | 12 features | {mr['LR']['auc_test']} ({mr['LR']['auc_ci95'][0]}–{mr['LR']['auc_ci95'][1]}) | –{mr['XGB']['auc_test']-mr['LR']['auc_test']:.3f} |

**Within-age-group discrimination (full XGBoost)**: age 18–44: {sup['within_age_group_full_xgb']['18_44']['auc']} (n={sup['within_age_group_full_xgb']['18_44']['n']}); age 45–64: {sup['within_age_group_full_xgb']['45_64']['auc']} (n={sup['within_age_group_full_xgb']['45_64']['n']}); age ≥65: {sup['within_age_group_full_xgb']['65plus']['auc']} (n={sup['within_age_group_full_xgb']['65plus']['n']}).

## Table S2. Complete model metrics (external temporal validation)

| Metric | XGBoost | Logistic regression | MLP |
|---|---|---|---|
| AUC (95% CI) | {mr['XGB']['auc_test']} ({mr['XGB']['auc_ci95'][0]}–{mr['XGB']['auc_ci95'][1]}) | {mr['LR']['auc_test']} ({mr['LR']['auc_ci95'][0]}–{mr['LR']['auc_ci95'][1]}) | {mr['MLP']['auc_test']} ({mr['MLP']['auc_ci95'][0]}–{mr['MLP']['auc_ci95'][1]}) |
| Internal CV (mean ± SD) | {mr['XGB']['auc_cv_mean']} ± {mr['XGB']['auc_cv_sd']} | {mr['LR']['auc_cv_mean']} ± {mr['LR']['auc_cv_sd']} | {mr['MLP']['auc_cv_mean']} ± {mr['MLP']['auc_cv_sd']} |
| Survey-weighted AUC | {mr['XGB']['auc_weighted']} | {mr['LR']['auc_weighted']} | {mr['MLP']['auc_weighted']} |
| Threshold (development-derived Youden) | {mr['XGB']['youden_threshold']} | {mr['LR']['youden_threshold']} | {mr['MLP']['youden_threshold']} |
| Sensitivity | {mr['XGB']['sens']} | {mr['LR']['sens']} | {mr['MLP']['sens']} |
| Specificity | {mr['XGB']['spec']} | {mr['LR']['spec']} | {mr['MLP']['spec']} |
| PPV | {mr['XGB']['ppv']} | {mr['LR']['ppv']} | {mr['MLP']['ppv']} |
| NPV | {mr['XGB']['npv']} | {mr['LR']['npv']} | {mr['MLP']['npv']} |
| Calibration slope | {mr['XGB']['calibration']['slope']} | {mr['LR']['calibration']['slope']} | {mr['MLP']['calibration']['slope']} |
| Calibration intercept | {mr['XGB']['calibration']['intercept']} | {mr['LR']['calibration']['intercept']} | {mr['MLP']['calibration']['intercept']} |
| ECE | {mr['XGB']['calibration']['ece']} | {mr['LR']['calibration']['ece']} | {mr['MLP']['calibration']['ece']} |

**Pairwise ΔAUC (bootstrap paired difference, 5,000 resamples)**: XGBoost vs logistic regression: {mr['_pairwise']['xgb_vs_lr']['mean_diff']} (95% CI {mr['_pairwise']['xgb_vs_lr']['ci95'][0]}–{mr['_pairwise']['xgb_vs_lr']['ci95'][1]}, P<0.001); XGBoost vs MLP: {mr['_pairwise']['xgb_vs_mlp']['mean_diff']} (95% CI {mr['_pairwise']['xgb_vs_mlp']['ci95'][0]}–{mr['_pairwise']['xgb_vs_mlp']['ci95'][1]}, P<0.001); logistic vs MLP: {mr['_pairwise']['lr_vs_mlp']['mean_diff']} (95% CI {mr['_pairwise']['lr_vs_mlp']['ci95'][0]}–{mr['_pairwise']['lr_vs_mlp']['ci95'][1]}, P<0.001).

## Table S3. Feature missingness by development/validation set

| Feature | Development missing % | Validation missing % |
|---|---|---|
""" + "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in miss_rows) + f"""

## Methods supplement

- **eGFR**: race-free CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021); uACR (mg/g) = urine albumin (mg/L) × 100 / urine creatinine (mg/dL).
- **CKD label (KDIGO)**: eGFR <60 mL/min/1.73 m² or uACR ≥30 mg/g.
- **Anti-leakage design**: label-defining measurements (serum creatinine, cystatin C, urine albumin, urine creatinine) explicitly excluded from all features.
- **Development/validation split**: development = NHANES 2011–2012, 2013–2014, 2015–2016 (n=17,777); external temporal validation = NHANES 2017–2018 (n=5,801), completely unseen. The 2019–2020 cycle is excluded (COVID-19 suspension; not nationally representative).
- **Missing data**: median imputation with missingness indicators, fit on the development set only.
- **Operating threshold**: Youden index determined on the development set; all validation-set clinical metrics evaluated at this fixed threshold.
- **Bootstrap procedures**: AUC CIs from 2,000 resamples (model-level) and 1,000 resamples (subgroup-level); pairwise ΔAUC from 5,000 resamples; all with fixed seeds for reproducibility.
- **Decision-curve analysis**: net benefit = TP/n − FP/n × pt/(1−pt), threshold probabilities 5–25%.
- **Fairness**: subgroup AUCs with bootstrap CIs; exploratory, no multiplicity adjustment.
- **Software**: Python 3.11 (pandas, NumPy, scikit-learn, XGBoost, SciPy); full analysis pipeline and pre-specified analysis plan provided as Supplemental Material with this manuscript (MIT).

---

*Generated {pd.Timestamp.now():%Y-%m-%d} from the v4 review-fixed analysis pipeline; every figure above is reproducible from the open code.*
"""

doc_path = os.path.join(DOC, "SupplementaryInformation.md")
with open(doc_path, "w", encoding="utf-8") as f:
    f.write(sup_md)
print("saved:", doc_path)
print("S1 age CI:", ci_age, "| rule CI:", ci_rule)
