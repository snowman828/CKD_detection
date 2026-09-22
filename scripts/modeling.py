"""MACKI-Dry 建模与分析 v3（审稿修复版，2026-08-28）
修复项（对应 5 位审稿人 P0/P1）：
  F1 测试集定义：训练 = G+H+I(2011-16)，外部时间验证 = 仅 J(2017-18)（杜绝 2019 混入）
  F2 插补协议：缺失统计量仅在训练集拟合，测试集用训练集统计量变换（防测试集泄漏）
  F3 Youden 阈值：在训练集确定，应用于测试集（防评估泄漏）
  F4 补算 XGB vs MLP bootstrap 配对差（原稿声称无代码支撑）
  F5 加权 AUC 写入 JSON（原稿引用但真源缺失）
设计：
- 特征（防泄漏）：人口学/体格/糖代谢/血脂 —— 无肾检验值
- 指标：AUC(95%CI bootstrap)/校准/亚组/加权
"""
import pandas as pd, numpy as np, json, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results"
cohort = pd.read_parquet(os.path.join(OUT, "cohort.parquet"))

FEATURES = ["age", "sex", "race", "poverty_ratio", "education", "bmi",
            "sbp", "dbp", "hba1c", "diabetes", "total_cholesterol", "hdl"]
TARGET = "ckd"

def prep_fit(X):
    """F2: 在训练集拟合插补统计量（中位数），返回可复用的 transform"""
    med = X.median()
    def transform(X_):
        X_ = X_.copy()
        miss = X_.isna()
        X_ = X_.fillna(med)
        X_ = pd.concat([X_, miss.add_prefix("miss_")], axis=1)
        return X_
    return transform

def add_dummies(X):
    X = X.copy()
    X["race_black"] = (X["race"] == 4).astype(float)
    X["race_hisp"] = (X["race"].isin([1, 2])).astype(float)
    X["race_other"] = (~X["race"].isin([3, 4, 1, 2])).astype(float)
    X["female"] = (X["sex"] == 2).astype(float)
    return X.drop(columns=["race", "sex"])

train = cohort[cohort["year"].isin([2011, 2013, 2015])]   # F1: G+H+I
test  = cohort[cohort["year"] == 2017]                     # F1: 仅 J
print(f"train n={len(train)} (ckd={train[TARGET].sum():.0f}) | test n={len(test)} (ckd={test[TARGET].sum():.0f})")

Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
fit = prep_fit(Xtr_raw)
Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)
ytr, yte = train[TARGET].values, test[TARGET].values

def boot_auc(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

def calibration(y, p, n_bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(mean_p=("p", "mean"), mean_y=("y", "mean"), n=("y", "size"))
    slope, intercept = np.polyfit(g["mean_p"], g["mean_y"], 1)
    ece = np.average(np.abs(g["mean_p"] - g["mean_y"]), weights=g["n"])
    return {"slope": float(slope), "intercept": float(intercept), "ece": float(ece)}

MODELS = {
    "LR": LogisticRegression(max_iter=2000, C=0.1),
    "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         n_jobs=4, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500,
                         early_stopping=True, random_state=42),
}

results = {}
for name, model in MODELS.items():
    model.fit(Xtr, ytr)
    p_tr = model.predict_proba(Xtr)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    auc_te = roc_auc_score(yte, p_te)
    ci = boot_auc(yte, p_te)
    cv = cross_val_score(model, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc")
    # F3: Youden 阈值在训练集确定
    fpr_tr, tpr_tr, th_tr = roc_curve(ytr, p_tr)
    youden = th_tr[np.argmax(tpr_tr - fpr_tr)]
    tn, fp, fn, tp = confusion_matrix(yte, p_te >= youden).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    cal = calibration(yte, p_te)
    # F5: 加权 AUC（调查权重）写入 JSON
    wt = test["wt"].fillna(0)
    auc_w = roc_auc_score(yte, p_te, sample_weight=np.where(yte == 1, wt, wt))
    subgroups = {}
    te = test.reset_index(drop=True)
    for label, mask in [
        ("age_18_44", te["age"] < 45), ("age_45_64", (te["age"] >= 45) & (te["age"] < 65)),
        ("age_65plus", te["age"] >= 65),
        ("sex_M", te["sex"] == 1), ("sex_F", te["sex"] == 2),
        ("race_white", te["race"] == 3), ("race_black", te["race"] == 4),
        ("race_hisp", te["race"].isin([1, 2])),
        ("pov_lt1.3", te["poverty_ratio"] < 1.3), ("pov_ge1.3", te["poverty_ratio"] >= 1.3),
    ]:
        m = mask.values if hasattr(mask, "values") else mask
        if m.sum() >= 50 and len(np.unique(yte[m])) == 2:
            subgroups[label] = {"n": int(m.sum()), "auc": round(float(roc_auc_score(yte[m], p_te[m])), 4)}
    results[name] = {
        "auc_test": round(float(auc_te), 4), "auc_ci95": [round(float(x), 4) for x in ci],
        "auc_cv_mean": round(float(cv.mean()), 4), "auc_cv_sd": round(float(cv.std()), 4),
        "youden_threshold": round(float(youden), 4),
        "sens": round(float(sens), 4), "spec": round(float(spec), 4),
        "ppv": round(float(ppv), 4) if not np.isnan(ppv) else None,
        "npv": round(float(npv), 4) if not np.isnan(npv) else None,
        "calibration": cal, "auc_weighted": round(float(auc_w), 4),
        "subgroups": subgroups,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    print(f"\n=== {name} ===\nAUC(test)={results[name]['auc_test']} CI={ci} CV={cv.mean():.3f}±{cv.std():.3f} weighted={auc_w:.4f}")
    print(f"  sens={sens:.3f} spec={spec:.3f} (Youden={youden:.3f} train-derived) | cal slope={cal['slope']:.2f} ECE={cal['ece']:.3f}")

# F4: 补算 XGB vs MLP（及 XGB vs LR）bootstrap 配对差
rng = np.random.default_rng(2026)
p_xgb = MODELS["XGB"].predict_proba(Xte)[:, 1]
p_lr  = MODELS["LR"].predict_proba(Xte)[:, 1]
p_mlp = MODELS["MLP"].predict_proba(Xte)[:, 1]
def boot_diff(y, p1, p2, n=5000):
    d = []
    idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    return {"mean_diff": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_boot": round(float((d <= 0).mean()), 4)}
results["_pairwise"] = {
    "xgb_vs_lr": boot_diff(yte, p_xgb, p_lr),
    "xgb_vs_mlp": boot_diff(yte, p_xgb, p_mlp),
    "lr_vs_mlp": boot_diff(yte, p_lr, p_mlp),
}
print("\nPairwise ΔAUC:", json.dumps(results["_pairwise"], ensure_ascii=False))

with open(os.path.join(OUT, "model_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
np.save(os.path.join(OUT, "test_y.npy"), yte)
np.save(os.path.join(OUT, "test_proba_lr.npy"), p_lr)
np.save(os.path.join(OUT, "test_proba_xgb.npy"), p_xgb)
np.save(os.path.join(OUT, "test_proba_mlp.npy"), p_mlp)
test.reset_index(drop=True).to_parquet(os.path.join(OUT, "test_set.parquet"), index=False)
print("\nsaved: model_results.json (v3, review-fixed) + test predictions")
