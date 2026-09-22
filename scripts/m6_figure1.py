"""M6 Figure 1 —— 子研究① 足迹探针主图（投稿级数据图）
a) 特征 SHAP 重要性条形图（Top 10，按 CKM 通路配色）
b) SHAP 作用模式聚类热图：特征×特征 |SHAP 相关| + Ward 树状图 + CKM 通路色带 + ARI 标注
输出：300dpi PNG + PDF 矢量（Nature 单栏宽度）
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2"
cohort = pd.read_parquet(r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/cohort.parquet")

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

train = cohort[cohort["year"].isin([2011, 2013, 2015])]
test  = cohort[cohort["year"] == 2017]
Xtr_raw, Xte_raw = add_dummies(train[FEATURES]), add_dummies(test[FEATURES])
fit = prep_fit(Xtr_raw)
Xtr, Xte = fit(Xtr_raw), fit(Xte_raw)
ytr, yte = train["ckd"].values, test["ckd"].values

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=4, random_state=42)
model.fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
print(f"test AUC = {auc:.4f}")

# ---- TreeSHAP（xgboost 内置 pred_contribs）----
dte = DMatrix(Xte, label=yte)
shap = model.get_booster().predict(dte, pred_contribs=True)[:, :-1]  # 去掉 bias 列
feat_names = list(Xte.columns)
shap_df = pd.DataFrame(shap, columns=feat_names)
mean_abs = shap_df.abs().mean().sort_values(ascending=False)

# ---- CKM 通路映射（预注册固定）----
PATHWAY = {"Hemodynamic": ["sbp","dbp"],
           "Glycotoxic": ["hba1c","diabetes"],
           "Metabolic": ["total_cholesterol","hdl"],
           "Demographic": ["age","female","race_black","race_hisp","race_other",
                           "poverty_ratio","education","bmi"],
           "Missing-indicator": [c for c in feat_names if c.startswith("miss_")]}
PW_COLOR = {"Hemodynamic": "#D62728", "Glycotoxic": "#FF7F0E", "Metabolic": "#1F77B4",
            "Demographic": "#7F7F7F", "Missing-indicator": "#C7C7C7"}
pw_of = {f: pw for pw, fs in PATHWAY.items() for f in fs}
colors = [PW_COLOR[pw_of[f]] for f in feat_names]

# ---- ARI（聚类 vs 映射，同 M2：squareform 压缩距离 + 5 层含 missing）----
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
corr = np.corrcoef(shap.T); corr = np.nan_to_num(corr, nan=0.0)
dist = 1 - np.abs(corr); np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2; dist = np.clip(dist, 0, None)
Z = linkage(squareform(dist), method="ward")
n_clusters = len(PATHWAY)  # 5 层（含 Missing-indicator）
cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")
ref_labels = np.array([list(PATHWAY).index(pw_of[f]) for f in feat_names])
ari = adjusted_rand_score(ref_labels, cluster_labels)
print(f"ARI = {ari:.3f}")

# ---- 置换零分布（快速 200 次）----
rng = np.random.default_rng(42)
nulls = []
for _ in range(200):
    perm = rng.permutation(ref_labels)
    nulls.append(adjusted_rand_score(perm, cluster_labels))
pval = (np.array(nulls) >= ari).mean()
print(f"permutation p = {pval:.4f}")

# ---- Figure 1 ----
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7})
fig = plt.figure(figsize=(7.2, 4.6))  # 双栏宽度
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.1], wspace=0.35)

# --- a) SHAP 重要性条形图 ---
axa = fig.add_subplot(gs[0])
top = mean_abs.head(10)[::-1]
top_colors = [PW_COLOR[pw_of[f]] for f in top.index]
axa.barh(range(len(top)), top.values, color=top_colors, edgecolor="none", height=0.72)
axa.set_yticks(range(len(top)))
axa.set_yticklabels(top.index, fontsize=7)
axa.set_xlabel("Mean |SHAP|")
axa.set_title("a  Feature importance", loc="left", fontweight="bold", fontsize=9)
axa.set_xlim(0, top.max() * 1.1)

# --- b) 聚类热图（DataFrame 保证特征名标签）---
corr_abs = pd.DataFrame(np.abs(corr), index=feat_names, columns=feat_names)
g = sns.clustermap(corr_abs, method="ward", metric="euclidean", cmap="Reds",
                   vmin=0, vmax=1, row_colors=colors, col_colors=colors,
                   xticklabels=True, yticklabels=True, figsize=(4.6, 4.6),
                   dendrogram_ratio=(0.16, 0.10), cbar_pos=(0.02, 0.88, 0.03, 0.08))
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=60, fontsize=6)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
g.ax_row_dendrogram.set_visible(True)
for lab in g.ax_heatmap.get_yticklabels():
    f = lab.get_text()
    lab.set_color(PW_COLOR[pw_of[f]])
# ARI 标注（图内小字）
g.ax_heatmap.text(0.02, -0.30, f"ARI = {ari:.3f}  (permutation p = {pval:.4f})",
                  transform=g.ax_heatmap.transAxes, fontsize=7, style="italic")
# 通路图例
handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markersize=8, label=pw)
           for pw, c in PW_COLOR.items() if pw != "Missing-indicator"]
g.fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=7, ncol=1,
             title="CKM pathway", title_fontsize=7)
g.fig.suptitle("b  SHAP effect-pattern clustering vs CKM pathway map", x=0.42, y=1.0,
               fontweight="bold", fontsize=9)
g.savefig(os.path.join(OUT, "figure1_panel_b.png"), dpi=300, bbox_inches="tight")

# 组合保存（a 已画在 fig，b 用 clustermap 单独 fig——合并保存）
fig.savefig(os.path.join(OUT, "figure1_panel_a.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 合成整图（用 PIL 拼接 a+b）
from PIL import Image
pa = Image.open(os.path.join(OUT, "figure1_panel_a.png"))
pb = Image.open(os.path.join(OUT, "figure1_panel_b.png"))
h = max(pa.height, pb.height)
wa = int(pa.width * h / pa.height); wb = int(pb.width * h / pb.height)
combo = Image.new("RGB", (wa + wb + 20, h), "white")
combo.paste(pa.resize((wa, h)), (0, 0))
combo.paste(pb.resize((wb, h)), (wa + 20, 0))
combo.save(os.path.join(OUT, "figure1_substudy1.png"), dpi=(300, 300))
print(f"已保存: figure1_substudy1.png / figure1_panel_a.png / figure1_panel_b.png")
