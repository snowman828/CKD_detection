# -*- coding: utf-8 -*-
"""npjDM 三图呈现层重制（journal-grade restyle）
核心纪律：**计算段逐字复用原脚本 m6/m7/m8 的头部代码**（切分标记之前原样 exec），
         只替换绘图段 → 数字零改动风险。
修复用户报告的三处缺陷：
  Fig1 图例压标题 + 字体不一致 + 横轴标签拥挤  → 单张 matplotlib 组合图 + 图例下置横排
  Fig2 y 轴中文 → 豆腐块                        → 全英文标签（英文稿本就不该有中文）
  Fig3 图例与柱体重叠                           → 图例移出坐标区、置于下边、横向排列
输出：300 dpi PNG + 矢量 PDF（pdf.fonttype=42 嵌入字体，Nature/Elsevier 通用）
"""
import io, os, shutil, sys, re

ROOT = r"G:\项目文件\CKD多模态AI早诊_本项目专属归档\09_无湿实验执行"
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "results", "m2")
PKG_FIG = os.path.join(ROOT, "投稿包", "02_npjDM_Followup", "figures")
BACKUP = os.path.join(OUT, "_prev_style_20260915")

# ---------- 0) 备份现有图件（不覆盖即不丢） ----------
os.makedirs(BACKUP, exist_ok=True)
for f in ["figure1_substudy1.png", "figure1_panel_a.png", "figure1_panel_b.png",
          "figure2_forest.png", "figure2_forest.pdf", "figure3_spectrum.png", "figure3_spectrum.pdf"]:
    src = os.path.join(OUT, f)
    dst = os.path.join(BACKUP, f)
    if os.path.exists(src) and not os.path.exists(dst):   # 只备一次：保住真原件
        shutil.copy2(src, dst)
print(f"备份旧版图件（已存在则跳过）-> {BACKUP}")

def head_of(fname, marker):
    """读取原脚本，返回 marker 之前的计算段（逐字；仅剥掉绘图段才用得到的 seaborn import）"""
    p = os.path.join(SCRIPTS, fname)
    t = io.open(p, encoding="utf-8").read()
    i = t.find(marker)
    if i < 0:
        raise SystemExit(f"❌ 标记未找到: {fname} :: {marker}")
    head = t[:i]
    lines = [l for l in head.splitlines(True) if "import seaborn" not in l]
    head2 = "".join(lines)
    if "sns." in head2:
        raise SystemExit(f"❌ {fname}: 计算段仍使用 sns.（不得剥离 seaborn）")
    if len(lines) != len(head.splitlines(True)):
        print(f"    · {fname}: 剥离 1 行 seaborn import（计算段无引用，已断言）")
    return head2, t

# =====================================================================
# M6 / Figure 1
# =====================================================================
M6_HEAD, _ = head_of("m6_figure1.py", "# ---- Figure 1 ----")
M6_PLOT = r'''
# ===================== 新版绘图（Arial + 图例下置横排 + 矢量输出）=====================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 1.2, "axes.edgecolor": "#333333",
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "text.color": "#1A1A1A", "axes.labelcolor": "#1A1A1A",
    "axes.spines.top": False, "axes.spines.right": False,
})

# 通路配色（与计算段一致；底部图例横排）
PW_ORDER = ["Hemodynamic", "Glycotoxic", "Metabolic", "Demographic"]
PW_COLOR["Demographic"] = "#3F3F3F"   # P1：Demographic 灰加深，与红色的灰度分离 ΔL* 0.053→~0.19（不改原脚本，仅覆盖）
PW_META = {"Hemodynamic": "Hemodynamic (SBP, DBP)",
           "Glycotoxic": "Glycotoxic (HbA1c, diabetes)",
           "Metabolic": "Metabolic (cholesterol, HDL)",
           "Demographic": "Demographic / anthropometric"}

n = len(feat_names)
corr_abs = np.abs(corr)
corr_ordered = np.eye(n) * 0  # placeholder（保持变量名清晰）

fig = plt.figure(figsize=(7.4, 5.4))
# 5 列布局：panel a | 行树状图 | 热图 | **行标签专列** | 色条（修：原为 4 列，右侧标签与色条同列竞争→叠压）
gs = fig.add_gridspec(4, 5, height_ratios=[0.30, 2.55, 1.05, 0.50],
                      width_ratios=[1.02, 0.22, 1.55, 0.45, 0.07],
                      wspace=0.10, hspace=0.06,
                      left=0.065, right=0.985, top=0.940, bottom=0.045)

# ---- a) SHAP 重要性条形图 ----
axa = fig.add_subplot(gs[1, 0])
top = mean_abs.head(10)[::-1]
top_colors = [PW_COLOR[pw_of[f]] for f in top.index]
axa.barh(range(len(top)), top.values, color=top_colors, edgecolor="none", height=0.70)
axa.set_yticks(range(len(top)))
axa.set_yticklabels(top.index, fontsize=7)
axa.set_xlabel("Mean |SHAP|", fontsize=7)
axa.set_title("a  Feature importance", loc="left", fontweight="bold", fontsize=8, pad=6)
axa.set_xlim(0, top.max() * 1.12)
axa.tick_params(axis="x", labelsize=7)
axa.grid(axis="x", color="#EFEFEF", lw=1.2, zorder=0)
axa.set_axisbelow(True)

# ---- b) 聚类树 + 重排热图（同一 Z，与报告 ARI 的聚类一致）----
dcol_ax = fig.add_subplot(gs[0, 2])
dcol = dendrogram(Z, orientation="top", no_labels=True, ax=dcol_ax,
                  color_threshold=0, above_threshold_color="#8C8C8C")
leaves = dcol["leaves"]
axcd = dcol_ax
axcd.set_axis_off()

axrd = fig.add_subplot(gs[1, 1])
dendrogram(Z, orientation="left", no_labels=True, ax=axrd,
           color_threshold=0, above_threshold_color="#8C8C8C")
axrd.invert_yaxis()          # 令行序与热图一致（自上而下 = leaves[0..]）
axrd.set_axis_off()

M = pd.DataFrame(corr_abs, index=feat_names, columns=feat_names).values[np.ix_(leaves, leaves)]
axh = fig.add_subplot(gs[1, 2])
im = axh.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
# 展示名（长变量名压短，避免竖排标签溢出；矩阵顺序与原始特征名不变）
DISPLAY = dict(poverty_ratio="PIR", race_black="race: Black", race_hisp="race: Hispanic",
               race_other="race: Other", total_cholesterol="total chol.",
               miss_total_cholesterol="miss: total chol.", miss_poverty_ratio="miss: PIR",
               miss_education="miss: education", miss_hba1c="miss: HbA1c",
               miss_diabetes="miss: diabetes", miss_total_cholesterol_="miss: total chol.")
def dn(f):
    if f.startswith("miss_") and f not in DISPLAY:
        return "miss: " + dn(f[5:])
    return DISPLAY.get(f, f)

xlabs_raw = [feat_names[i] for i in leaves]
axh.set_xticks(range(n))
axh.set_xticklabels([dn(f) for f in xlabs_raw], rotation=90, fontsize=7)
axh.set_yticks(range(n))
axh.set_yticklabels([dn(f) for f in xlabs_raw], fontsize=7)
axh.yaxis.tick_right(); axh.yaxis.set_label_position("right")
axh.tick_params(axis="y", pad=1.5)   # 修：给右侧行标签留出专列空间，避免与色条刻度叠压
for lab, f in zip(axh.get_yticklabels(), xlabs_raw):
    lab.set_color(PW_COLOR[pw_of[f]])
for lab, f in zip(axh.get_xticklabels(), xlabs_raw):
    lab.set_color(PW_COLOR[pw_of[f]])
axh.tick_params(length=0)
for s in axh.spines.values():
    s.set_visible(False)
axh.set_title("b  SHAP effect-pattern clustering vs CKM pathway map",
              loc="left", fontweight="bold", fontsize=8, pad=22)

# 色条（竖排，置于热图右侧**专列**；修：原与行标签同列导致叠压）
axcb = fig.add_subplot(gs[1, 4])
cb = fig.colorbar(im, cax=axcb, orientation="vertical")
cb.set_label("|SHAP correlation|", fontsize=7, labelpad=3)
cb.ax.tick_params(labelsize=7, width=1.2, length=2.5)

# ARI 统计量（与计算段同一数值）
_p_txt = "P < 0.0002" if pval <= 0 else f"P = {pval:.4f}"
fig.text(0.065, 0.088, f"ARI = {ari:.3f} (permutation {_p_txt}, 5,000 shuffles)",
         fontsize=7, style="italic", color="#333333")

# ---- 统一图例：图的最下边、单行横向排列 ----
handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=PW_COLOR[pw],
                      markeredgecolor=PW_COLOR[pw], markeredgewidth=1.2,
                      markersize=7, label=PW_META[pw]) for pw in PW_ORDER]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.012),
           ncol=4, frameon=False, fontsize=7, handletextpad=0.5, columnspacing=1.1,
           title="CKM pathway", title_fontsize=7)

fig.savefig(os.path.join(OUT, "figure1_substudy1.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "figure1_substudy1.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已保存: figure1_substudy1.png / .pdf（新版式）")
'''

# =====================================================================
# M7 / Figure 2
# =====================================================================
M7_HEAD, _ = head_of("m7_figure2.py", "# ---- 森林图 ----")
M7_PLOT = r'''
# ===================== 新版绘图（全英文标签 + 图例下置横排）=====================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 1.2, "axes.edgecolor": "#333333",
    "text.color": "#1A1A1A", "axes.labelcolor": "#1A1A1A",
    "axes.spines.top": False, "axes.spines.right": False,
})

# 中文角色名 -> 英文（投稿件必须全英文；表内不出现任何中文）
EN = {
    "全体人群（参考）": "Overall cohort (reference)",
    "uACR<30 盲区人群": "uACR < 30 (albuminuria-blind)",
    "  盲区 + 糖尿病（NADKD 风险）": "      + diabetes (NADKD risk)",
    "  盲区 + 非糖尿病": "      + no diabetes",
    "年龄 18–44": "Age 18–44 y",
    "年龄 45–64": "Age 45–64 y",
    "年龄 ≥65": "Age ≥65 y",
    "PIR < 1.3（低社会经济）": "PIR < 1.3 (lower income)",
    "PIR ≥ 1.3": "PIR ≥ 1.3",
}
missing = [r["name"] for r in rows if r["name"] not in EN]
if missing:
    raise SystemExit(f"❌ 未覆盖的层名（禁止静默）: {missing}")
GRP = {"Overall": "#333333", "uACR < 30 strata": "#D62728",
       "Age strata": "#1F77B4", "PIR strata": "#2CA6A4"}

fig, ax = plt.subplots(figsize=(6.8, 4.15))
ypos = np.arange(len(rows))[::-1]
for yi, r in zip(ypos, rows):
    ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=r["color"], lw=1.8, zorder=2,
            solid_capstyle="round")
    ax.plot(r["auc"], yi, "o", color=r["color"], ms=6.6, mec="white", mew=1.2, zorder=3)

ens = [EN[r["name"]] for r in rows]
ax.set_yticks(ypos)
ax.set_yticklabels(ens, fontsize=7)
ax.set_ylim(-0.9, len(rows) - 0.1)

ax.axvline(0.75, color="#888888", ls="--", lw=1.2, zorder=1)
ax.axvline(rows[0]["auc"], color="#BBBBBB", ls=":", lw=1.2, zorder=1)
ax.set_xlabel("AUC (95% CI, 2000× bootstrap)", fontsize=7)
ax.set_xlim(0.55, 0.95)
ax.set_xticks(np.arange(0.55, 0.96, 0.05))
ax.grid(axis="x", color="#EFEFEF", lw=1.2, zorder=0)
ax.set_axisbelow(True)
ax.set_title("Stratified CKD detection performance — NHANES 2017–18 external validation",
             loc="left", fontsize=8, fontweight="bold", pad=8)

# n 标注：贴左轴内侧、与 y 标签留白，避免与标题/标签相撞
for yi, r in zip(ypos, rows):
    ax.annotate(f"n = {r['n']:,}", xy=(0.553, yi), fontsize=7, color="#555555",
                va="center", ha="left", zorder=4)

# ---- 统一图例：最下边、横向排列（组别配色 + 两条参考线）----
handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                  markeredgecolor=c, markeredgewidth=1.2, markersize=6.5, label=k)
           for k, c in GRP.items()]
handles += [Line2D([0], [0], color="#888888", ls="--", lw=1.2, label="Clinical threshold 0.75"),
            Line2D([0], [0], color="#BBBBBB", ls=":", lw=1.2, label=f"Overall AUC {rows[0]['auc']:.3f}")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3,
           frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.0)
fig.subplots_adjust(left=0.27, right=0.985, top=0.90, bottom=0.20)
fig.savefig(os.path.join(OUT, "figure2_forest.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "figure2_forest.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已保存: figure2_forest.png / .pdf（全英文标签 + 图例下置）")
'''

# =====================================================================
# M8 / Figure 3
# =====================================================================
M8_HEAD, _ = head_of("m8_figure3.py", "plt.rcParams.update")
M8_PLOT = r'''
# ===================== 新版绘图（图例移出坐标区、下置横排）=====================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 1.2, "axes.edgecolor": "#333333",
    "text.color": "#1A1A1A", "axes.labelcolor": "#1A1A1A",
    "axes.spines.top": False, "axes.spines.right": False,
})
CAT_EN = {"albuminuria-only": "Albuminuria only (uACR ≥ 30)",
          "g3a (45–60)": "G3a (eGFR 45–59)",
          "g3b (<45)": "G3b (eGFR < 45)"}

fig, ax = plt.subplots(figsize=(5.9, 3.9))
x = np.arange(len(groups))
bottom = np.zeros(len(groups))
for ci, c in enumerate(cats):
    ax.bar(x, props[:, ci], bottom=bottom, color=COLORS[c], width=0.52,
           label=CAT_EN[c], edgecolor="white", linewidth=1.2)
    for xi, v in enumerate(props[:, ci]):
        if v > 0.03:
            ax.text(xi, bottom[xi] + v / 2, f"{v:.0%}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        elif v > 0:   # 极小段：数值外置 + 细引线（避免被误读为缺失）
            ax.annotate(f"{v:.0%}", xy=(xi + 0.27, bottom[xi] + v / 2),
                        xytext=(xi + 0.34, bottom[xi] + v / 2 - 0.06),
                        fontsize=7, color="#333333", va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", lw=1.2, color="#999999"))
    bottom += props[:, ci]

ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in groups], fontsize=7)
ax.set_xlabel("Age group (NHANES 2017–18 CKD cases)", fontsize=7)
ax.set_ylabel("Proportion of CKD cases", fontsize=7)
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
ax.set_xlim(-0.6, len(groups) - 0.15)
ax.set_title("Case severity composition by age — 91% of young-adult CKD is albuminuria-only",
             loc="left", fontsize=8, fontweight="bold", pad=8)

# ---- 统一图例：坐标区之外、图的下边、单行横向排列 ----
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
          fontsize=7, handlelength=1.1, handletextpad=0.45, columnspacing=0.9,
          title="Case severity spectrum", title_fontsize=7)
fig.subplots_adjust(left=0.13, right=0.985, top=0.90, bottom=0.32)
fig.savefig(os.path.join(OUT, "figure3_spectrum.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "figure3_spectrum.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已保存: figure3_spectrum.png / .pdf（图例移出坐标区）")
'''

# ---------- 依次执行（计算段逐字 + 新绘图段）----------
for tag, head, plot in [("M6/Figure1", M6_HEAD, M6_PLOT),
                        ("M7/Figure2", M7_HEAD, M7_PLOT),
                        ("M8/Figure3", M8_HEAD, M8_PLOT)]:
    print(f"\n{'='*70}\n▶ {tag}（计算段 {len(head.splitlines())} 行逐字复用）\n{'='*70}")
    g = {"__name__": "__main__", "__file__": os.path.join(SCRIPTS, tag)}
    exec(compile(head + plot, f"<{tag}>", "exec"), g)

# ---------- 分发到投稿包 figures/（PNG 300dpi + PDF 矢量）----------
print("\n== 分发到投稿包 ==")
PAIRS = [("figure1_substudy1", "Figure1"), ("figure2_forest", "Figure2"), ("figure3_spectrum", "Figure3")]
for stem, dst in PAIRS:
    for ext in [".png", ".pdf"]:
        s = os.path.join(OUT, stem + ext)
        if os.path.exists(s):
            d = os.path.join(PKG_FIG, dst + ext)
            shutil.copy2(s, d)
            print(f"  ✅ {dst}{ext}  {os.path.getsize(d)//1024} KB")
