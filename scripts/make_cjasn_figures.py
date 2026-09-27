# -*- coding: utf-8 -*-
"""make_cjasn_figures.py —— CJASN Detection 主图 1–5 生成器（投稿出版级 / 可复现 / 全文字号 ≥8.5 pt）

【R1 修复】官方条款 78：「Text within the figure must be at least 8-point font size」；条款 79：面板标签 11 pt 粗体。
           本脚本把一切文字下限设为 8.5 pt（留 0.5 pt 余量），并在 __main__ 内置字号自检。
【R4 补齐】产出上传图件的脚本此前未留存（仓库只有 PNG 级旧脚本）→ 本脚本是可复现的唯一入口。
【风格一致】Arial；Okabe–Ito 色盲友好色板；图例置于图下、单行、无边框；仅左/下轴线；淡灰网格；标题粗体左对齐。
【数据准确】Fig1←results/participant_flow.json；Fig2–4←test_proba_{xgb,lr,mlp}.npy + test_y.npy
           （年龄-only 与三变量临床规则在开发集拟合，随机种子固定）；Fig5 用正文 Table 3 权威值（与台账一致）。
【输出】PDF（矢量、字体嵌入 fonttype=42）+ PNG(600 dpi)，落 results/figures_cjasn_v2/
"""
import os, re, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 仓内相对路径（GitHub 检出即可运行 Fig1–4；Fig5 需外部稿件，见下）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures_cjasn_v2")
MAN = os.environ.get("CKD_MANUSCRIPT_MD", os.path.join(ROOT, "CJASN_Manuscript.md"))
# 数据依赖（DATA_AVAILABILITY.md 说明）：results/test_proba_{xgb,lr,mlp}.npy、
# test_y.npy、test_set.parquet、cohort.parquet 为 gitignore 的中间产物，
# 由 scripts/download_nhanes.py + build_cohort.py + modeling.py 重建。
os.makedirs(OUT, exist_ok=True)

# ---------------- 版式唯一真源 ----------------
MIN_PT  = 8.5
PANEL_PT = 11.0
PALETTE = {"xgb": "#D55E00", "lr": "#0072B2", "mlp": "#56B4E9", "rule": "#E69F00", "age": "#CC79A7"}
FAMILY  = {"age": "#D55E00", "sex": "#0072B2", "race": "#009E73", "pov": "#CC79A7"}

matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": MIN_PT, "axes.titlesize": 9.5, "axes.labelsize": 9.0,
    "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT, "legend.fontsize": MIN_PT,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.6, "grid.alpha": 0.55,
    "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})
LEG = dict(loc="upper center", bbox_to_anchor=(0.5, -0.155), frameon=False,
           ncol=3, handlelength=1.5, columnspacing=1.3, handletextpad=0.7)

def _save(fig, name):
    a = os.path.join(OUT, name + ".pdf"); b = os.path.join(OUT, name + ".png")
    fig.savefig(a); fig.savefig(b, dpi=600); plt.close(fig)
    return a, b

# ============================== Figure 1 ==============================
def figure1():
    d = json.load(open(os.path.join(RES, "participant_flow.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.09, 5.05)); ax.axis("off"); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(-0.35, 10)
    def box(x, y, w, h, txt, fc="#F2F2F2", ec="#4D4D4D"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.055,rounding_size=0.10",
                                    linewidth=0.9, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=MIN_PT, linespacing=1.5)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.9, color="#4D4D4D", shrinkA=0, shrinkB=0))
    box(1.9, 8.60, 6.2, 1.15, f"NHANES 2011–2018 participants\n(4 survey cycles, G–J) screened: n = {d['n_screened']:,}", fc="#E8EEF4")
    arrow(5.0, 8.58, 5.0, 7.02)
    box(6.35, 7.20, 3.60, 1.25, f"Excluded: Age < 18 years (n = {d['excl_age_lt18']:,})\nExcluded: Pregnant (n = {d['excl_pregnant']:,})\nExcluded: No CKD label component available (n = {d['excl_no_label_component']:,})", fc="#FBEEEE", ec="#B03A2E")
    box(1.9, 5.75, 6.2, 1.25, f"Analytic cohort: n = {d['n_analytic']:,}\nCKD (eGFR<60 or uACR≥30 mg/g): n = {d['n_ckd']:,}", fc="#E8F4EA")
    arrow(3.9, 5.73, 2.7, 5.05); arrow(6.1, 5.73, 7.3, 5.05)
    box(0.25, 3.70, 4.6, 1.35, f"Development (cycles G–I, 2011–2016)\nn = {d['n_development_G_I']:,}", fc="#EEF3FA")
    box(5.15, 3.70, 4.6, 1.35, f"Temporal test (cycle J, 2017–2018)\nn = {d['n_temporal_test_J']:,}", fc="#FDF3E7")
    ax.plot([0.25, 9.75], [3.20, 3.20], ":", color="#7F7F7F", lw=1.0)
    box(1.9, 1.35, 6.2, 1.30, "External cross-system validation\nKNHANES 2022–2024 (Korea): n = 17,045", fc="#EEF3FA")
    ax.text(0.25, -0.25, "Counts reproduce the frozen cohort build (participant_flow.json).", fontsize=MIN_PT, color="#595959")
    return _save(fig, "Figure1")

# ============================== 数据 ==============================
def load_preds():
    y = np.load(os.path.join(RES, "test_y.npy"))
    px = np.load(os.path.join(RES, "test_proba_xgb.npy"))
    pl = np.load(os.path.join(RES, "test_proba_lr.npy"))
    pm = np.load(os.path.join(RES, "test_proba_mlp.npy"))
    test = pd.read_parquet(os.path.join(RES, "test_set.parquet"))
    coh = pd.read_parquet(os.path.join(RES, "cohort.parquet"))
    return y, px, pl, pm, test, coh

def baselines(y, test, coh):
    tr = coh[coh["year"].isin([2011, 2013, 2015])]
    age = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)
    age.fit(tr[["age"]], tr["ckd"].values); pa = age.predict_proba(test[["age"]])[:, 1]
    F = ["age", "diabetes", "sbp"]
    rule = LogisticRegression(max_iter=2000, C=0.1); rule.fit(tr[F].fillna(tr[F].median()), tr["ckd"].values)
    pr = rule.predict_proba(test[F].fillna(tr[F].median()))[:, 1]
    return pa, pr, float(roc_auc_score(y, pa)), float(roc_auc_score(y, pr))

# ============================== Figure 2 ==============================
def figure2(y, px, pl, pm, pa, pr):
    fig, ax = plt.subplots(figsize=(8.17, 4.52))
    for nm, p, c in [("XGBoost (full)", px, PALETTE["xgb"]), ("Logistic regression", pl, PALETTE["lr"]),
                     ("MLP", pm, PALETTE["mlp"]), ("Clinical rule", pr, PALETTE["rule"]), ("Age-only", pa, PALETTE["age"])]:
        f, t, _ = roc_curve(y, p)
        ax.plot(f, t, "-", color=c, lw=1.9, label=f"{nm} — AUC {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#8C8C8C", lw=0.9, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("CKD detection — external temporal validation (NHANES 2017–2018)", loc="left", fontweight="bold", pad=8)
    ax.legend(**{**LEG, "ncol": 3})
    return _save(fig, "Figure2")

# ============================== Figure 3 ==============================
def figure3(y, px, pl, pr):
    fig, ax = plt.subplots(figsize=(5.76, 4.52))
    for nm, p, c, mk in [("XGBoost", px, PALETTE["xgb"], "o"), ("Logistic regression", pl, PALETTE["lr"], "s"),
                         ("Clinical rule", pr, PALETTE["rule"], "^")]:
        df = pd.DataFrame({"y": y, "p": p}); df["b"] = pd.qcut(df["p"], 10, duplicates="drop")
        g = df.groupby("b", observed=True).agg(mp=("p", "mean"), my=("y", "mean"))
        ax.plot(g["mp"], g["my"], marker=mk, ms=3.4, ls="-", color=c, lw=1.6, label=nm)
    ax.plot([0, 0.8], [0, 0.8], "--", color="#8C8C8C", lw=0.9, label="Perfect calibration")
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration (external validation)", loc="left", fontweight="bold", pad=8)
    ax.legend(**{**LEG, "ncol": 2})
    return _save(fig, "Figure3")

# ============================== Figure 4 ==============================
def figure4(y, px, pl, pr):
    def nb(p, pts):
        return np.array([((p >= t) & (y == 1)).sum() / len(y) - ((p >= t) & (y == 0)).sum() / len(y) * t / (1 - t) for t in pts])
    pts = np.linspace(0.02, 0.30, 30); prev = y.mean()
    fig, ax = plt.subplots(figsize=(5.90, 4.67))
    h = [Line2D([], [], color="#D9D9D9", lw=7, alpha=0.55, label="Clinically relevant range (5%–25%)")]
    ax.plot(pts, nb(px, pts), "-", color=PALETTE["xgb"], lw=1.9, label="XGBoost (full)")
    ax.plot(pts, nb(pl, pts), "--", color=PALETTE["lr"], lw=1.9, label="Logistic regression")
    ax.plot(pts, nb(pr, pts), ":", color=PALETTE["rule"], lw=1.9, label="Clinical rule")
    ax.plot(pts, np.maximum(0, prev - pts / (1 - pts) * (1 - prev)), "-.", color="#8C8C8C", lw=1.4, label="Screen all")
    ax.plot(pts, np.zeros_like(pts), "-", color="#333333", lw=1.0, label="Screen none")
    ax.axvspan(0.05, 0.25, color="#D9D9D9", alpha=0.28, lw=0, zorder=0)
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_yticks(np.arange(0.0, 0.20, 0.025))
    ax.set_yticklabels([f"{v:.3f}" for v in list(ax.get_yticks())])
    ax.set_title("Decision curve analysis (external validation)", loc="left", fontweight="bold", pad=8)
    hh, ll = ax.get_legend_handles_labels()
    ax.legend(h + hh, [x.get_label() for x in h + hh], **{**LEG, "ncol": 2})
    return _save(fig, "Figure4")

# ============================== Figure 5 ==============================
def _fam(lab):
    return ("age" if lab.startswith("Age") else "sex" if lab in ("Male", "Female")
            else "race" if lab in ("White", "Black", "Hispanic") else "pov")

ALIAS = {"Age 18–44 y": "Age 18–44 y", "Age 45–64 y": "Age 45–64 y", "Age ≥65 y": "Age ≥65 y",
         "Male": "Male", "Female": "Female", "White": "White", "Black": "Black",
         "Hispanic": "Hispanic / Mexican American", "Poverty-income ratio <1.3": "PIR < 1.3",
         "Poverty-income ratio ≥1.3": "PIR ≥ 1.3"}

def figure5():
    # 数据源（首选）：仓内 results/ JSON，round(…,3) 后与稿件 Table 3 数值一致
    # （2026-09-28 逐值核对：n/auc/lo/hi/lr 十行全同）。稿件 md（MAN）为兜底路径。
    ci_p, dca_p = os.path.join(RES, "supplementary_results.json"), os.path.join(RES, "subgroup_dca_incremental.json")
    if os.path.exists(ci_p) and os.path.exists(dca_p):
        ci = json.load(open(ci_p, encoding="utf-8"))["subgroup_auc_ci_xgb"]
        sg = json.load(open(dca_p, encoding="utf-8"))["subgroups"]
        assert len(sg) == 10, f"子组数异常: {len(sg)}"
        r3 = lambda v: round(float(v), 3)
        data = [dict(lab=ALIAS.get(sg[k]["label"], sg[k]["label"]), n=int(sg[k]["n"]),
                     auc=r3(ci[k]["auc"]), lo=r3(ci[k]["ci95"][0]), hi=r3(ci[k]["ci95"][1]),
                     lr=r3(sg[k]["auc_lr"]), fam=_fam(sg[k]["label"])) for k in sg]
        overall = r3(json.load(open(dca_p, encoding="utf-8"))["overall"]["auc_xgb"])
    else:
        md = open(MAN, encoding="utf-8").read()
        rows = re.findall(r"^\| ([^|]+?) \| ([\d,]+) \| ([\d.]+) \(([\d.]+)–([\d.]+)\) \| ([\d.]+) \|$", md, re.M)
        assert len(rows) == 10, f"Table 3 解析行数异常: {len(rows)}"
        data = []
        for lab, n, auc, lo, hi, lrauc in rows:
            lab = lab.strip()
            data.append(dict(lab=ALIAS.get(lab, lab), n=int(n.replace(",", "")), auc=float(auc),
                             lo=float(lo), hi=float(hi), lr=float(lrauc), fam=_fam(lab)))
        overall = 0.810
    fig, ax = plt.subplots(figsize=(6.76, 4.87))
    ys = np.arange(len(data))[::-1]
    for yy, d in zip(ys, data):
        c = FAMILY[d["fam"]]
        ax.plot([d["lo"], d["hi"]], [yy, yy], "-", color=c, lw=1.6, solid_capstyle="round")
        ax.plot([d["auc"]], [yy], "o", color=c, ms=4.8, zorder=3)
        ax.plot([d["lr"]], [yy], "x", color="#7F7F7F", ms=4.0, zorder=3)
        ax.text(0.858, yy, f"{d['auc']:.3f} ({d['lo']:.3f}–{d['hi']:.3f})", va="center", ha="right", fontsize=MIN_PT)
    ax.axvline(overall, color="#333333", ls="--", lw=1.0)
    ax.set_yticks(ys); ax.set_yticklabels([d["lab"] for d in data])
    ax.set_xlim(0.5, 1.0); ax.set_ylim(-0.8, len(data) - 0.15)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUC (95% CI, 1000× bootstrap)")
    ax.set_title("Subgroup discrimination — XGBoost, external validation", loc="left", fontweight="bold", pad=8)
    handles = [Line2D([], [], marker="o", ls="none", color=FAMILY[k], ms=4.8, label=v) for k, v in
               [("age", "Age group"), ("sex", "Sex"), ("race", "Race / ethnicity"), ("pov", "Poverty–income ratio")]]
    handles.append(Line2D([], [], marker="x", ls="none", color="#7F7F7F", ms=4.0, label="Logistic regression"))
    handles.append(Line2D([], [], color="#333333", ls="--", lw=1.0, label=f"Overall AUC {overall:.3f}"))
    ax.legend(handles=handles, **{**LEG, "bbox_to_anchor": (0.5, -0.135), "ncol": 6, "handlelength": 1.0, "columnspacing": 0.9})
    return _save(fig, "Figure5")

if __name__ == "__main__":
    y, px, pl, pm, test, coh = load_preds()
    pa, pr, auc_a, auc_r = baselines(y, test, coh)
    print(f"[基线] Age-only={auc_a:.4f} | 临床规则={auc_r:.4f} | XGB={roc_auc_score(y, px):.4f} | LR={roc_auc_score(y, pl):.4f} | MLP={roc_auc_score(y, pm):.4f}")
    outs = [figure1(), figure2(y, px, pl, pm, pa, pr), figure3(y, px, pl, pr), figure4(y, px, pl, pr), figure5()]
    import pymupdf
    print("[产出 + 字号自检]")
    for pdf, png in outs:
        d = pymupdf.open(pdf); sizes = []
        for pg in d:
            for b in pg.get_text("dict")["blocks"]:
                for l in b.get("lines", []):
                    for s in l["spans"]:
                        if s["text"].strip(): sizes.append(round(s["size"], 1))
        print(f"   {os.path.basename(pdf):13s} {os.path.getsize(pdf)//1024:4d} KB | span {len(sizes):3d} | 最小 {min(sizes)} pt "
              f"{'✅' if min(sizes) >= 8.0 else '❌'} | 页 {d[0].rect.width/72*25.4:.0f}×{d[0].rect.height/72*25.4:.0f} mm")
        d.close()
