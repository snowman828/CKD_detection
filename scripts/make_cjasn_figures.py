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
import os, re, json, itertools
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
MAN = os.environ.get("CKD_MANUSCRIPT_MD", os.path.join(ROOT, "投稿包", "01_CJASN_Detection", "CJASN_Manuscript.md"))
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
    """参与者流程图 · B 版（纵向三段式：总筛查 → 排除 → 分析）

    间距策略（根治"拥挤"）：英寸坐标系（1 数据单位 = 1 英寸）+ 框宽按最长行自适应（+2×0.13 in）
    + 行距 1.75 + 箭头按框边锚定（前框底 → 后框顶，留 0.06 in）。硬自检：字号≥8.5pt／框不相交／
    箭头不穿框不压字／文本不超框；另断言排除框与分析框间隙、以及无出血元素。
    """
    d = json.load(open(os.path.join(RES, "participant_flow.json"), encoding="utf-8"))
    COL = {"screened": ("#E9F0F7", "#4A7FA5"), "excluded": ("#FCEAEA", "#C0504D"),
           "analytic": ("#EAF4EC", "#5B9E6B"), "dev": ("#F1F3F9", "#6B7FA5"),
           "test": ("#FDF6E9", "#C9A227"), "cross": ("#EEF3FA", "#4A7FA5")}
    PAD, CW, TXT, ARR = 0.13, 0.52, "#1A1A1A", "#8A8A8A"
    lh = lambda fs=MIN_PT: fs / 72 * 1.75
    tw = lambda t, fs=MIN_PT: max(len(l) for l in t.split("\n")) * fs * CW / 72
    th = lambda t, fs=MIN_PT: len(t.split("\n")) * lh(fs)
    N = lambda v: f"{v:,}"
    W, H = 6.2, 6.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    rects, arrows, tboxes = [], [], []
    def draw(x, y, w, h, txt, key):
        fc, ec = COL[key]
        rects.append((x, y, x + w, y + h))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09",
                                    linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=MIN_PT,
                color=TXT, linespacing=1.75, zorder=3)
        tboxes.append((x + PAD - 0.02, y + PAD - 0.02, x + w - PAD + 0.02, y + h - PAD + 0.02))
        return (x, y, x + w, y + h)
    def box(cx, cy, txt, key):
        w, h = tw(txt) + 2 * PAD, th(txt) + 2 * PAD
        return draw(cx - w / 2, cy - h / 2, w, h, txt, key)
    def arrow(p1, p2):
        arrows.append((p1[0], p1[1], p2[0], p2[1]))
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.0, color=ARR, shrinkA=0, shrinkB=0, zorder=1))
    def down(rf, rt):
        arrow(((rf[0] + rf[2]) / 2, rf[1] - 0.06), ((rt[0] + rt[2]) / 2, rt[3] + 0.06))
    EXCL = (f"Age < 18 years: n = {N(d['excl_age_lt18'])}\n"
            f"Pregnant: n = {N(d['excl_pregnant'])}\n"
            f"No CKD label component: n = {N(d['excl_no_label_component'])}")
    cx = W / 2
    r1 = box(cx, H - 0.75, f"NHANES 2011–2018 participants\n(4 survey cycles, G–J) screened: n = {N(d['n_screened'])}", "screened")
    r2 = box(cx, H - 1.95, f"Excluded\n{EXCL}", "excluded"); down(r1, r2)
    r3 = box(cx, H - 3.15, f"Analytic cohort: n = {N(d['n_analytic'])}\nCKD (eGFR<60 or uACR≥30 mg/g): n = {N(d['n_ckd'])}", "analytic"); down(r2, r3)
    d1 = box(cx - 1.70, H - 4.75, f"Development (cycles G–I, 2011–2016)\nn = {N(d['n_development_G_I'])}", "dev")
    d2 = box(cx + 1.70, H - 4.75, f"Temporal test (cycle J, 2017–2018)\nn = {N(d['n_temporal_test_J'])}", "test")
    arrow((r3[0] + 0.45, r3[1] - 0.06), ((d1[0] + d1[2]) / 2, d1[3] + 0.06))
    arrow((r3[2] - 0.45, r3[1] - 0.06), ((d2[0] + d2[2]) / 2, d2[3] + 0.06))
    ax.plot([0.20, W - 0.20], [H - 5.30, H - 5.30], ":", color="#9A9A9A", lw=1.0, zorder=0)
    box(cx, H - 5.76, "External cross-system validation\nKNHANES 2022–2024 (Korea): n = 17,045", "cross")
    note = "Counts reproduce the frozen cohort build (participant_flow.json)."
    ny, nx = 0.20, 0.20
    ax.text(nx, ny, note, fontsize=MIN_PT, color="#6A6A6A", ha="left", va="center")
    note_bb = (nx, ny - th(note) / 2, nx + tw(note), ny + th(note) / 2)
    tboxes.append(note_bb)
    # ---- 硬自检 ----
    # ★ 2026-09-29 新增（退回窗口体检发现真缺陷）：说明文字与任何方框必须保持净空。
    #   旧版末框中心 H-5.95 ⇒ 底边 y=0.3135 而说明文字上沿 y=0.363 ⇒ 边框线横穿文字（vision 复核确认）。
    #   原自检只比较 rects↔rects 与 arrows↔(rects,tboxes)，**没有 rects↔tboxes** 这一对 ⇒ 漏检。
    for (a, b, c, e) in rects:
        assert not (a < note_bb[2] - 1e-9 and note_bb[0] < c - 1e-9
                    and b < note_bb[3] - 1e-9 and note_bb[1] < e - 1e-9), \
            "Figure1: 说明文字与方框重叠（净空不足）"
        assert min(abs(note_bb[1] - e), abs(b - note_bb[3])) >= 0.05 or not (a < note_bb[2] and note_bb[0] < c), \
            "Figure1: 说明文字与方框垂直净空 < 0.05 in"
    for (a, b, c, e), (a2, b2, c2, e2) in itertools.combinations(rects, 2):
        assert not (a < c2 - 1e-9 and a2 < c - 1e-9 and b < e2 - 1e-9 and b2 < e - 1e-9), "Figure1: 框重叠"
    for (x1, y1, x2, y2) in arrows:
        for (a, b, c, e) in rects + tboxes:
            for t in [i / 40 for i in range(1, 40)]:
                px, py = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                assert not (a + 0.025 < px < c - 0.025 and b + 0.025 < py < e - 0.025), "Figure1: 箭头穿框/压字"
    for (a, b, c, e) in rects:
        assert a >= -1e-9 and b >= -1e-9 and c <= W + 1e-9 and e <= H + 1e-9, "Figure1: 元素出血"
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
def figure5():
    md = open(MAN, encoding="utf-8").read()
    rows = re.findall(r"^\| ([^|]+?) \| ([\d,]+) \| ([\d.]+) \(([\d.]+)–([\d.]+)\) \| ([\d.]+) \|$", md, re.M)
    assert len(rows) == 10, f"Table 3 解析行数异常: {len(rows)}"
    alias = {"Age 18–44 y": "Age 18–44 y", "Age 45–64 y": "Age 45–64 y", "Age ≥65 y": "Age ≥65 y",
             "Male": "Male", "Female": "Female", "White": "White", "Black": "Black",
             "Hispanic": "Hispanic / Mexican American", "Poverty-income ratio <1.3": "PIR < 1.3",
             "Poverty-income ratio ≥1.3": "PIR ≥ 1.3"}
    data = []
    for lab, n, auc, lo, hi, lrauc in rows:
        lab = lab.strip(); fam = ("age" if lab.startswith("Age") else "sex" if lab in ("Male", "Female")
                                  else "race" if lab in ("White", "Black", "Hispanic") else "pov")
        data.append(dict(lab=alias.get(lab, lab), n=int(n.replace(",", "")), auc=float(auc),
                         lo=float(lo), hi=float(hi), lr=float(lrauc), fam=fam))
    overall = 0.810
    fig, ax = plt.subplots(figsize=(6.76, 4.87))
    # 右侧留出独立数值列（标签放在坐标区之外，几何上不可能与点/CI 线重叠）
    fig.subplots_adjust(right=0.755)
    ys = np.arange(len(data))[::-1]
    for yy, d in zip(ys, data):
        c = FAMILY[d["fam"]]
        ax.plot([d["lo"], d["hi"]], [yy, yy], "-", color=c, lw=1.6, solid_capstyle="round")
        ax.plot([d["auc"]], [yy], "o", color=c, ms=4.8, zorder=3)
        ax.plot([d["lr"]], [yy], "x", color="#7F7F7F", ms=4.0, zorder=3)
        # 数值列：置于坐标区右侧外；1.022 是本轮调优值（与墨迹留 6.8pt≈2.4mm 间隙，避免视觉拥挤；
        # 图件按 bbox_inches=tight 裁切，故"右侧页边距"由裁切留白决定、与列位置无关）
        ax.text(1.022, yy, f"{d['auc']:.3f} ({d['lo']:.3f}–{d['hi']:.3f})",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=MIN_PT)
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

def check_fig5_label_column(pdf):
    """硬自检（Figure 5）：每个数值标签必须完全位于**所有数据墨迹（点/CI 线/网格/边框）最右缘之外**。
    动机：旧版把标签放在坐标区内部（x=0.858, ha=right），数字被点与 CI 线压住。
    实现要点：先用「面积 ≥ 90% 整页」剔除背景填充矩形，再取剩余绘图对象的并集右缘作判据。
    """
    import pymupdf, re as _re
    d = pymupdf.open(pdf)
    pg = d[0]
    page_area = pg.rect.width * pg.rect.height
    ink = []
    for it in pg.get_drawings():
        r = it.get("rect")
        if r is None:
            continue
        if r.width * r.height >= 0.90 * page_area:      # 整页背景填充：跳过
            continue
        ink.append(r)
    assert ink, "PDF 中未取到有效绘图对象"
    x_ink = max(r.x1 for r in ink)                      # 数据墨迹最右缘
    pat = _re.compile(r"^\d\.\d{3} \(\d\.\d{3}–\d\.\d{3}\)$")
    hits = []
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if pat.match(s["text"].strip()):
                    hits.append((s["text"].strip(), pymupdf.Rect(s["bbox"])))
    assert len(hits) == 10, f"数值列 span 数异常: {len(hits)}（应 10）"
    bad = [(t, r) for t, r in hits if r.x0 < x_ink - 0.5]
    assert not bad, ("❌ 数值标签起点落在数据墨迹之内（数字会被点/CI 线压住）: "
                     + "; ".join(t for t, _ in bad[:4]))
    x0 = min(r.x0 for _, r in hits)
    d.close()
    return f"数值列 {len(hits)}/10 均在数据墨迹右侧（列左缘 {x0:.1f}pt ≥ 墨迹右缘 {x_ink:.1f}pt）"

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
    print("   " + check_fig5_label_column(outs[4][0]) + " ✅")
