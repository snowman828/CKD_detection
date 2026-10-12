# -*- coding: utf-8 -*-
"""figures_to_print_formats.py —— 由「当前矢量 PDF」派生印刷/上传用格式（可复现、可审计、可移植）

为什么需要它：本仓此前**没有** CMYK/EPS/TIFF 的生成脚本 ⇒ 那批件是仓外产物，出现了
「09-27 矢量集 vs 09-30 主集」的**版本偏斜**（同一图不同措辞）。本脚本把派生链纳入仓库，
使所有格式都从**同一份矢量 PDF** 生成，版本永远一致。

产出（每个 FigureN.pdf）：
  1) FigureN_CMYK_1200dpi.tif —— 1200 dpi 光栅 + ICC 色彩换算（RGB→CMYK，非朴素反相）
  2) FigureN_CMYK.pdf        —— 矢量 CMYK（需 ghostscript；本机若缺则**跳过并打印一键命令**，不静默）
  3) FigureN.eps 由 make_cjasn_figures.py 直接产出（matplotlib 矢量后端），本脚本不重复生成。

可移植性：ICC 与 ghostscript 均按「环境变量 → 标准安装目录（由 SystemRoot/ProgramFiles/ProgramData
等环境变量推导）→ PATH」的顺序查找，**不写死任何机器专属绝对路径**，便于第三方检出后直接复跑。

用法:
  python figures_to_print_formats.py --dir <figures 目录> [--dpi 1200] [--icc <CMYK profile.icc>]
                                     [--gs <gswin64c 路径>] [--only Figure1,Figure2] [--dry-run]

环境变量（可选）:
  CKD_CMYK_ICC      指定 ICC 文件；等价于 --icc
  CKD_GHOSTSCRIPT   指定 ghostscript 可执行文件；等价于 --gs
"""
import os, io, sys, glob, subprocess, argparse, shutil

# 首选 ICC 名（按印前通用度排序）；实际路径由系统色彩目录推导
ICC_PREFERRED = ["CoatedFOGRA39.icc", "USWebCoatedSWOP.icc", "CoatedFOGRA27.icc", "ISOcoated_v2_eci.icc"]


def _system_color_dir():
    """Windows 系统 ICC 目录（由 SystemRoot 推导，不写死盘符）。"""
    root = os.environ.get("SystemRoot") or os.environ.get("windir")
    return os.path.join(root, "System32", "spool", "drivers", "color") if root else None


def find_icc(explicit=None):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    env = os.environ.get("CKD_CMYK_ICC")
    if env and os.path.isfile(env):
        return env
    d = _system_color_dir()
    if not d or not os.path.isdir(d):
        return None
    for nm in ICC_PREFERRED:                      # 精确名优先
        p = os.path.join(d, nm)
        if os.path.isfile(p):
            return p
    hits = sorted(glob.glob(os.path.join(d, "*.icc")))   # 退路：目录内任一 ICC
    return hits[0] if hits else None


def _gs_candidates():
    """ghostscript 候选路径：环境变量 → PATH → 标准安装目录（全部由环境变量推导）。"""
    out = []
    env = os.environ.get("CKD_GHOSTSCRIPT")
    if env:
        out.append(env)
    for exe in ("gswin64c", "gswin32c", "gs"):
        w = shutil.which(exe)
        if w:
            out.append(w)
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(var)
        if root and os.path.isdir(root):
            out += sorted(glob.glob(os.path.join(root, "gs", "*", "bin", "gswin64c.exe")), reverse=True)
    for var in ("ProgramData",):
        root = os.environ.get(var)
        if root:
            out += sorted(glob.glob(os.path.join(root, "miniconda3", "pkgs",
                                                 "ghostscript-*", "Library", "bin", "gswin64c.exe")), reverse=True)
            out += sorted(glob.glob(os.path.join(root, "anaconda3", "pkgs",
                                                 "ghostscript-*", "Library", "bin", "gswin64c.exe")), reverse=True)
    return [p for p in out if p]


def _find_gs_in_conda_pkgs():
    """conda 包缓存里可能已有完整 gs（无需安装、不改系统）——按版本号从高到低取第一个可用的。

    候选根：sys.prefix、用户家目录下的常见 conda 发行版（miniconda3/anaconda3/miniforge3/mambaforge3）、
    以及 ProgramData —— 全部由环境变量/家目录推导，不写死盘符或用户名。
    """
    bases = []
    for b in (sys.prefix, os.path.dirname(sys.prefix), os.path.expanduser("~"), os.environ.get("ProgramData")):
        if b and os.path.isdir(b):
            bases.append(b)
    for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge3", "conda"):
        h = os.environ.get("USERPROFILE") or os.path.expanduser("~")
        if h:
            bases.append(os.path.join(h, d))
    pats = []
    for base in bases:
        pats.append(os.path.join(base, "pkgs", "ghostscript-*", "Library", "bin", "gswin64c.exe"))
    hits = []
    for p in pats:
        hits += glob.glob(p)
    def ver(p):
        import re
        m = re.search(r"ghostscript-(\d+)\.(\d+)\.(\d+)", p)
        return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)
    for h in sorted(hits, key=ver, reverse=True):
        try:
            r = subprocess.run([h, "--version"], capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                return h, r.stdout.strip()
        except Exception:
            continue
    return None, None


def find_gs(explicit=None):
    if explicit and os.path.isfile(explicit):
        return explicit
    for p in _gs_candidates():
        if os.path.isfile(p):
            return p
    return _find_gs_in_conda_pkgs()[0]


def convert_tiff(pdf, out, icc, dpi):
    """矢量 PDF → 1200 dpi CMYK TIFF（ICC 换算；透明底合成白底）"""
    import pymupdf
    from PIL import Image, ImageCms
    doc = pymupdf.open(pdf)
    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    rgb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    srgb = ImageCms.createProfile("sRGB")
    cmyk_profile = ImageCms.getOpenProfile(icc)
    img = ImageCms.profileToProfile(rgb, srgb, cmyk_profile, outputMode="CMYK",
                                    renderingIntent=ImageCms.Intent.PERCEPTUAL)
    img.save(out, "TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    doc.close()
    return img.size


def convert_pdf_cmyk(pdf, out, gs):
    """矢量 PDF → 矢量 CMYK PDF（ghostscript）"""
    cmd = [gs, "-dBATCH", "-dNOPAUSE", "-dQUIET", "-sDEVICE=pdfwrite",
           "-dProcessColorModel=/DeviceCMYK", "-sColorConversionStrategy=CMYK",
           "-dEmbedAllFonts=true", "-dSubsetFonts=true", "-dAutoRotatePages=/None",
           f"-sOutputFile={out}", pdf]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-300:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dpi", type=int, default=1200)
    ap.add_argument("--icc", default=None)
    ap.add_argument("--gs", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    icc = find_icc(a.icc)
    gs = find_gs(a.gs)
    pdfs = sorted(glob.glob(os.path.join(a.dir, "Figure*.pdf")))
    pdfs = [p for p in pdfs if "_CMYK" not in os.path.basename(p)]
    if a.only:
        want = set(x.strip() for x in a.only.split(","))
        pdfs = [p for p in pdfs if os.path.splitext(os.path.basename(p))[0] in want]
    print(f"[环境] ICC(CMYK) = {icc or '**缺失**（TIFF 将跳过；用 --icc 或 CKD_CMYK_ICC 指定）'}")
    print(f"[环境] ghostscript = {gs or '**缺失**（CMYK PDF 将跳过）'}")
    print(f"[目标] {len(pdfs)} 个矢量 PDF @ {a.dpi} dpi\n")
    if a.dry_run:
        for p in pdfs:
            print("  DRY", os.path.basename(p))
        sys.exit(0)
    ok = 0
    for p in pdfs:
        base = os.path.splitext(p)[0]
        tif = base + "_CMYK_%ddpi.tif" % a.dpi
        if icc:
            w, h = convert_tiff(p, tif, icc, a.dpi)
            print(f"  ✅ {os.path.basename(tif):34s} {w}×{h} px | {os.path.getsize(tif)/1e6:6.2f} MB")
            ok += 1
        else:
            print("  ⚠ 跳过 TIFF：找不到 CMYK ICC profile（用 --icc 指定）")
        if gs:
            rc, msg = convert_pdf_cmyk(p, base + "_CMYK.pdf", gs)
            print(("  ✅ " if rc == 0 else "  ❌ ") + f"{os.path.basename(base)}_CMYK.pdf  gs exit={rc} {msg if rc else ''}")
        else:
            print(f"  ⏭ 跳过 CMYK PDF（缺 ghostscript）→ 有 gs 时执行：")
            print(f"      gswin64c -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dProcessColorModel=/DeviceCMYK \\")
            print(f"               -sColorConversionStrategy=CMYK -sOutputFile=\"{os.path.basename(base)}_CMYK.pdf\" \"{os.path.basename(p)}\"")
    print(f"\n完成：TIFF {ok}/{len(pdfs)}")
