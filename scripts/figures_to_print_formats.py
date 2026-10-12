# -*- coding: utf-8 -*-
"""figures_to_print_formats.py —— 由「当前矢量 PDF」派生印刷/上传用格式（可复现、可审计）

为什么需要它：本仓此前**没有** CMYK/EPS/TIFF 的生成脚本 ⇒ 那批件是仓外产物，出现了
「09-27 矢量集 vs 09-30 主集」的**版本偏斜**（同一图不同措辞）。本脚本把派生链纳入仓库，
使所有格式都从**同一份矢量 PDF** 生成，版本永远一致。

产出（每个 FigureN.pdf）：
  1) FigureN_CMYK_1200dpi.tif —— 1200 dpi 光栅 + ICC 色彩换算（RGB→CMYK，非朴素反相）
  2) FigureN_CMYK.pdf        —— 矢量 CMYK（需 ghostscript；本机若缺则**跳过并打印一键命令**，不静默）
  3) FigureN.eps 由 make_cjasn_figures.py 直接产出（matplotlib 矢量后端），本脚本不重复生成。

用法:
  python figures_to_print_formats.py --dir <figures 目录> [--dpi 1200] [--icc <CMYK profile.icc>]
                                     [--only Figure1,Figure2] [--dry-run]
"""
import os, io, sys, glob, subprocess, argparse, shutil

DEF_ICC_CANDIDATES = [
    r"C:\Windows\System32\spool\drivers\color\CoatedFOGRA39.icc",   # ISO Coated v2 300% (ECI) —— 通用印前
    r"C:\Windows\System32\spool\drivers\color\USWebCoatedSWOP.icc",
    r"C:\Windows\System32\spool\drivers\color\CoatedFOGRA27.icc",
]
GS_CANDIDATES = [
    r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe",
    r"C:\Program Files\gs\gs10.04.0\bin\gswin64c.exe",
    r"C:\Users\AIcncc\miniconda3\Library\bin\gswin64c.exe",
]


def _find_gs_in_conda_pkgs():
    """conda 包缓存里可能已有完整 gs（无需安装、不改系统）——按版本号从高到低取第一个可用的。"""
    import glob as _glob
    pats = [
        os.path.expanduser(r"~\miniconda3\pkgs\ghostscript-*\Library\bin\gswin64c.exe"),
        os.path.expanduser(r"~\anaconda3\pkgs\ghostscript-*\Library\bin\gswin64c.exe"),
        r"C:\ProgramData\miniconda3\pkgs\ghostscript-*\Library\bin\gswin64c.exe",
    ]
    hits = []
    for p in pats:
        hits += _glob.glob(p)
    # 版本号降序（ghostscript-10.08.0-... 优先于 9.x）
    def ver(p):
        m = __import__("re").search(r"ghostscript-(\d+)\.(\d+)\.(\d+)", p)
        return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)
    for h in sorted(hits, key=ver, reverse=True):
        try:
            r = subprocess.run([h, "--version"], capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                return h, r.stdout.strip()
        except Exception:
            continue
    return None, None


def find_gs():
    for p in GS_CANDIDATES:
        if os.path.isfile(p):
            return p
    for root in (r"C:\Program Files\gs", r"C:\Program Files (x86)\gs"):
        if os.path.isdir(root):
            for d in sorted(os.listdir(root), reverse=True):
                c = os.path.join(root, d, "bin", "gswin64c.exe")
                if os.path.isfile(c):
                    return c
    return _find_gs_in_conda_pkgs()[0] or shutil.which("gswin64c") or shutil.which("gs")


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
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    icc = a.icc or next((p for p in DEF_ICC_CANDIDATES if os.path.isfile(p)), None)
    gs = find_gs()
    pdfs = sorted(glob.glob(os.path.join(a.dir, "Figure*.pdf")))
    pdfs = [p for p in pdfs if "_CMYK" not in os.path.basename(p)]
    if a.only:
        want = set(x.strip() for x in a.only.split(","))
        pdfs = [p for p in pdfs if os.path.splitext(os.path.basename(p))[0] in want]
    print(f"[环境] ICC(CMYK) = {icc}")
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
            print(f"  ⚠ 跳过 TIFF：找不到 CMYK ICC profile（用 --icc 指定）")
        if gs:
            rc, msg = convert_pdf_cmyk(p, base + "_CMYK.pdf", gs)
            print(("  ✅ " if rc == 0 else "  ❌ ") + f"{os.path.basename(base)}_CMYK.pdf  gs exit={rc} {msg if rc else ''}")
        else:
            print(f"  ⏭ 跳过 CMYK PDF（缺 ghostscript）→ 有 gs 时执行：")
            print(f"      gswin64c -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dProcessColorModel=/DeviceCMYK \\")
            print(f"               -sColorConversionStrategy=CMYK -sOutputFile=\"{os.path.basename(base)}_CMYK.pdf\" \"{os.path.basename(p)}\"")
    print(f"\n完成：TIFF {ok}/{len(pdfs)}")
