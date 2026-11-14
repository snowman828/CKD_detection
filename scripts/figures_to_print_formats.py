# -*- coding: utf-8 -*-
"""Derive print and upload formats from the current vector PDFs (reproducible, auditable, portable).

Rationale: these formats had no generator in the repository, so the print files were produced outside it and drifted
from the vector set. This script brings the derivation into the repository so every format
is generated from the same vector PDF and cannot diverge.

Outputs (per FigureN.pdf):
  1) FigureN_CMYK_1200dpi.tif - 1200 dpi raster with ICC RGB-to-CMYK conversion
  2) FigureN_CMYK.pdf - vector CMYK via ghostscript; if ghostscript is absent the step is skipped and the exact command printed
  3) FigureN.eps is produced directly by the figure scripts and is not regenerated here.

Portability: ICC profile and ghostscript are resolved from environment variables, then standard install roots
and PATH; no machine-specific absolute path is hard-coded.

Usage:
  python figures_to_print_formats.py --dir <figures 目录> [--dpi 1200] [--icc <CMYK profile.icc>]
                                     [--gs <gswin64c 路径>] [--only Figure1,Figure2] [--dry-run]

Optional environment variables:
  CKD_CMYK_ICC      ICC file; equivalent to --icc
  CKD_GHOSTSCRIPT   ghostscript executable; equivalent to --gs
"""
import os, io, sys, glob, subprocess, argparse, shutil

# preferred ICC profiles in order of general use; the actual path is derived from the system colour directory
ICC_PREFERRED = ["CoatedFOGRA39.icc", "USWebCoatedSWOP.icc", "CoatedFOGRA27.icc", "ISOcoated_v2_eci.icc"]


def _system_color_dir():
    """System ICC directory on Windows, derived from SystemRoot."""
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
    for nm in ICC_PREFERRED:                      # exact names first
        p = os.path.join(d, nm)
        if os.path.isfile(p):
            return p
    hits = sorted(glob.glob(os.path.join(d, "*.icc")))   # fallback: any ICC in the directory
    return hits[0] if hits else None


def _gs_candidates():
    """ghostscript candidates: environment variable, PATH, then standard install roots."""
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
    """A complete ghostscript may already exist in a conda package cache; take the highest usable version.

    Candidate roots: sys.prefix, common conda distributions under the home directory,
    and ProgramData; all derived from environment variables rather than hard-coded drives or usernames.
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
    """Vector PDF to 1200 dpi CMYK TIFF (ICC conversion; transparency flattened onto white)."""
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
    """Vector PDF to vector CMYK PDF via ghostscript."""
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
    print(f"[env] ICC (CMYK) = {icc or '**missing** (TIFF will be skipped; set --icc or CKD_CMYK_ICC)'}")
    print(f"[env] ghostscript = {gs or '**missing** (CMYK PDF will be skipped)'}")
    print(f"[目标] {len(pdfs)}  vector PDFs at {a.dpi} dpi\n")
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
            print("  skipping TIFF: no CMYK ICC profile found (pass --icc)")
        if gs:
            rc, msg = convert_pdf_cmyk(p, base + "_CMYK.pdf", gs)
            print(("  ✅ " if rc == 0 else "  ❌ ") + f"{os.path.basename(base)}_CMYK.pdf  gs exit={rc} {msg if rc else ''}")
        else:
            print(f"  skipping CMYK PDF (no ghostscript); run this when gs is available:")
            print(f"      gswin64c -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dProcessColorModel=/DeviceCMYK \\")
            print(f"               -sColorConversionStrategy=CMYK -sOutputFile=\"{os.path.basename(base)}_CMYK.pdf\" \"{os.path.basename(p)}\"")
    print(f"\ndone: TIFF {ok}/{len(pdfs)}")
