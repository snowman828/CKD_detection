# -*- coding: utf-8 -*-
"""Graphical Abstract 生成（重试版）—— Gemini API 多模型轮换 + 指数退避
429/500 时换模型重试；key 从环境变量读；SOCKS5 10808
"""
import os, json, base64, urllib.request, socks, socket, sys, time

key = os.environ.get("GEMINI_IMAGE_KEY")
if not key:
    print("ERR: GEMINI_IMAGE_KEY 未设置"); sys.exit(1)
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 10808)
socket.socket = socks.socksocket

MODELS = ["gemini-3-pro-image-preview", "gemini-3-pro-image", "nano-banana-pro-preview",
          "gemini-2.5-flash-image", "gemini-3.1-flash-image-preview"]

PROMPT = """Professional medical graphical abstract for a top-tier journal (Nature-series style), landscape orientation, left-to-right narrative flow, flat 2.5D vector illustration, clean white background with very light cool-gray gradient, no clutter.

LEFT THIRD — INPUT: non-kidney-specific routine clinical data. A subtle human silhouette in light gray outline. Four floating source icons arranged vertically around it: (1) blood pressure cuff with small heart pulse line [color #D62728 red], (2) glucose meter with droplet [color #FF7F0E orange], (3) lipid panel tube with cholesterol molecule [color #1F77B4 blue], (4) demographic identity card with person glyph [color #7F7F7F gray]. A thin label-less arrow streams these icons rightward into the model.

CENTER THIRD — MODEL: a minimal decision-tree ensemble / neural network node cluster in cool white-blue (#EAF2FA), with an abstract fingerprint motif emerging from it: a rounded fingerprint outline whose ridge loops are colored in the four pathway colors (red hemodynamic, orange glycotoxic, blue metabolic, gray demographic), representing the CKM (cardiovascular-kidney-metabolic) pathophysiological footprint encoded from routine data.

RIGHT THIRD — OUTCOME: split into two connected panels. (a) Kidney: left kidney healthy smooth surface, right kidney with subtle granular texture and small warning dots, separated by a thin vertical divider; above them a small detection score badge showing "AUC 0.865" in dark gray with a green up-arrow (albuminuria-blind population outperforms). (b) Heart: clean heart icon with small ECG line, above it a badge "C-index 0.85" with a heart-rate symbol. A soft rounded bracket links kidney and heart panels labeled implicitly by a small CKM acronym badge (cardiovascular-kidney-metabolic) in muted teal.

STYLE: flat professional medical illustration, consistent soft shadows, muted academic palette (red #D62728, orange #FF7F0E, blue #1F77B4, gray #7F7F7F, teal #2CA6A4, white background), high resolution, crisp edges, publication quality.

NEGATIVE PROHIBITIONS: absolutely no readable text except the two numeric badges; no garbled letters; no handwriting; no cartoon faces; no comic style; no photorealism; no 3D render gloss; no drop-shadow text; no red-alert alarm aesthetics; no medical crosses; no clutter; no watermark; no logo."""

OUT = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/results/m2/graphical_abstract_v1.png"

for attempt, model in enumerate(MODELS):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": PROMPT}]}],
               "generationConfig": {"responseModalities": ["IMAGE"]}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    print(f"[try {attempt+1}/{len(MODELS)}] model={model}", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=540) as r:
            data = json.loads(r.read())
        parts = data["candidates"][0]["content"]["parts"]
        b64 = next(p["inlineData"]["data"] for p in parts if "inlineData" in p)
        with open(OUT, "wb") as f:
            f.write(base64.b64decode(b64))
        print("SAVED:", OUT, os.path.getsize(OUT), "bytes | model:", model)
        sys.exit(0)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:300]
        print(f"  HTTP {e.code}: {body}", flush=True)
        if e.code == 429:
            wait = 45 * (attempt + 1)
            print(f"  quota/rate-limit → wait {wait}s", flush=True)
            time.sleep(wait)
        elif e.code in (400, 404):
            print("  model 不可用，换下一个", flush=True)
            time.sleep(5)
        else:
            time.sleep(15)
    except Exception as e:
        print("  ERR:", type(e).__name__, str(e)[:200], flush=True)
        time.sleep(10)

print("ALL MODELS FAILED")
sys.exit(1)
