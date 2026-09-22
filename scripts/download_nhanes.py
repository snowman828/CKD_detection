"""NHANES CKD 研究数据下载脚本（真实公开数据，无湿实验管道第 1 步）
周期：2011-2012(G) / 2015-2016(I) / 2017-2018(J) / 2019-2020(K)
文件：人口学 / 尿白蛋白肌酐 / 血清生化 / 血压 / 身体测量 / 糖尿病问卷 / 糖化血红蛋白
下载至 G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data/
"""
import urllib.request, os, sys, time

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{fname}"
DATA_DIR = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/data"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    2011: ["DEMO_G", "ALB_CR_G", "BIOPRO_G", "BPX_G", "BMX_G", "DIQ_G", "GHB_G"],
    2015: ["DEMO_I", "ALB_CR_I", "BIOPRO_I", "BPX_I", "BMX_I", "DIQ_I", "GHB_I"],
    2017: ["DEMO_J", "ALB_CR_J", "BIOPRO_J", "BPX_J", "BMX_J", "DIQ_J", "GHB_J"],
    2019: ["DEMO_K", "ALB_CR_K", "BIOPRO_K", "BPX_K", "BMX_K", "DIQ_K", "GHB_K"],
}

def download(url, dest, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            print(f"  retry {i+1} {url}: {e}", flush=True)
            time.sleep(3)
    return False

ok, fail = 0, []
for year, files in FILES.items():
    for f in files:
        dest = os.path.join(DATA_DIR, f + ".XPT")
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            ok += 1; print(f"[skip] {f}.XPT", flush=True); continue
        url = BASE.format(year=year, fname=f + ".XPT")
        print(f"[get ] {f}.XPT <- {url}", flush=True)
        if download(url, dest):
            ok += 1
        else:
            fail.append(f)
            print(f"[FAIL] {f}.XPT", flush=True)

print(f"\nDONE ok={ok} fail={fail}")
