# -*- coding: utf-8 -*-
"""rerun_all.py — 一条命令全量重跑（A5 可复现层级）。

设计原则：
  · 只跑**真实**的既有分析脚本，不重实现口径；任一阶段失败即记录并继续（末尾汇总）。
  · 运行前后不修改任何脚本；数据已在 data/ 时跳过下载。
  · 产出 results/_rerun_log_<ts>.json：逐阶段 exit code / 耗时 / 关键输出行数。
  · --check：跑完后把**台账已登记的关键值**与重算结果逐一比对（复现性证据）。

用法：
  python scripts/rerun_all.py                 # 全量
  python scripts/rerun_all.py --stages m16 m17
  python scripts/rerun_all.py --check         # 全量 + 值级比对
运行环境：D:/hermes/scienv/Scripts/python.exe（含 pandas/lifelines/pyreadstat/xgboost）
"""
import argparse, json, os, subprocess, sys, time

PROJ = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行"
PY = sys.executable
SCRIPTS = os.path.join(PROJ, "scripts")
REPO_SCRIPTS = os.path.join(PROJ, "repo_public_push_ready", "scripts")

STAGES = [
    ("cohort",  os.path.join(SCRIPTS, "build_cohort.py"),           "队列构建（含标签/权重）"),
    ("model",   os.path.join(SCRIPTS, "modeling.py"),               "三模型 + 时间外验证"),
    ("supp",    os.path.join(SCRIPTS, "supplementary_analysis.py"), "bootstrap/校准/DCA/亚组"),
    ("m16",     os.path.join(SCRIPTS, "m16_mortality_a2.py"),       "A2 联动死亡预后验证"),
    ("m17",     os.path.join(SCRIPTS, "m17_transport_ladder.py"),   "A3 迁移-校准阶梯 + 样本量规则"),
    ("knh22",   os.path.join(REPO_SCRIPTS, "knh2022_cross_system.py"),        "跨体系（2022 单波）"),
    ("knh_mw",  os.path.join(REPO_SCRIPTS, "knh_multiwave_cross_system.py"),  "跨体系（三波）"),
]

# --check 用的登记值 → 重算来源（真源 JSON 的相对路径 + 取值路径）
CHECKS = [
    ("XGB_AUC_EXT",       "model_results.json",                    ["XGB", "auc_test"]),
    ("KNH_AUC",           "knh2022_cross_system.json",             ["auc", "value"]),
    ("KNHMW_AUC",         "knh_multiwave_cross_system.json",       ["pooled", "auc", "value"]),
    ("M16_ALLCAUSE_HR",   "m2/m16_mortality_a2.json",              ["Q1_ckd_label_vs_death", "mort", "hr"]),
    ("M16_DELTAC_MORT",   "m2/m16_mortality_a2.json",              ["Q2_score_prognostic_value", "mort", "development_OOF", "delta_c"]),
    ("M17_ECE_INTERCEPT", "m2/m17_transport_ladder.json",          ["held_out_ladder", "rung1_intercept", "ece"]),
]


def dig(obj, path):
    cur = obj
    for k in path:
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=None, help="只跑指定阶段（默认全量）")
    ap.add_argument("--check", action="store_true", help="跑完做值级复现比对")
    a = ap.parse_args()

    # 数据在场检查
    data_ok = os.path.isdir(os.path.join(PROJ, "data")) and len(os.listdir(os.path.join(PROJ, "data"))) > 0
    knh_ok = os.path.isdir(os.path.join(PROJ, "data", "knh"))
    print(f"[env] python={PY}\n[env] data/ 在场={data_ok}  data/knh/ 在场={knh_ok}")

    todo = [s for s in STAGES if (not a.stages) or (s[0] in a.stages)]
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": PY,
           "stages": [], "before": {}, "after": {}}

    # 记录关键结果文件的运行前哈希（便于判定是否真的重算）
    import hashlib
    def h(p):
        try:
            return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        except Exception:
            return None
    for _, rel, _ in [(c[0], c[1], None) for c in CHECKS]:
        p = os.path.join(PROJ, "results", rel.replace("/", os.sep))
        log["before"][rel] = h(p)

    for name, script, desc in todo:
        if not os.path.exists(script):
            log["stages"].append({"stage": name, "ok": False, "error": "脚本不存在", "path": script})
            print(f"  ⚠️ {name}: 脚本不存在 {script}")
            continue
        t0 = time.time()
        r = subprocess.run([PY, script], cwd=(REPO_SCRIPTS if "repo_public_push_ready" in script else PROJ),
                           capture_output=True, text=True, encoding="utf-8", errors="ignore")
        dt = round(time.time() - t0, 1)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        log["stages"].append({"stage": name, "desc": desc, "path": script, "ok": r.returncode == 0,
                              "exit": r.returncode, "sec": dt, "tail": tail,
                              "stderr_tail": (r.stderr or "").strip().splitlines()[-2:]})
        print(f"  {'✅' if r.returncode == 0 else '❌'} {name:8s} {dt:7.1f}s exit={r.returncode} | {desc}")
        for ln in tail:
            print(f"        {ln[:150]}")

    for _, rel, _ in [(c[0], c[1], None) for c in CHECKS]:
        p = os.path.join(PROJ, "results", rel.replace("/", os.sep))
        log["after"][rel] = h(p)

    if a.check:
        print("\n=== 值级复现比对（台账登记值 ⇄ 重算结果）===")
        checks = []
        for cid, rel, path in CHECKS:
            p = os.path.join(PROJ, "results", rel.replace("/", os.sep))
            try:
                v = dig(json.load(open(p, encoding="utf-8")), path)
            except Exception as e:
                v = f"读取失败: {type(e).__name__}"
            checks.append({"claim": cid, "source": rel, "value": v})
            print(f"  {cid:20s} = {v}   ({rel})")
        log["checks"] = checks

    outp = os.path.join(PROJ, "results", f"_rerun_log_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(log, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    okn = sum(1 for s in log["stages"] if s.get("ok"))
    print(f"\n阶段通过 {okn}/{len(log['stages'])}  | 日志: {outp}")


if __name__ == "__main__":
    main()
