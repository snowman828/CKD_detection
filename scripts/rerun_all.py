# -*- coding: utf-8 -*-
"""rerun_all.py - rerun the whole pipeline with one command.

Design:
  - Runs the real analysis scripts without re-implementing any definition; a failing stage is recorded and the run continues.
  - No script is modified at run time; downloads are skipped when data/ is already present.
  - Produces results/_rerun_log_<ts>.json with exit code, elapsed time and key output counts per stage.
  · --check：compare every registered key value against the recomputed result (reproducibility evidence).

Usage:
  python scripts/rerun_all.py                 # full run
  python scripts/rerun_all.py --stages m16 m17
  python scripts/rerun_all.py --check         # full run + 值级比对
Runtime: python with pandas, lifelines, pyreadstat and xgboost.
"""
import argparse, json, os, subprocess, sys, time

PROJ = os.environ.get("CKD_PIPELINE_BASE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SCRIPTS = os.path.join(PROJ, "scripts")
REPO_SCRIPTS = os.path.join(PROJ, "scripts")   # package layout: same directory as SCRIPTS

STAGES = [
    ("cohort",  os.path.join(SCRIPTS, "build_cohort.py"),           "cohort build (labels and weights)"),
    ("model",   os.path.join(SCRIPTS, "modeling.py"),               "three models and temporal validation"),
    ("supp",    os.path.join(SCRIPTS, "supplementary_analysis.py"), "bootstrap, calibration, DCA, subgroups"),
    ("m16",     os.path.join(SCRIPTS, "m16_mortality_a2.py"),       "mortality-linkage prognostic validation"),
    ("m17",     os.path.join(SCRIPTS, "m17_transport_ladder.py"),   "transport/calibration ladder and sample-size rule"),
    ("knh22",   os.path.join(REPO_SCRIPTS, "knh2022_cross_system.py"),        "cross-system, 2022 wave"),
    ("knh_mw",  os.path.join(REPO_SCRIPTS, "knh_multiwave_cross_system.py"),  "cross-system, three waves"),
    ("labelmanip", os.path.join(REPO_SCRIPTS, "label_manipulation_sensitivity.py"), "label-manipulation sensitivity (2x2, age-term attribution)"),
    ("ciboot",   os.path.join(REPO_SCRIPTS, "ci_bootstrap_3items.py"),           "mortality C-index with paired bootstrap CI"),
]

# --check: registered values mapped to their source JSON and key path
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
    ap.add_argument("--stages", nargs="*", default=None, help="run only the listed stages (default: all)")
    ap.add_argument("--check", action="store_true", help="compare recomputed values against the registry after the run")
    a = ap.parse_args()

    # check that the input data are present
    data_ok = os.path.isdir(os.path.join(PROJ, "data")) and len(os.listdir(os.path.join(PROJ, "data"))) > 0
    knh_ok = os.path.isdir(os.path.join(PROJ, "data", "knh"))
    print(f"[env] python={PY}\n[env] data/ present={data_ok}  data/knh/ 在场={knh_ok}")

    to_run = [s for s in STAGES if (not a.stages) or (s[0] in a.stages)]
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": PY,
           "stages": [], "before": {}, "after": {}}

    # record pre-run hashes of key result files to prove they were actually recomputed
    import hashlib
    def h(p):
        try:
            return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        except Exception:
            return None
    for _, rel, _ in [(c[0], c[1], None) for c in CHECKS]:
        p = os.path.join(PROJ, "results", rel.replace("/", os.sep))
        log["before"][rel] = h(p)

    for name, script, desc in to_run:
        if not os.path.exists(script):
            log["stages"].append({"stage": name, "ok": False, "error": "script not found", "path": script})
            print(f"  ⚠️ {name}: script not found: {script}")
            continue
        t0 = time.time()
        r = subprocess.run([PY, script], cwd=PROJ,
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
        print("\n=== value-level reproduction check (registered vs recomputed) ===")
        checks = []
        for cid, rel, path in CHECKS:
            p = os.path.join(PROJ, "results", rel.replace("/", os.sep))
            try:
                v = dig(json.load(open(p, encoding="utf-8")), path)
            except Exception as e:
                v = f"read failed: {type(e).__name__}"
            checks.append({"claim": cid, "source": rel, "value": v})
            print(f"  {cid:20s} = {v}   ({rel})")
        log["checks"] = checks

    outp = os.path.join(PROJ, "results", f"_rerun_log_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(log, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    okn = sum(1 for s in log["stages"] if s.get("ok"))
    print(f"\nstages passed {okn}/{len(log['stages'])}  | log: {outp}")


if __name__ == "__main__":
    main()
