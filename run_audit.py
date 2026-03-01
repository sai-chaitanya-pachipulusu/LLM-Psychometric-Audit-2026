"""
run_audit.py
------------
Master orchestrator.  Run this single script to execute the full
psychometric audit pipeline end-to-end:

  Step 1 → Load dual corpus (ASAP 2.0 + LEAF)      (leaf_loader)
  Step 2 → Score essays via LLM       (llm_scorer)
  Step 3 → Run combined psychometric analysis       (psychometric_analysis)
  Step 4 → ASAP-only intersectional DIF deep-dive   (psychometric_analysis)
  Step 5 → LEAF cross-validation analysis            (psychometric_analysis)
  Step 6 → Generate all figures                      (visualizations)
  Step 7 → Write research report                     (report_generator)

Usage:
  python run_audit.py                # Mock mode (no API key needed) — ⚠ FAKE RESULTS
  SCORING_MODE=live python run_audit.py  # Live OpenAI API — REAL FINDINGS

Data sources:
  ASAP 2.0 (primary fairness anchor) – certified ELL + race/gender/SES/disability
  LEAF (external validation set)     – ETS NAACL 2024 feedback corpus

Author: Sai Chaitanya Pachipulusu (Independent Psychometric Audit, 2026)
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Improve Windows console compatibility for Unicode output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from leaf_loader           import load_asap, load_leaf
from llm_scorer            import score_corpus, SCORING_MODE
from psychometric_analysis import run_full_analysis, intersectional_dif
from visualizations        import generate_all_figures
from report_generator      import write_report

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   PSYCHOMETRIC AUDIT PIPELINE                                    ║
║   "A Psychometric Evaluation of LLM-Scored Communication Skills" ║
║   Sai Chaitanya Pachipulusu — Independent Research, 2026         ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


# Default upgraded sample configuration:
# - ASAP: 400 per group = ~800 essays
# - LEAF: ~400 essays total
# Combined total ~1,200 essays (> 800 live calls).
ASAP_SAMPLE_PER_GROUP = _env_int("ASAP_SAMPLE_PER_GROUP", 400)
LEAF_SAMPLE_TOTAL     = _env_int("LEAF_SAMPLE_TOTAL", 400)
RANDOM_SEED           = _env_int("AUDIT_RANDOM_SEED", 42)


def _load_dual_corpus(save_path: str = "data/corpus.csv") -> pd.DataFrame:
    print(
        "STEP 1/7 — Loading BOTH corpora (ASAP 2.0 + LEAF) "
        "with upgraded sample size …"
    )

    df_asap = load_asap(
        sample_n=ASAP_SAMPLE_PER_GROUP,
        random_seed=RANDOM_SEED,
    )
    if df_asap is None or df_asap.empty:
        raise RuntimeError("ASAP 2.0 load failed or returned no rows.")

    df_leaf = load_leaf(
        sample_n=LEAF_SAMPLE_TOTAL,
        random_seed=RANDOM_SEED,
    )
    if df_leaf is None or df_leaf.empty:
        raise RuntimeError("LEAF load failed or returned no rows.")

    df = pd.concat([df_asap, df_leaf], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n  [Corpus] Saved → {out.resolve()}")
    print(f"  Total essays: {len(df)}")
    print(f"  Source mix: {df['source'].value_counts().to_dict()}")
    print(f"  Group mix: {df['group'].value_counts().to_dict()}")
    return df


def _safe_sign(value: float) -> int:
    if value is None or np.isnan(value) or abs(value) < 1e-12:
        return 0
    return int(np.sign(value))


def _leaf_cross_validation(scored_df: pd.DataFrame, combined_results: dict) -> dict:
    """Run full analysis on LEAF-only subset and compare signal directionality."""
    leaf_df = scored_df[scored_df["source"] == "LEAF"].copy()
    if leaf_df.empty:
        return {"status": "skipped", "reason": "No LEAF rows in scored corpus."}

    group_counts = leaf_df["group"].value_counts()
    if len(group_counts) < 2 or group_counts.min() < 30:
        return {
            "status": "skipped",
            "reason": f"Insufficient LEAF group balance: {group_counts.to_dict()}",
        }

    leaf_results = run_full_analysis(leaf_df)

    combined_mh = float(combined_results["pillar2a_mh_dif"]["MH_D_DIF"])
    leaf_mh = float(leaf_results["pillar2a_mh_dif"]["MH_D_DIF"])
    combined_qwk = float(combined_results["pillar1_overall"]["QWK"])
    leaf_qwk = float(leaf_results["pillar1_overall"]["QWK"])
    combined_civ = bool(combined_results["pillar3_civ"]["CIV_Detected"])
    leaf_civ = bool(leaf_results["pillar3_civ"]["CIV_Detected"])

    return {
        "status": "ok",
        "leaf_n": int(len(leaf_df)),
        "leaf_group_counts": group_counts.to_dict(),
        "leaf_results": leaf_results,
        "consistency_checks": {
            "mh_direction_match": _safe_sign(combined_mh) == _safe_sign(leaf_mh),
            "mh_abs_delta": round(abs(combined_mh - leaf_mh), 4),
            "qwk_abs_delta": round(abs(combined_qwk - leaf_qwk), 4),
            "civ_match": combined_civ == leaf_civ,
        },
    }


def main():
    print(BANNER)
    t0 = time.time()

    # ── Mode warning ──────────────────────────────────
    if SCORING_MODE != "live":
        print("⚠" * 30)
        print("  WARNING: SCORING_MODE is set to MOCK.")
        print("  All AI scores will be ARTIFICIAL. Results are NOT real.")
        print("  Set SCORING_MODE=live in .env to use the OpenAI API.")
        print("⚠" * 30 + "\n")
    else:
        print("✓  SCORING_MODE = LIVE — Using real OpenAI API\n")

    # ── STEP 1: Dual-Corpus Data ───────────────────────────
    df = _load_dual_corpus(save_path="data/corpus.csv")

    # Print demographic overview
    print("\n  ── Demographic Overview ──")
    print(f"  Total essays: {len(df)}")
    for col in ["group", "ell_status", "race_ethnicity", "gender",
                 "economically_disadvantaged", "student_disability_status"]:
        if col in df.columns:
            counts = df[col].value_counts()
            print(f"  {col}: {counts.to_dict()}")

    # ── STEP 2: AI Scoring ────────────────────────────────
    print("\nSTEP 2/7 — Scoring essays with LLM …")
    df = score_corpus(df, save_path="data/scored_corpus.csv")

    # ── STEP 3: Main Combined Analysis ─────────────────────
    print("\nSTEP 3/7 — Running psychometric analyses on COMBINED corpus …")
    results = run_full_analysis(df)

    # ── STEP 4: Dedicated ASAP Intersectional Deep Dive ────
    print("\nSTEP 4/7 — Running ASAP-only intersectional DIF deep dive …")
    asap_scored = df[df["source"] == "ASAP_2.0"].copy()
    if len(asap_scored) >= 100:
        results["intersectional_asap_only"] = intersectional_dif(asap_scored)
    else:
        results["intersectional_asap_only"] = {
            "status": "skipped",
            "reason": f"ASAP subset too small (n={len(asap_scored)})",
        }

    # ── STEP 5: LEAF Cross-Validation ──────────────────────
    print("\nSTEP 5/7 — Running LEAF cross-validation analysis …")
    results["leaf_cross_validation"] = _leaf_cross_validation(df, results)

    results["audit_metadata"] = {
        "sample_config": {
            "asap_sample_per_group": ASAP_SAMPLE_PER_GROUP,
            "leaf_sample_total": LEAF_SAMPLE_TOTAL,
            "random_seed": RANDOM_SEED,
        },
        "corpus_n": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "group_counts": df["group"].value_counts().to_dict(),
    }

    # Persist results JSON for downstream use
    Path("data").mkdir(exist_ok=True)
    json_path = Path("data/analysis_results.json")
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[Pipeline] Analysis results saved → {json_path.resolve()}")

    # Pretty-print key results
    _print_summary(results)

    # ── STEP 6: Figures ────────────────────────────────────
    print("\nSTEP 6/7 — Generating publication figures …")
    generate_all_figures(df, results)

    # ── STEP 7: Research Report ────────────────────────────
    print("\nSTEP 7/7 — Writing ETS-style research report …")
    write_report(results)

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  ✓  Full audit complete in {elapsed:.1f}s")
    print(f"  Report → reports/RESEARCH_REPORT.md")
    print(f"  Figures → reports/figures/")
    print(f"  Data    → data/scored_corpus.csv")
    print(f"  Results → data/analysis_results.json")
    if SCORING_MODE != "live":
        print(f"\n  ⚠  REMINDER: Results are from MOCK MODE.")
        print(f"     Run with SCORING_MODE=live for real findings.")
    print(f"{'═' * 60}\n")


def _print_summary(results: dict):
    p1 = results["pillar1_overall"]
    p2a = results["pillar2a_mh_dif"]
    p2b = results["pillar2b_olr_dif"]
    p3 = results["pillar3_civ"]
    p4 = results["pillar4_significance"]["t_test"]

    print("\n" + "─" * 60)
    print("  AUDIT SUMMARY — Core Pillars")
    print("─" * 60)
    print(f"  PILLAR 1 · QWK (Human–AI Agreement) : {p1['QWK']:.3f}")
    print(f"  PILLAR 2a· MH D-DIF (ELL)           : {p2a['MH_D_DIF']:.3f}  [{p2a['ETS_Class']}]")
    if p2b:
        udif = "DETECTED ⚠" if p2b.get("Uniform_DIF_Detected") else "None ✓"
        nudif = "DETECTED ⚠" if p2b.get("NonUniform_DIF_Detected") else "None ✓"
        print(f"  PILLAR 2b· OLR DIF (Uniform/Non-U)  : {udif} / {nudif}")
    print(f"  PILLAR 3 · CIV (Word Count Bias)    : {'DETECTED ⚠' if p3['CIV_Detected'] else 'Not detected ✓'}")
    print(f"  PILLAR 4 · Cohen's d                : {p4['Cohens_d']:.3f}  [{p4['Effect_Magnitude']} effect]  p={p4['p_value']:.4f}")

    # Intersectional DIF summary
    p5 = results.get("pillar5_intersectional", {})
    if p5:
        print("\n  ── Intersectional DIF (★ NOVEL) ──")
        for axis, axis_results in p5.items():
            for focal, r in axis_results.items():
                print(f"    {axis}/{focal}: MH D-DIF = {r['MH_D_DIF']:.3f} [{r['ETS_Class']}]")

    asap_p5 = results.get("intersectional_asap_only", {})
    if isinstance(asap_p5, dict) and asap_p5 and "status" not in asap_p5:
        print("\n  ── ASAP-only Intersectional Deep Dive ──")
        for axis, axis_results in asap_p5.items():
            for focal, r in axis_results.items():
                print(f"    {axis}/{focal}: MH D-DIF = {r['MH_D_DIF']:.3f} [{r['ETS_Class']}]")

    # Sub-dimensional DIF summary
    p6 = results.get("pillar6_subdimensional", {})
    p6_mh = p6.get("mh_dif_by_subdimension", {}) if isinstance(p6, dict) else {}
    if p6_mh:
        print("\n  ── Sub-Dimensional DIF (★ NOVEL) ──")
        for dim, r in p6_mh.items():
            print(f"    {dim}: MH D-DIF = {r['MH_D_DIF']:.3f} [{r['ETS_Class']}]")

    p7 = results.get("pillar7_clearys_rule", {})
    if p7:
        print("\n  ── Cleary's Rule & ETS SMD (★ GOLD STANDARD) ──")
        for group, r in p7.items():
            viol = "DETECTED ⚠" if r['Clearys_Rule_Violated'] else "Passed ✓"
            smd_warn = "⚠" if r.get('SMD_Exceeds_ETS_Threshold') else "✓"
            print(f"    [{group}]: Predictive Bias: {viol} | Residual SMD: {r.get('SMD_Residuals', 0):.3f} {smd_warn}")

    leaf_cv = results.get("leaf_cross_validation", {})
    if leaf_cv.get("status") == "ok":
        chk = leaf_cv["consistency_checks"]
        print("\n  ── LEAF Cross-Validation ──")
        print(
            "    "
            f"MH direction match={chk['mh_direction_match']}, "
            f"|ΔMH|={chk['mh_abs_delta']:.3f}, "
            f"|ΔQWK|={chk['qwk_abs_delta']:.3f}, "
            f"CIV match={chk['civ_match']}"
        )
    elif leaf_cv:
        print(f"\n  ── LEAF Cross-Validation ──\n    {leaf_cv.get('reason', 'Skipped')}")

    print("─" * 60)


if __name__ == "__main__":
    main()
