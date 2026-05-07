"""
Phase 17 -- Comparison: blind-discovered winners vs reference baselines.

Reads:
  - simulations/results/raw/phase17/{bennett,drkpv}/state.json
  - Phase 14B BIN_mh2000 reference (hardcoded specs)
  - Phase 9B DRKPV defaults (hardcoded specs)

Produces:
  - simulations/results/summaries/phase17_comparison.md
  - docs/archive/phase17_archive.md (structured archive for Chat R)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "phase17"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"
_ARCHIVE_DIR = _BASE_DIR / "docs" / "archive"

# Reference baselines (hardcoded from project history)
BIN_MH2000_BASELINE = {
    "method": "BennettIndepFunctionalDR",
    "name": "BIN_mh2000",
    "params": {
        "m_h": 2000, "m_c": 500,
        "ell_h": 2.75, "ell_c": 2.0,
        "lambda_h": 1e-5, "gamma_critic": 1e-4,
        "lambda_r": 1e-2, "n_features_r": 500, "ell_scale_r": 3.5,
    },
    "history": "Phase 14B Wave 5 disjoint validation winner",
    "validated_metrics": {
        "DGP2_n2000_SNR95": {"bse_ref": 0.0065, "cov_ref": 0.97, "SER": 0.99},
    },
}

PHASE9B_DRKPV_DEFAULTS = {
    "method": "DRKernel + KPVPolicyBridgeQ + KPVBridgeH",
    "name": "Phase9B_defaults",
    "params": {
        "lambda_1": 3e-5, "lambda_2": 3e-5,
        "ell_scale": 3.5,
        "lambda_Q": 1e-3,
        "clip": None,
    },
    "history": "Phase 9B+ benchmark, never properly tuned",
    "validated_metrics": {
        "DGP2_n2000_SNR95": {"bse_ref": 0.38, "cov_ref": 0.94, "SER": 0.94},
    },
}


def _load_state(method: str) -> Dict:
    path = _RAW_DIR / method / "state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stage_history(method: str) -> Dict[int, Dict]:
    """Load each stage's report markdown if present."""
    out = {}
    for stage in range(1, 6):
        path = _SUMM_DIR / f"phase17_{method}_stage{stage}.md"
        if path.exists():
            out[stage] = path.read_text(encoding="utf-8")
    return out


def _format_config_diff(baseline_params: Dict, new_params: Dict) -> str:
    """Format a side-by-side diff of params."""
    keys = sorted(set(baseline_params.keys()) | set(new_params.keys()))
    lines = ["| Param | Baseline | Phase 17 winner | Match? |",
             "|-------|----------|-----------------|--------|"]
    for k in keys:
        b = baseline_params.get(k, "—")
        n = new_params.get(k, "—")
        match = "✓" if b == n else "**DIFFERS**"
        lines.append(f"| {k} | {b} | {n} | {match} |")
    return "\n".join(lines)


def write_comparison_report():
    """Master comparison: blind winners vs baselines."""
    bennett_state = _load_state("bennett")
    drkpv_state = _load_state("drkpv")

    import datetime
    lines = []
    ap = lines.append

    ap("# Phase 17 — Comparison Report (Blind Tuning vs Baselines)")
    ap(f"*Generated {datetime.date.today().isoformat()}*")
    ap("")
    ap("This report compares the blind-discovered winners (Phase 17) to the")
    ap("project's reference baselines (Phase 14B BIN_mh2000 for Bennett,")
    ap("Phase 9B defaults for DRKPV).")
    ap("")
    ap("**IMPORTANT**: BIN_mh2000 remains the official baseline of the essay.")
    ap("Phase 17 winners are exploratory — documented but NOT replacing baselines.")
    ap("")
    ap("---")
    ap("")

    # ── BENNETT ────────────────────────────────────────────────────────────
    ap("## Bennett")
    ap("")
    ap(f"Phase 14B baseline: **{BIN_MH2000_BASELINE['name']}**")
    ap(f"History: {BIN_MH2000_BASELINE['history']}")
    ap("")
    ap("Validated metrics on DGP2 n=2000 SNR=0.95:")
    for k, v in BIN_MH2000_BASELINE["validated_metrics"]["DGP2_n2000_SNR95"].items():
        ap(f"- {k}: {v}")
    ap("")

    if bennett_state and bennett_state.get("winner"):
        winner = bennett_state["winner"]
        ap(f"**Phase 17 blind winner**:")
        ap(f"Compute: {bennett_state.get('compute_used_minutes', 0):.1f} min")
        ap(f"Envelope: {bennett_state.get('envelope', 'unknown')}")
        ap("")
        new_params = winner.get("config", {})
        ap("### Config diff")
        ap("")
        ap(_format_config_diff(BIN_MH2000_BASELINE["params"], new_params))
        ap("")
        # Identify if winner == BIN_mh2000
        keys_match = all(
            BIN_MH2000_BASELINE["params"].get(k) == new_params.get(k)
            for k in BIN_MH2000_BASELINE["params"]
        )
        if keys_match:
            ap("**Verdict: Phase 17 RE-CONFIRMS BIN_mh2000.**")
            ap("→ A practitioner using only blind metrics would have arrived at the same config.")
        else:
            ap("**Verdict: Phase 17 found a DIFFERENT config.**")
            ap("→ Document as alternative; baseline still BIN_mh2000.")
    else:
        ap("**Phase 17 Bennett tuning incomplete — no winner declared.**")
    ap("")

    # ── DRKPV ──────────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## DRKPV")
    ap("")
    ap(f"Phase 9B baseline: **{PHASE9B_DRKPV_DEFAULTS['name']}**")
    ap(f"History: {PHASE9B_DRKPV_DEFAULTS['history']}")
    ap("")
    ap("Validated metrics on DGP2 n=2000 SNR=0.95:")
    for k, v in PHASE9B_DRKPV_DEFAULTS["validated_metrics"]["DGP2_n2000_SNR95"].items():
        ap(f"- {k}: {v}")
    ap("")

    if drkpv_state and drkpv_state.get("winner"):
        winner = drkpv_state["winner"]
        ap(f"**Phase 17 blind winner**:")
        ap(f"Compute: {drkpv_state.get('compute_used_minutes', 0):.1f} min")
        ap(f"Envelope: {drkpv_state.get('envelope', 'unknown')}")
        ap("")
        new_params = winner.get("config", {})
        ap("### Config diff")
        ap("")
        ap(_format_config_diff(PHASE9B_DRKPV_DEFAULTS["params"], new_params))
        ap("")
        ap("**Verdict: this is the FIRST tuning of DRKPV.** Recommended replacement of Phase 9B defaults.")
    else:
        ap("**Phase 17 DRKPV tuning incomplete — no winner declared.**")
    ap("")

    # ── Summary ────────────────────────────────────────────────────────────
    ap("---")
    ap("")
    ap("## Summary table")
    ap("")
    ap("| Method | Baseline | Phase 17 Winner | Conclusion |")
    ap("|--------|----------|-----------------|------------|")

    if bennett_state and bennett_state.get("winner"):
        win = bennett_state["winner"]
        score = win.get("mean_score", win.get("score", float("nan")))
        new_params = win.get("config", {})
        keys_match = all(BIN_MH2000_BASELINE["params"].get(k) == new_params.get(k)
                          for k in BIN_MH2000_BASELINE["params"])
        verdict = "RE-CONFIRMED" if keys_match else "ALTERNATIVE found"
        ap(f"| Bennett | BIN_mh2000 | score={score:.4f} | {verdict} |")
    else:
        ap(f"| Bennett | BIN_mh2000 | (incomplete) | — |")

    if drkpv_state and drkpv_state.get("winner"):
        win = drkpv_state["winner"]
        score = win.get("mean_score", win.get("score", float("nan")))
        ap(f"| DRKPV | Phase9B defaults | score={score:.4f} | New baseline |")
    else:
        ap(f"| DRKPV | Phase9B defaults | (incomplete) | — |")

    out_path = _SUMM_DIR / "phase17_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


def write_archive():
    """Structured archive for Chat R consumption."""
    bennett_state = _load_state("bennett")
    drkpv_state = _load_state("drkpv")
    bennett_stages = _load_stage_history("bennett")
    drkpv_stages = _load_stage_history("drkpv")

    import datetime
    lines = []
    ap = lines.append

    ap("# Phase 17 Archive — Adaptive Blind Tuning")
    ap("*Archive document for Chat R essay redaction*")
    ap(f"*Last updated: {datetime.date.today().isoformat()}*")
    ap("")
    ap("## 1. Executive summary")
    ap("")
    ap("Phase 17 implemented a 5-stage adaptive blind tuning protocol for the two")
    ap("primary PCI methods (Bennett, DRKPV), using ONLY metrics validated as")
    ap("predictive in the Phase 16 metric audit (residual_norm_h, eff_rank, kappa,")
    ap("RR, SER) — and explicitly EXCLUDING ESS_min_ratio (counter-intuitive).")
    ap("")
    ap("The protocol uses disjoint seeds across tuning waves and validation,")
    ap("with M=100 obligatoire for final validation, to avoid the winner-")
    ap("evaporation pattern documented 6 times in earlier phases.")
    ap("")
    ap("**Result for Bennett**: ")
    if bennett_state and bennett_state.get("winner"):
        winner = bennett_state["winner"]
        ap(f"compute={bennett_state.get('compute_used_minutes', 0):.1f} min, ")
        ap(f"envelope={bennett_state.get('envelope', 'unknown')}.")
    else:
        ap("(tuning incomplete or pending)")
    ap("")
    ap("**Result for DRKPV**: ")
    if drkpv_state and drkpv_state.get("winner"):
        winner = drkpv_state["winner"]
        ap(f"compute={drkpv_state.get('compute_used_minutes', 0):.1f} min, ")
        ap(f"envelope={drkpv_state.get('envelope', 'unknown')}.")
    else:
        ap("(tuning incomplete or pending)")
    ap("")

    ap("## 2. Methodology")
    ap("")
    ap("### Composite blind scores (validated Phase 16 audit)")
    ap("")
    ap("Bennett:")
    ap("```")
    ap("score = 0.30·residual_norm_h/0.005 + 0.20·max(0, 50-eff_rank_r)/50")
    ap("      + 0.15·max(0, 50-eff_rank_h)/50 + 0.15·RR_ref/0.005")
    ap("      + 0.10·log10(kappa_h)·... + 0.10·log10(kappa_r)·...")
    ap("Hard fails: kappa>1e10, RR>1e-1")
    ap("```")
    ap("")
    ap("DRKPV (with adapted thresholds for DGP2 catastrophic kappa_r):")
    ap("```")
    ap("score = 0.30·RR/0.005 + 0.20·residual_norm_r/0.05")
    ap("      + 0.20·log10(kappa_r) penalty + 0.30·|SER-1|/0.5")
    ap("Hard fails: RR>1e-1, residual_norm_r>0.5")
    ap("```")
    ap("")

    ap("### 5-stage protocol")
    ap("- **Stage 1**: 1D coarse scans on each axis (M=30)")
    ap("- **Stage 2**: 2D refinement around top axes (M=50)")
    ap("- **Stage 3**: cross-cell consistency (3 cells, M=50)")
    ap("- **Stage 4**: disjoint validation (3 cells, M=100, seed_base + 500K)")
    ap("- **Stage 5**: regime robustness (4 cells: low SNR + W/Z asymmetric)")
    ap("")

    ap("### Discipline")
    ap("- Seed_base: 60M Bennett (60.5M val, 61M robust); 70M DRKPV (70.5M val, 71M robust)")
    ap("- M=30 explore → M=50 refine → M=100 validate")
    ap("- ESS_min_ratio EXCLUDED (counter-intuitive correlation in Phase 16)")
    ap("- BIN_mh2000 (Phase 14B) reste baseline officielle (Phase 17 exploratoire)")
    ap("")

    ap("## 3. Bennett results")
    ap("")
    if bennett_stages.get(5):
        ap("### Stage 5 final summary (excerpt)")
        ap("")
        # Extract just the summary section
        s5 = bennett_stages[5]
        lines.extend(s5.split("\n")[:30])  # first 30 lines
    else:
        ap("(Stage reports not yet available)")
    ap("")

    ap("## 4. DRKPV results")
    ap("")
    if drkpv_stages.get(5):
        ap("### Stage 5 final summary (excerpt)")
        ap("")
        s5 = drkpv_stages[5]
        lines.extend(s5.split("\n")[:30])
    else:
        ap("(Stage reports not yet available)")
    ap("")

    ap("## 5. Position in essay")
    ap("")
    ap("This material is suitable for **S6.7 — Stability, tuning, and")
    ap("practitioner diagnostics**. Key narrative elements:")
    ap("")
    ap("> *We propose an adaptive 5-stage blind tuning protocol that uses only")
    ap("> diagnostics computable without access to ground truth. Using composite")
    ap("> scores validated empirically in our Phase 16 metric audit (residual_norm_h,")
    ap("> eff_rank, kappa, RR, SER), we tune Bennett's hyperparameters and arrive at")
    ap("> a config that [matches / differs from] the Phase 14B oracle-tuned BIN_mh2000.*")
    ap("")
    ap("> *For DRKPV, this is the first proper tuning beyond Phase 9B defaults. The")
    ap("> resulting config is reported with its envelope of validity (which SNR/n")
    ap("> regimes the SER calibration holds).*")
    ap("")

    ap("## 6. Reproducibility")
    ap("")
    ap("```bash")
    ap("# Run Bennett tuning (full mode, ~12h)")
    ap("python -u -m simulations.experiments.phase17_blind_tuning --method bennett --mode full")
    ap("")
    ap("# Run DRKPV tuning (full mode, ~6h)")
    ap("python -u -m simulations.experiments.phase17_blind_tuning --method drkpv --mode full")
    ap("")
    ap("# Generate comparison")
    ap("python -m simulations.archive.utilities._phase17_compare")
    ap("```")
    ap("")
    ap("Resume after interruption: pass `--resume` and `--start_stage <N>`.")
    ap("")
    ap("---")
    ap("*End of Phase 17 archive.*")

    out_path = _ARCHIVE_DIR / "phase17_archive.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


def main():
    print("=== Phase 17 Comparison + Archive ===")
    write_comparison_report()
    write_archive()


if __name__ == "__main__":
    main()
