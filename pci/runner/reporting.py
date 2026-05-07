"""
pci.runner.reporting -- Markdown / stdout report helpers for stage results.

Three small utilities used by the S7 simulation scripts (S7.5, S7.6a,
S7.6b) and previously embedded in ``phase14B_bennett_indep.py``:

* :func:`_print_table` -- pretty-print a per-config metrics dict to stdout
* :func:`_build_stage_report` -- build a Markdown stage-summary string
* :func:`_save_stage_report` -- write the report to
  ``simulations/results/summaries/{prefix}_stage{label}.md``

Promoted to ``pci.runner`` in C12 (May 2026) so that the active S7
scripts no longer need to reach back into the archived
``phase14B_bennett_indep.py``.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np


_DEFAULT_COLS = [
    "bias_ref", "bse_ref", "bse_a18", "bse_a30", "cov_ref", "ser_ref",
    "ESS_min_mean", "w_p99_mean", "riesz_res_mean",
]

# Default summary directory: ``simulations/results/summaries/`` at the
# repo root. Resolved lazily so the function is portable.
_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SUMM_DIR = _PKG_ROOT / "simulations" / "results" / "summaries"


def _aggregate_metrics(
    stage_results: Dict[str, dict], ref_idx: int = 2,
) -> Dict[str, dict]:
    """For each config, extract a row of summary metrics from a `_decompose`
    output. ``ref_idx`` defaults to 2 (DGP2 reference dose)."""
    rows: Dict[str, dict] = {}
    for cid, decomp in stage_results.items():
        if decomp is None:
            continue
        bse = decomp["bse_grid"]
        bias = decomp["bias_dr"]
        rows[cid] = {
            "bias_ref": float(bias[ref_idx]),
            "bse_ref": float(bse[ref_idx]),
            "bse_a18": float(bse[0]),
            "bse_a30": float(bse[-1]),
            "cov_ref": float(decomp["coverage_ref"]),
            "ser_ref": float(decomp.get("ser_ref", float("nan"))),
            "ESS_min_mean": float(decomp.get("ess_min_mean", float("nan"))),
            "w_p99_mean": float(decomp.get("w_p99_mean", float("nan"))),
            "riesz_res_mean": float(decomp.get("riesz_res_mean", float("nan"))),
            "V_total": float(np.mean(decomp.get("V_total", float("nan"))))
                if "V_total" in decomp else float("nan"),
            "V_corr": float(np.mean(decomp.get("V_corr", float("nan"))))
                if "V_corr" in decomp else float("nan"),
        }
    return rows


def _print_table(rows: Dict[str, dict]) -> None:
    """Pretty-print a per-config metrics dict to stdout."""
    cols = _DEFAULT_COLS
    head = f"{'config':<25}" + "".join(f"{c:>11}" for c in cols)
    print(head)
    print("-" * len(head))
    for cid, m in rows.items():
        line = f"{cid:<25}"
        for c in cols:
            v = m.get(c, float("nan"))
            if not np.isfinite(v):
                line += f"{'---':>11}"
            elif abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
                line += f"{v:>11.2e}"
            else:
                line += f"{v:>11.4f}"
        print(line)


def _build_stage_report(
    stage_label: str,
    rows: Dict[str, dict],
    notes: str = "",
    title_prefix: str = "Phase",
) -> str:
    """Build a Markdown stage-summary string from per-config metrics."""
    cols = _DEFAULT_COLS
    lines = []
    lines.append(f"# {title_prefix} Stage {stage_label}")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append("")
    if notes:
        lines.append(notes)
        lines.append("")
    lines.append("## Per-config metrics")
    lines.append("")
    lines.append("| config | " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for cid, m in rows.items():
        cells = [cid]
        for c in cols:
            v = m.get(c, float("nan"))
            if not np.isfinite(v):
                cells.append("---")
            elif abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
                cells.append(f"{v:.2e}")
            else:
                cells.append(f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _save_stage_report(
    stage_label: str,
    content: str,
    summ_dir: Optional[Path] = None,
    file_prefix: str = "phase14B",
) -> Path:
    """Write the report to ``{summ_dir}/{file_prefix}_stage{label}.md``.

    Returns the resolved output Path.
    """
    if summ_dir is None:
        summ_dir = _DEFAULT_SUMM_DIR
    summ_dir = Path(summ_dir)
    summ_dir.mkdir(parents=True, exist_ok=True)
    path = summ_dir / f"{file_prefix}_stage{stage_label}.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  Stage report: {path}")
    return path
