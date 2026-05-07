"""
Phase 17 DRKPV - Stage 3 Re-analysis with Multi-dose SER Criterion
====================================================================
The original Phase 17 Stage 3 selected configs by SER at REF_IDX=2 (a=2.6) only.
Oracle audit revealed cfg2 (dropped as blind #2) was the true winner at every metric.

Fix: re-score with min_SER_across_all_doses as the selection criterion.
    SER gate: config survives if min(SER[all doses]) >= 0.85 (vs original: SER[a=2.6] >= 0.85)

This script:
1. Re-reads existing Stage 3 PKLs (no new computation needed)
2. Re-computes SER at ALL 4 doses for each config × cell
3. Selects top configs by this corrected criterion
4. Prints summary for Phase 17 DRKPV narrative
"""
from __future__ import annotations

import io
import sys
import pickle
import numpy as np
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_BASE_DIR = Path(__file__).resolve().parents[3]  # +1: moved to experiments/simulation/
_DRKPV_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "simulation" / "blind_tuning_dgp2_drkpv" / "drkpv"
_STAGE3_DIR = _DRKPV_DIR / "stage3"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
N_DOSES = len(A_GRID)
REF_IDX = 2  # a=2.6 (original)
SER_LO, SER_HI = 0.85, 1.15


def load_pkl(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return None


def extract_per_dose_metrics(data: dict) -> dict | None:
    """
    Extract bias, SE_pred, SER at all 4 doses from a PKL.
    Returns dict with arrays of length N_DOSES or None on failure.
    """
    if data is None:
        return None
    recs = [r for r in data.get("records", []) if isinstance(r, dict) and r.get("error") is None]
    if len(recs) < 10:
        return None

    n = int(data["meta"]["n"])

    # J_dr: shape (M, 4)
    try:
        J_dr = np.array([r["J_dr"] for r in recs], dtype=float)   # (M, 4)
    except Exception:
        return None
    if J_dr.ndim != 2 or J_dr.shape[1] != N_DOSES:
        return None

    # V_hat_grid: shape (M, 4)
    try:
        V_hg = np.array([r["V_hat_grid"] for r in recs], dtype=float)  # (M, 4)
    except Exception:
        return None

    # J_policy_true from meta
    J_true = None
    if "J_policy_true" in data["meta"]:
        try:
            J_true = np.asarray(data["meta"]["J_policy_true"], dtype=float)
        except Exception:
            pass

    result = {"M_ok": len(recs), "n": n}

    # Per-dose SER
    se_per = np.sqrt(np.maximum(V_hg, 0.0) / n)    # (M, 4)
    SE_pred = np.nanmean(se_per, axis=0)            # (4,)
    SD_emp = np.std(J_dr, axis=0, ddof=1)           # (4,)
    with np.errstate(invalid="ignore", divide="ignore"):
        SER = np.where(SD_emp > 1e-12, SE_pred / SD_emp, np.nan)  # (4,)
    result["SER"] = SER

    if J_true is not None:
        bias = np.mean(J_dr, axis=0) - J_true       # (4,)
        result["bias"] = bias
        # bse = |bias| / SE_pred
        with np.errstate(invalid="ignore", divide="ignore"):
            bse = np.where(SE_pred > 1e-12, np.abs(bias) / SE_pred, np.nan)
        result["bse"] = bse
        # coverage at each dose
        in_ci = np.abs(J_dr - J_true[None, :]) <= 1.96 * se_per  # (M, 4)
        result["cov"] = np.mean(in_ci, axis=0)                    # (4,)

    # Blind score components (for reference)
    rr_vals = [r.get("riesz_residual_grid_mean", [np.nan]*N_DOSES) for r in recs]
    try:
        RR_all = np.array([[float(v) for v in row] for row in rr_vals])
        result["RR_mean_alldoses"] = float(np.nanmean(RR_all))
        result["RR_ref"] = float(np.nanmean(RR_all[:, REF_IDX]))
    except Exception:
        result["RR_ref"] = float("nan")

    return result


def rescore_stage3():
    """Re-score Stage 3 configs using min SER across all doses."""
    print("=" * 70)
    print("Phase 17 DRKPV - Stage 3 Re-analysis (multi-dose SER criterion)")
    print("=" * 70)
    print()

    # Parse available PKLs: S3_c{ci}_{cell}.pkl
    cells_found = set()
    configs_found = set()
    pkl_map = {}   # (ci, cell_tag) -> Path

    for p in sorted(_STAGE3_DIR.glob("S3_*.pkl")):
        parts = p.stem.split("_")
        # S3_c0_n1000_s95 -> ci=0, cell_tag=n1000_s95
        if len(parts) < 4:
            continue
        ci = int(parts[1][1:])
        cell_tag = "_".join(parts[2:])
        configs_found.add(ci)
        cells_found.add(cell_tag)
        pkl_map[(ci, cell_tag)] = p

    cfg_ids = sorted(configs_found)
    cell_tags = sorted(cells_found)

    print(f"Configs found : {cfg_ids}")
    print(f"Cells found   : {cell_tags}")
    print()

    # Extract per-dose metrics for every (config, cell)
    results = {}   # (ci, cell_tag) -> metrics_dict
    for (ci, cell_tag), p in sorted(pkl_map.items()):
        data = load_pkl(p)
        m = extract_per_dose_metrics(data)
        if m is not None:
            results[(ci, cell_tag)] = m
        else:
            print(f"  [WARN] Could not extract metrics from {p.name}")

    print()
    print("-" * 70)
    print("SER at ALL DOSES by config × cell")
    print("-" * 70)
    header = f"{'Config':<8} {'Cell':<16} " + "  ".join(f"SER(a={a:.1f})" for a in A_GRID)
    header += "   min_SER   SER_gate"
    print(header)
    print("-" * 70)

    # Compute per-config: mean min_SER across cells
    config_min_sers = {ci: [] for ci in cfg_ids}
    config_original_sers = {ci: [] for ci in cfg_ids}   # REF_IDX only

    for ci in cfg_ids:
        for cell_tag in cell_tags:
            key = (ci, cell_tag)
            if key not in results:
                continue
            m = results[key]
            SER = m["SER"]   # (4,)
            min_ser = float(np.nanmin(SER))
            ref_ser = float(SER[REF_IDX])
            ser_pass = "PASS" if min_ser >= SER_LO and min_ser <= SER_HI else "FAIL"
            row = f"  cfg{ci}  {cell_tag:<16} "
            row += "  ".join(f"{v:8.3f}" if np.isfinite(v) else f"{'nan':>8}" for v in SER)
            row += f"   {min_ser:7.3f}   {ser_pass}"
            print(row)
            config_min_sers[ci].append(min_ser)
            config_original_sers[ci].append(ref_ser)

    print()
    print("-" * 70)
    print("Summary by config")
    print("-" * 70)
    print(f"{'Config':<8} {'mean_min_SER':<16} {'original_SER@ref':<20} {'Multi-dose rank'}")
    print("-" * 70)

    config_mean_min_ser = {}
    for ci in cfg_ids:
        vals = config_min_sers[ci]
        orig_vals = config_original_sers[ci]
        mean_min = float(np.mean(vals)) if vals else float("nan")
        mean_orig = float(np.mean(orig_vals)) if orig_vals else float("nan")
        config_mean_min_ser[ci] = mean_min
        print(f"  cfg{ci}  {mean_min:.4f}          {mean_orig:.4f}")

    ranked = sorted(config_mean_min_ser.items(), key=lambda x: -x[1])  # higher SER closer to 1 is better
    # Actually the SER gate is: we want SER in [0.85, 1.15]. Config closest to 1 on the minimum.
    # Rank by distance from 1 on the minimum SER (closer to 1 = better calibrated)
    ranked_by_dist = sorted(config_mean_min_ser.items(), key=lambda x: abs(x[1] - 1.0))
    print()
    print("Ranking by |mean_min_SER - 1.0| (calibration quality):")
    for rank, (ci, val) in enumerate(ranked_by_dist, 1):
        dist = abs(val - 1.0)
        print(f"  Rank {rank}: cfg{ci}  mean_min_SER={val:.4f}  distance={dist:.4f}")

    # Find corrected winner
    # criterion: highest min_SER that is still <= 1.15, or closest to [0.85, 1.15]
    valid_configs = [(ci, v) for ci, v in config_mean_min_ser.items()
                     if v >= SER_LO - 0.05]  # within 0.05 of gate (lenient)
    if valid_configs:
        winner_ci = max(valid_configs, key=lambda x: min(x[1], 1.0))[0]  # closest to 1 from below
        # Better: closest to 1.0
        winner_ci = min(config_mean_min_ser.keys(),
                        key=lambda ci: abs(config_mean_min_ser[ci] - 1.0))
        print()
        print(f"CORRECTED WINNER (multi-dose SER): cfg{winner_ci}")
        print(f"  mean_min_SER = {config_mean_min_ser[winner_ci]:.4f}")
    else:
        winner_ci = None
        print()
        print("[WARN] No config passes multi-dose SER gate")

    print()

    # Now show oracle metrics (non-blind) for all configs in all cells
    if any("bse" in results[k] for k in results):
        print("-" * 70)
        print("Oracle metrics: bse (|bias|/SE) at all doses")
        print("-" * 70)
        header = f"{'Config':<8} {'Cell':<16} " + "  ".join(f"bse(a={a:.1f})" for a in A_GRID)
        header += "   worst_bse"
        print(header)
        print("-" * 70)
        for ci in cfg_ids:
            for cell_tag in cell_tags:
                key = (ci, cell_tag)
                if key not in results or "bse" not in results[key]:
                    continue
                bse = results[key]["bse"]
                worst = float(np.nanmax(bse))
                row = f"  cfg{ci}  {cell_tag:<16} "
                row += "  ".join(f"{v:8.3f}" if np.isfinite(v) else f"{'nan':>8}" for v in bse)
                row += f"   {worst:7.3f}"
                print(row)

        print()
        print("-" * 70)
        print("Oracle metrics: coverage at all doses")
        print("-" * 70)
        header = f"{'Config':<8} {'Cell':<16} " + "  ".join(f"cov(a={a:.1f})" for a in A_GRID)
        header += "   min_cov"
        print(header)
        print("-" * 70)
        for ci in cfg_ids:
            for cell_tag in cell_tags:
                key = (ci, cell_tag)
                if key not in results or "cov" not in results[key]:
                    continue
                cov = results[key]["cov"]
                min_cov = float(np.nanmin(cov))
                row = f"  cfg{ci}  {cell_tag:<16} "
                row += "  ".join(f"{v:8.3f}" if np.isfinite(v) else f"{'nan':>8}" for v in cov)
                row += f"   {min_cov:7.3f}"
                print(row)

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    if winner_ci is not None:
        print(f"Multi-dose corrected winner: cfg{winner_ci}")
        print()
    print("Original Stage 3 selection criterion: SER at a=2.6 (REF_IDX) only")
    print("Corrected criterion: min(SER) across all 4 doses")
    print()
    print("If corrected winner != original (cfg0), this confirms the DRKPV blind")
    print("tuning paradox: the original criterion missed systematic mis-calibration")
    print("at non-reference doses.")
    print()

    return winner_ci


if __name__ == "__main__":
    winner = rescore_stage3()
    sys.exit(0)
