"""
Phase 12B Bennett-lite RFF -- SNR stress test (post-hoc robustness check).

Phase 12B established Case B (boundary improvement) at SNR=0.95. This script
checks whether the boundary advantage persists at SNR=0.80 (stronger
confounding -> weaker proxies).

Configs:
  REG_Hsuper, DRK_Hsuper, RFF500_e35_L1e2_Hs, RFF250_e25_L1e2_Hs

Protocol:
  DGP: MichaelisMentenDGP(SNR=0.80)
  n = 1000, M = 50 (lighter than full validation -- this is robustness)
  seed_base = 25_000_000 (disjoint from all prior phases)

Usage:
  python -m simulations.archive.utilities.bennett_rff_snr_stress

Output:
  simulations/results/raw/bennett_rff/snr_stress/<label>.pkl
  simulations/results/summaries/phase12B_bennett_rff_snr_stress.md
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

_BASE_DIR = Path(__file__).resolve().parents[2]
_RAW_DIR = _BASE_DIR / "simulations" / "results" / "raw" / "bennett_rff" / "snr_stress"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"

A_GRID = np.array([1.8, 2.2, 2.6, 3.0])
REF_IDX = 2
N_FOLDS = 5
RANDOM_STATE = 42

SUPER_HP = dict(lambda_h=3e-5, ell_scale=3.5)


from simulations.experiments.dgp2_bias_diagnostics import (
    _run_block, _decompose, _silverman_h, _median_bandwidth,
    _compute_J_policy_true,
)
from simulations.archive.utilities.bennett_rff_overnight import (
    _make_reg_hsuper, _make_drk_hsuper, _make_bennett_rff,
    _augment_decomp_with_ser, _row_for_cid,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=0.80)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--M", type=int, default=50)
    args = parser.parse_args()

    snr = args.snr
    n = args.n
    M = args.M
    seed_base = 25_000_000

    print("=" * 72)
    print(f"Phase 12B SNR stress -- SNR={snr}, n={n}, M={M}, seed_base={seed_base}")
    print("=" * 72)

    from simulations.dgp.michaelis_menten import MichaelisMentenDGP
    dgp = MichaelisMentenDGP(snr_W=snr, snr_Z=snr)

    ref_sample = dgp.generate(n=2000, seed=1)
    h_KDE = _silverman_h(ref_sample.A)
    ell_W0 = _median_bandwidth(ref_sample.W)
    ell_A0 = _median_bandwidth(ref_sample.A)
    ell_Z0 = _median_bandwidth(ref_sample.Z)
    print(f"h_KDE={h_KDE:.4f}  ell_W0={ell_W0:.4f}  ell_A0={ell_A0:.4f}  ell_Z0={ell_Z0:.4f}")

    J_pt = _compute_J_policy_true(dgp, A_GRID, h_KDE)
    print(f"J_policy_true (SNR={snr}): {[f'{v:.4f}' for v in J_pt]}")
    print()

    configs = {
        "REG_Hsuper": _make_reg_hsuper(h_KDE),
        "DRK_Hsuper": _make_drk_hsuper(h_KDE, ell_W0, ell_A0, ell_Z0),
        "RFF500_e35_L1e2_Hs": _make_bennett_rff(
            n_features=500, ell_scale_rff=3.5, lambda_r=1e-2,
            h_KDE=h_KDE, rff_seed_base=25500,
        ),
        "RFF250_e25_L1e2_Hs": _make_bennett_rff(
            n_features=250, ell_scale_rff=2.5, lambda_r=1e-2,
            h_KDE=h_KDE, rff_seed_base=25250,
        ),
    }

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()
    for cfg_idx, (cid, factory_fn) in enumerate(configs.items()):
        label = f"{cid}_snr{int(snr*100)}_n{n}_M{M}"
        seed_b = seed_base + cfg_idx * 1_000_000 + n
        print(f"  [{cfg_idx+1}/{len(configs)}] {cid}  n={n}  M={M}")
        t1 = time.time()
        data = _run_block(
            dgp, factory_fn, A_GRID, J_pt,
            n=n, M=M, seed_base=seed_b,
            label=label, raw_dir=_RAW_DIR,
        )
        elapsed = time.time() - t1
        decomp = _decompose(data)
        if decomp is not None:
            decomp = _augment_decomp_with_ser(decomp, data)
        results[cid] = decomp
        if decomp is not None:
            bse = float(decomp["bse_grid"][REF_IDX])
            cov = float(decomp["coverage_ref"])
            ser = float(decomp.get("ser_ref", float("nan")))
            print(f"    |b|/SE={bse:.4f}  cov={cov:.3f}  SER={ser:.3f}  [{elapsed:.0f}s]")
    total = time.time() - t0
    print(f"\nTotal SNR stress: {total:.0f}s ({total/60:.1f} min)")

    # Build report
    import datetime
    lines = [
        f"# Phase 12B Bennett-lite RFF -- SNR stress test (SNR={snr})",
        f"Date: {datetime.date.today().isoformat()}",
        f"DGP: MichaelisMentenDGP(SNR={snr})  |  n={n}  |  M={M}  |  seed_base={seed_base}",
        "",
        "## Robustness check at lower SNR",
        "",
        "Phase 12B (full validation, SNR=0.95): boundary a=1.8 reduced 86-96% by RFF.",
        f"This stress test verifies the pattern persists at SNR={snr}.",
        "",
        "## Results",
        "",
        ("| Config | M | bias_DR | |b|/SE | cov | SER | bse_a1.8 | bse_a2.2 | bse_a2.6 | bse_a3.0 |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cid, dec in results.items():
        if dec is None:
            lines.append(f"| {cid} | --- |")
            continue
        bse = dec["bse_grid"]
        bias_dr = float(dec["bias_dr"][REF_IDX])
        bse_ref = float(bse[REF_IDX])
        cov = float(dec["coverage_ref"])
        ser = float(dec.get("ser_ref", float("nan")))
        lines.append(
            f"| {cid} | {M} | {bias_dr:+.4f} | {bse_ref:.4f} | {cov:.3f} | {ser:.3f} | "
            f"{float(bse[0]):.3f} | {float(bse[1]):.3f} | {float(bse[2]):.3f} | "
            f"{float(bse[3]):.3f} |"
        )
    # Boundary comparison
    lines.append("")
    lines.append("## Boundary improvement vs DRK at a=1.8 and a=2.2")
    lines.append("")
    if "DRK_Hsuper" in results and results["DRK_Hsuper"] is not None:
        bench = results["DRK_Hsuper"]
        bse_b = bench["bse_grid"]
        for cid in ["RFF500_e35_L1e2_Hs", "RFF250_e25_L1e2_Hs"]:
            if cid in results and results[cid] is not None:
                bse_c = results[cid]["bse_grid"]
                if bse_b[0] > 0:
                    red18 = (1 - bse_c[0] / bse_b[0]) * 100
                    lines.append(
                        f"- {cid} at a=1.8: bse={float(bse_c[0]):.3f} vs DRK {float(bse_b[0]):.3f} "
                        f"-> {red18:+.1f}% reduction"
                    )
    lines.append("")
    out = "\n".join(lines)
    out_path = _SUMM_DIR / f"phase12B_bennett_rff_snr_stress_snr{int(snr*100)}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"\nReport: {out_path}")
    print()
    for line in out.split("\n"):
        print(line)


if __name__ == "__main__":
    main()
