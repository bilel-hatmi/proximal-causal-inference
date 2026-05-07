"""
Cleanup smoke test PKLs for Phase 19 real-data tuning.

The smoke test (M=3/5) creates PKLs with the same labels as the full run
(M=30/100). Without cleanup, the full run would `[SKIP]` smoke PKLs and use
M=3 results — wrong.

Usage:
    python -m simulations.experiments.real_data_cleanup_smoke --dataset rhc
    python -m simulations.experiments.real_data_cleanup_smoke --dataset rhc --baseline
    python -m simulations.experiments.real_data_cleanup_smoke --dataset rhc --diagnose
    python -m simulations.experiments.real_data_cleanup_smoke --all  # ALL datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[3]
_RAW_BASE = _BASE_DIR / "simulations" / "results" / "raw" / "real_data"
_SUMM_DIR = _BASE_DIR / "simulations" / "results" / "summaries"


def _cleanup_dataset(name: str, *, baseline: bool = False, diagnose: bool = False,
                     stages: bool = True) -> None:
    base = _RAW_BASE / name / "bennett"
    if not base.exists():
        print(f"  [{name}] no PKLs to clean (dir does not exist)")
    else:
        deleted = 0
        if stages:
            for stg in ("stage1", "stage2", "stage3", "stage4", "stage5"):
                d = base / stg
                if d.exists():
                    for p in list(d.glob("*.pkl")):
                        p.unlink(); deleted += 1
                    for p in list(d.glob("*.partial.pkl")):
                        p.unlink(); deleted += 1
            state = base / "state.json"
            if state.exists():
                state.unlink(); deleted += 1
                print(f"  [{name}] removed state.json")
            for f in _SUMM_DIR.glob(f"realdata_{name}_bennett_stage*.md"):
                f.unlink(); deleted += 1
            print(f"  [{name}] tuning artefacts removed: {deleted}")

    if baseline:
        bd = _RAW_BASE / name / "baseline"
        if bd.exists():
            for p in list(bd.glob("*.pkl")):
                p.unlink()
                print(f"  [{name}] removed baseline pkl: {p.name}")
        bm = _SUMM_DIR / f"realdata_{name}_baseline.md"
        if bm.exists():
            bm.unlink(); print(f"  [{name}] removed baseline.md")

    if diagnose:
        dd = _RAW_BASE / name / "diagnose"
        if dd.exists():
            for p in list(dd.glob("*.pkl")):
                p.unlink()
                print(f"  [{name}] removed diagnose pkl: {p.name}")
        dm = _SUMM_DIR / f"realdata_{name}_checkpoint0.md"
        if dm.exists():
            dm.unlink(); print(f"  [{name}] removed checkpoint0.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   choices=["rhc", "nhanes_cadmium", "nlsy79_father", "eth_ess"])
    p.add_argument("--all", action="store_true",
                   help="Apply to all 4 datasets")
    p.add_argument("--baseline", action="store_true",
                   help="Also remove baseline pkl/md")
    p.add_argument("--diagnose", action="store_true",
                   help="Also remove diagnose pkl/md")
    p.add_argument("--no_stages", action="store_true",
                   help="Skip stage cleanup (only baseline/diagnose)")
    args = p.parse_args()

    targets = (["rhc", "nhanes_cadmium", "nlsy79_father", "eth_ess"]
               if args.all else ([args.dataset] if args.dataset else []))
    if not targets:
        p.error("Pass --dataset NAME or --all")

    for name in targets:
        _cleanup_dataset(name, baseline=args.baseline, diagnose=args.diagnose,
                         stages=(not args.no_stages))
    print("\nDone.")


if __name__ == "__main__":
    main()
