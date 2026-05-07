"""
Cleanup Phase 18 smoke test PKLs before full run.

The smoke test (M=3/5) creates PKLs with the same labels as the full run (M=30/100).
If not cleaned, the full run will load M=3 PKLs as [SKIP] instead of computing M=30.

This script removes all phase18/bennett/ PKLs and state.json, forcing a clean slate.
"""
import shutil
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[2]
_PHASE18_BENNETT = _BASE_DIR / "simulations" / "results" / "raw" / "phase18" / "bennett"

stages = ["stage1", "stage2", "stage3", "stage4", "stage5"]
deleted = 0

for stage in stages:
    d = _PHASE18_BENNETT / stage
    if d.exists():
        pkls = list(d.glob("*.pkl"))
        for p in pkls:
            p.unlink()
            deleted += 1
            print(f"  Deleted: {p.name}")
        print(f"  Stage {stage[-1]}: {len(pkls)} PKLs removed")

# Remove state.json
state = _PHASE18_BENNETT / "state.json"
if state.exists():
    state.unlink()
    print(f"  Deleted: state.json")

# Remove any stage reports
for f in _PHASE18_BENNETT.glob("phase18_bennett_stage*.md"):
    f.unlink()
    print(f"  Deleted: {f.name}")

print(f"\nTotal PKLs deleted: {deleted}")
print("Ready for full run.")
