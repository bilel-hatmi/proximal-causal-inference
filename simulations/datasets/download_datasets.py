"""
download_datasets.py -- Tier A automated dataset downloads for PCI structure evaluation.
No user registration required for any of these sources.

Run: python -m simulations.datasets.download_datasets
"""

import io
import os
import sys
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _get(url, timeout=120, stream=False):
    headers = {"User-Agent": "Mozilla/5.0 (academic research)"}
    r = requests.get(url, timeout=timeout, stream=stream, headers=headers)
    r.raise_for_status()
    return r


def download_url(url, dest, desc, timeout=120):
    """Download a single URL to dest. Return True on success."""
    print(f"  GET {url}")
    try:
        r = _get(url, timeout=timeout, stream=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
        size = os.path.getsize(dest)
        print(f"  Saved {desc} -> {dest.name} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Dataset 1: RHC / SUPPORT (Connors 1996)
# ---------------------------------------------------------------------------

def download_rhc():
    dest = RAW_DIR / "rhc.csv"
    if dest.exists():
        print(f"[RHC] already present ({dest.stat().st_size:,} bytes)")
        return True
    print("[RHC] Downloading from Vanderbilt Biostatistics...")
    return download_url(
        "https://hbiostat.org/data/repo/rhc.csv",
        dest,
        "RHC/SUPPORT Connors 1996",
    )


# ---------------------------------------------------------------------------
# Dataset 2: 401k Pension / SIPP
# ---------------------------------------------------------------------------

def download_pension():
    dest = RAW_DIR / "pension_401k.csv"
    if dest.exists():
        print(f"[Pension] already present ({dest.stat().st_size:,} bytes)")
        return True
    print("[Pension] Downloading 401k pension dataset...")

    candidate_urls = [
        # DoubleML package internal data (raw GitHub)
        "https://raw.githubusercontent.com/DoubleML/doubleml-for-py/main/doubleml/datasets/data/pension.csv",
        # Backup mirror via docs repo
        "https://raw.githubusercontent.com/DoubleML/doubleml-docs/main/doc/examples/data/pension.csv",
    ]
    for url in candidate_urls:
        if download_url(url, dest, "401k pension SIPP"):
            return True

    # Fallback: install and use doubleml
    print("  Trying pip install doubleml as fallback...")
    try:
        os.system("pip install doubleml -q")
        from doubleml.datasets import fetch_401k

        df = fetch_401k(return_type="DataFrame")
        df.to_csv(dest, index=False)
        print(f"  Saved via doubleml.datasets ({len(df):,} rows)")
        return True
    except Exception as e:
        print(f"  doubleml fallback failed: {e}")

    print("[Pension] FAILED -- all sources unavailable")
    return False


# ---------------------------------------------------------------------------
# Dataset 3: EPA AQS Daily PM2.5 -- Pennsylvania counties 2002
# ---------------------------------------------------------------------------

def download_epa_pm25(year=2002):
    dest = RAW_DIR / f"epa_pm25_pa_{year}.csv"
    if dest.exists():
        print(f"[EPA PM2.5] already present ({dest.stat().st_size:,} bytes)")
        return True
    print(f"[EPA PM2.5] Downloading annual PM2.5 file for {year} (may be large)...")
    url = f"https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"
    try:
        r = _get(url, timeout=180, stream=True)
        raw = b""
        for chunk in r.iter_content(chunk_size=131072):
            raw += chunk
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as fz:
                import pandas as pd
                df = pd.read_csv(fz, dtype=str)
        # Filter Pennsylvania (state code "42")
        col_state = "State Code"
        df_pa = df[df[col_state].astype(str).str.zfill(2) == "42"].copy()
        df_pa.to_csv(dest, index=False)
        print(f"  Saved PA subset ({len(df_pa):,} rows) -> {dest.name}")
        return True
    except Exception as e:
        print(f"[EPA PM2.5] FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Dataset 4: Donohue-Levitt abortion-crime US state panel
# ---------------------------------------------------------------------------

def download_donohue_levitt():
    dest = RAW_DIR / "abortion_crime_panel.csv"
    if dest.exists():
        print(f"[Donohue-Levitt] already present ({dest.stat().st_size:,} bytes)")
        return True
    print("[Donohue-Levitt] Searching for abortion-crime panel...")

    # Try Zhang et al. ICLR 2024 replication repo (proximal causal for continuous A)
    candidate_urls = [
        "https://raw.githubusercontent.com/liyuan9988/IVFunctions/main/data/abortion_crime.csv",
        "https://raw.githubusercontent.com/liyuan9988/DML_PCI/main/data/abortion_crime.csv",
        # The NBER working paper replication (Donohue & Levitt 2001 QJE, WP8004)
        "https://www.nber.org/sites/default/files/working_papers/w8004/w8004.pdf",  # will fail, just test
    ]
    for url in candidate_urls:
        if "pdf" in url:
            continue
        if download_url(url, dest, "Donohue-Levitt abortion-crime", timeout=30):
            return True

    # Fallback: assemble from AER R package Guns dataset (has crime, prisoners, beer)
    # The Guns dataset has: state, year, violent, murder, robbery, prisoners,
    #   afam, cauc, male, population, income, density, law (concealed carry)
    # Missing: abortion rate (the key Donohue-Levitt variable)
    # Use Wooldridge 'crime2' or assemble from public sources
    print("  Trying Wooldridge crime datasets...")
    try:
        import wooldridge
        available = dir(wooldridge)
        for name in ["crime2", "crime1", "bwght", "kielmc"]:
            if name in available:
                df = wooldridge.data(name)
                outf = RAW_DIR / f"wooldridge_{name}.csv"
                df.to_csv(outf, index=False)
                print(f"  Saved wooldridge.{name} ({len(df)} rows) -> {outf.name}")
    except ImportError:
        print("  wooldridge not installed")

    # Try fetching the Guns dataset from the AER R package mirror
    guns_urls = [
        "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/Guns.csv",
    ]
    for url in guns_urls:
        if download_url(url, dest, "AER Guns panel (Donohue-Levitt proxy)", timeout=30):
            print("  NOTE: Guns dataset (concealed carry panel) is a PROXY for full")
            print("  Donohue-Levitt abortion-crime; lacks abortion_rate variable.")
            print("  Rename to distinguish: abortion_crime_panel.csv <- Guns.csv")
            return True

    print("[Donohue-Levitt] FAILED -- replication data not auto-accessible.")
    print("  Manual action: download from QJE replication archive or")
    print("  Zhang et al. ICLR 2024 GitHub (check arXiv:2309.12819 for repo link).")
    return False


# ---------------------------------------------------------------------------
# Dataset 5: AddHealth (peer effects, restricted -- informational only)
# ---------------------------------------------------------------------------

def note_addhealth():
    print("[AddHealth] RESTRICTED -- requires DUA from https://addhealth.cpc.unc.edu/")
    print("  Used in: Egami & Tchetgen JRSSB 2024 (peer GPA as continuous A)")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "RHC_SUPPORT":      download_rhc,
    "Pension_401k":     download_pension,
    "EPA_PM25_PA2002":  download_epa_pm25,
    "Donohue_Levitt":   download_donohue_levitt,
}

if __name__ == "__main__":
    print("=" * 60)
    print("PCI Dataset Download -- Tier A (auto, no registration)")
    print("=" * 60)
    results = {}
    for name, fn in DOWNLOADERS.items():
        print(f"\n[{name}]")
        results[name] = fn()

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK  " if ok else "FAIL"
        print(f"  {status}  {name}")

    note_addhealth()
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} datasets downloaded successfully.")
    print(f"Raw files in: {RAW_DIR}")
