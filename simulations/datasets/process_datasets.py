"""
process_datasets.py -- Standardise raw downloads to A/Y/W/Z columns.

Each processed CSV has at minimum: A, Y, W, Z (plus any X covariates).
Missing datasets are skipped with a warning.

Run: python -m simulations.datasets.process_datasets
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
PROC_DIR = Path(__file__).parent / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def save(df, name, notes):
    dest = PROC_DIR / f"{name}.csv"
    df.to_csv(dest, index=False)
    print(f"  Saved {name}.csv ({len(df):,} rows x {len(df.columns)} cols)")
    print(f"  Notes: {notes}")
    return dest


# ---------------------------------------------------------------------------
# 1. RHC / SUPPORT (Connors 1996)
# ---------------------------------------------------------------------------

def process_rhc():
    src = RAW_DIR / "rhc.csv"
    if not src.exists():
        print("[RHC] raw file missing -- skip")
        return None
    print("[RHC] Processing...")
    df = pd.read_csv(src)

    # Binary treatment
    df["A"] = (df["swang1"] == "RHC").astype(float)
    # Y: 30-day mortality (1 = died within 30 days)
    df["Y"] = df["death"].map({"Yes": 1, "No": 0}).astype(float)
    # W (NCO): serum pH -- measures disease severity U, not affected by RHC procedure
    # Rationale: pH measured at admission BEFORE procedure; reflects U_severity
    df["W"] = pd.to_numeric(df["ph1"], errors="coerce")
    # Z (NCE): PaO2/FiO2 ratio -- respiratory state correlated with disease severity U
    # No direct effect on Y beyond through U and medical treatment
    df["Z"] = pd.to_numeric(df["pafi1"], errors="coerce")
    # Additional proxy: hematocrit (W2) and PaCO2 (Z2) as used in Cui 2023
    df["W2"] = pd.to_numeric(df["hema1"], errors="coerce")
    df["Z2"] = pd.to_numeric(df["paco21"], errors="coerce")
    # Covariates
    df["X_age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    df["X_meanbp"] = pd.to_numeric(df["meanbp1"], errors="coerce")

    out = df[["A", "Y", "W", "Z", "W2", "Z2", "X_age", "X_meanbp"]].dropna()
    return save(
        out, "rhc",
        "A=RHC(binary), Y=30day-death, W=pH(NCO:severity proxy), "
        "Z=PaO2/FiO2(NCE:resp.state proxy). Used in Cui2023, Tchetgen2020."
    )


# ---------------------------------------------------------------------------
# 2. 401k Pension / SIPP
# ---------------------------------------------------------------------------

def process_pension():
    src = RAW_DIR / "pension_401k.csv"
    if not src.exists():
        print("[Pension] raw file missing -- skip")
        return None
    print("[Pension] Processing...")
    df = pd.read_csv(src)
    # Columns: nifa, net_tfa, tw, age, inc, fsize, educ, db, marr, twoearn,
    #          e401, p401, pira, hown

    # Design 1: Continuous A = income (inc), Y = net_tfa (net total financial assets)
    # U = savings propensity (unobserved)
    # W = pira (IRA participation binary, proxy for savings propensity U)
    #     W is caused by U (savers open IRAs) but not by 401k eligibility/income per se
    # Z = e401 (employer 401k eligibility) -- CAVEAT: this is more IV than NCE
    #     z correlates with income via employer type (U_employer = firm quality)
    #     However: e401 has some direct effect on 401k participation A (IV logic)
    #     We use it as Z with explicit caveat in diagnostic report

    df["A"] = pd.to_numeric(df["inc"], errors="coerce")          # income ($)
    df["Y"] = pd.to_numeric(df["net_tfa"], errors="coerce")      # net financial assets
    df["W"] = pd.to_numeric(df["pira"], errors="coerce")         # IRA participation (0/1)
    df["Z"] = pd.to_numeric(df["e401"], errors="coerce")         # 401k eligibility (0/1)
    # Design 2 sub-columns for alternative interpretation
    df["A2"] = pd.to_numeric(df["p401"], errors="coerce")        # 401k participation (binary)
    df["W2"] = pd.to_numeric(df["hown"], errors="coerce")        # homeownership (0/1)
    df["X_age"] = pd.to_numeric(df["age"], errors="coerce")
    df["X_educ"] = pd.to_numeric(df["educ"], errors="coerce")
    df["X_fsize"] = pd.to_numeric(df["fsize"], errors="coerce")
    df["X_marr"] = pd.to_numeric(df["marr"], errors="coerce")

    out = df[["A", "Y", "W", "Z", "A2", "W2",
              "X_age", "X_educ", "X_fsize", "X_marr"]].dropna()
    return save(
        out, "pension_401k",
        "A=income(continuous), Y=net_tfa, W=IRA(NCO:savings proxy), "
        "Z=e401(CAVEAT:closer to IV than NCE; employer correlation with U). "
        "Benchmark: ATE ~$8k-9k (DoubleML/Chernozhukov2018)."
    )


# ---------------------------------------------------------------------------
# 3. EPA AQS Daily PM2.5 -- Philadelphia County
# ---------------------------------------------------------------------------

def process_epa_pm25():
    src = RAW_DIR / "epa_pm25_pa_2002.csv"
    if not src.exists():
        print("[EPA PM2.5] raw file missing -- skip")
        return None
    print("[EPA PM2.5] Processing...")
    df = pd.read_csv(src, dtype=str)

    # Filter Philadelphia county (FIPS 42101)
    df_philly = df[
        (df["County Code"].str.zfill(3) == "101") |
        (df["County Name"].str.lower().str.contains("philadelphia", na=False))
    ].copy()

    # Convert
    df_philly["Date"] = pd.to_datetime(df_philly["Date Local"], errors="coerce")
    df_philly["A_raw"] = pd.to_numeric(df_philly["Arithmetic Mean"], errors="coerce")

    # Aggregate to daily mean per date (multiple monitors)
    daily = (
        df_philly.groupby("Date")["A_raw"]
        .mean()
        .reset_index()
        .rename(columns={"A_raw": "A"})
        .sort_values("Date")
    )
    daily = daily.dropna(subset=["A"])

    # Temporal NCE (Z) and NCO (W): lag/lead structure per Miao & Tchetgen 2024
    # Z = PM2.5 tomorrow (lead +1 day): correlates with today's A through shared
    #     atmospheric/seasonal conditions U; does NOT cause today's Y (temporal direction)
    # W = PM2.5 yesterday (lag -1 day): caused by same U (regional weather/industry),
    #     NOT caused by today's treatment A (temporally prior)
    # Note: Without death data (CDC WONDER), Y = AQI as proxy for health risk

    daily = daily.set_index("Date")
    daily["Z"] = daily["A"].shift(-1)    # PM2.5 next day (NCE)
    daily["W"] = daily["A"].shift(1)     # PM2.5 prev day (NCO)
    daily["Y"] = daily["A"].rolling(7, center=True).mean()  # 7-day moving avg as
    # surrogate Y (in absence of death data); real Y needs CDC WONDER linkage
    daily = daily.reset_index()
    daily["day_of_week"] = daily["Date"].dt.dayofweek
    daily["month"] = daily["Date"].dt.month

    out = daily[["A", "Y", "W", "Z", "day_of_week", "month"]].dropna()
    return save(
        out, "epa_pm25_philly",
        "A=PM2.5_today(ug/m3), Y=7day_rolling_avg(PROXY, no death data available), "
        "Z=PM2.5_tomorrow(NCE:temporal lead), W=PM2.5_yesterday(NCO:temporal lag). "
        "Real Y=daily_deaths needs CDC WONDER linkage (free, manual form). "
        "Structure validated in Miao & Tchetgen (Statistics & Its Interface, 2024)."
    )


# ---------------------------------------------------------------------------
# 4. AER Guns panel (Donohue-Levitt proxy -- lacks abortion_rate)
# ---------------------------------------------------------------------------

def process_guns():
    src = RAW_DIR / "abortion_crime_panel.csv"
    if not src.exists():
        print("[Guns] raw file missing -- skip")
        return None
    print("[Guns] Processing (AER Guns proxy for Donohue-Levitt)...")
    df = pd.read_csv(src)
    # Columns: rownames, year, violent, murder, robbery, prisoners, afam, cauc,
    #          male, population, income, density, state, law

    # NOTE: This is the Guns (concealed carry law) panel, NOT the abortion-crime panel.
    # The abortion_rate variable is absent. The full Donohue-Levitt dataset needs
    # manual download from QJE replication files or Zhang et al. ICLR 2024 GitHub.

    # Closest PCI mapping for the Guns panel:
    # A = imprisonment rate per 100k (continuous, across states/years)
    # Y = murder rate (log, per 100k)
    # U = social conditions (poverty, inequality, racial demographics)
    # W = robbery rate (NCO: caused by same social conditions U as murder, but robbery
    #     is not directly caused by imprisonment rate A in the same mechanism)
    # Z = income per capita (NCE: correlated with imprisonment via socioeconomic U;
    #     income doesn't directly cause murder rate beyond through social conditions)

    df["A"] = pd.to_numeric(df["prisoners"], errors="coerce")
    df["Y"] = np.log1p(pd.to_numeric(df["murder"], errors="coerce"))
    df["W"] = np.log1p(pd.to_numeric(df["robbery"], errors="coerce"))
    df["Z"] = np.log1p(pd.to_numeric(df["income"], errors="coerce"))
    df["X_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["X_afam"] = pd.to_numeric(df["afam"], errors="coerce")
    df["X_density"] = pd.to_numeric(df["density"], errors="coerce")
    df["law_binary"] = (df["law"] == "yes").astype(float)

    out = df[["A", "Y", "W", "Z", "X_year", "X_afam", "X_density", "law_binary"]].dropna()
    return save(
        out, "guns_crime_panel",
        "PROXY for Donohue-Levitt (no abortion_rate). "
        "A=prisoners_per_100k, Y=log(murder), "
        "W=log(robbery)(NCO:same social-conditions U), "
        "Z=log(income)(NCE:corr with A via SES U). "
        "True Donohue-Levitt: fetch from Zhang et al. arXiv:2309.12819 GitHub."
    )


# ---------------------------------------------------------------------------
# 5. Ethiopia ESS Wave 3 2015 -- Fertilizer & Yield
# ---------------------------------------------------------------------------

def process_eth_ess():
    """
    Merge sect4 (fertilizer) + sect9 (yield) + geovariables (soil proxy).
    A = fertilizer kg, Y = log(1+harvest_qty), W = plot_twi (NCO), Z = ea_id mean fert share (NCE).
    """
    eth_dir = RAW_DIR / "eth_2015_ess"
    if not eth_dir.exists():
        print("[ETH ESS] raw directory missing -- skip (expected: raw/eth_2015_ess/)")
        return None
    print("[ETH ESS] Processing Wave 3 fertilizer + yield...")

    PP = eth_dir / "Post-Planting"
    PH = eth_dir / "Post-Harvest"
    GEO = eth_dir / "Geovariables"

    for p in [PP, PH, GEO]:
        if not p.exists():
            print(f"  Subdir missing: {p} -- skip")
            return None

    # Fertilizer
    df_s4 = pd.read_csv(PP / "sect4_pp_w3.csv", low_memory=False)
    df_s4["fert_used"] = (pd.to_numeric(df_s4["pp_s4q08"], errors="coerce") == 1).astype(float)
    df_s4["A"] = pd.to_numeric(df_s4["pp_s4q10"], errors="coerce").fillna(0.0)
    df_s4.loc[df_s4["fert_used"] == 0, "A"] = 0.0
    fert = df_s4[["household_id2", "parcel_id", "field_id", "ea_id", "fert_used", "A"]].copy()

    # Village-level NCE: mean fertilizer use share per enumeration area
    ea_fert = fert.groupby("ea_id")["fert_used"].mean().reset_index().rename(columns={"fert_used": "Z"})

    # Yield (cereal crops only)
    df_s9 = pd.read_csv(PH / "sect9_ph_w3.csv", low_memory=False)
    cereals = ["TEFF", "MAIZE", "WHEAT", "BARLEY", "SORGHUM"]
    df_s9_c = df_s9[df_s9["crop_name"].isin(cereals)].copy()
    df_s9_c["yield_qty"] = pd.to_numeric(df_s9_c["ph_s9q04_a"], errors="coerce")
    yield_data = df_s9_c[["household_id2", "parcel_id", "field_id", "ea_id", "crop_name", "yield_qty"]].dropna(subset=["yield_qty"])

    # Geovariables (NCO = plot TWI = soil quality proxy)
    geo = pd.read_csv(GEO / "ETH_PlotGeovariables_Y3.csv", low_memory=False)
    geo_sel = geo[["household_id2", "parcel_id", "field_id", "plot_twi", "plot_srtmslp"]].copy()

    # Merge
    merged = yield_data.merge(
        fert[["household_id2", "parcel_id", "field_id", "A", "fert_used"]],
        on=["household_id2", "parcel_id", "field_id"], how="inner"
    )
    merged = merged.merge(geo_sel, on=["household_id2", "parcel_id", "field_id"], how="left")
    merged = merged.merge(ea_fert, on="ea_id", how="left")

    merged["Y"] = np.log1p(merged["yield_qty"])
    merged["W"] = merged["plot_twi"]   # NCO: topographic wetness index
    merged["W2"] = merged["plot_srtmslp"]
    merged["X_crop"] = merged["crop_name"].astype("category").cat.codes
    merged["X_ea"] = pd.Categorical(merged["ea_id"]).codes

    out = merged[["A", "Y", "W", "W2", "Z", "X_crop", "X_ea"]].dropna()
    return save(
        out, "eth_ess_fertilizer",
        "A=fertilizer_kg(0 if no fertilizer), Y=log(1+harvest_qty)(cereal only), "
        "W=plot_twi(NCO:soil topography; not caused by A), "
        "Z=ea_mean_fert_share(NCE:village fertilizer adoption via extension access U). "
        "306 enumeration areas, 15k+ obs. Ref: Duflo-Kremer-Robinson 2008 AEJ:Applied."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_nhanes_cadmium():
    src = RAW_DIR / "nhanes_metals" / "nhanes_metals_pooled.csv"
    cot_src = list((RAW_DIR / "nhanes_metals").glob("COT_D_*.parquet"))
    if not src.exists():
        print("[NHANES] pooled CSV missing -- run download_nhanes.py first")
        return None
    print("[NHANES Cadmium] Processing Design B (A=log-cadmium, Z=log-cotinine)...")
    df = pd.read_csv(src)
    if not cot_src:
        print("  Cotinine parquet missing -- skipping")
        return None
    import pyarrow.parquet as pq
    cot = pd.read_parquet(cot_src[0])
    cot.columns = [c.upper() for c in cot.columns]
    cot_col = next((c for c in cot.columns if "COT" in c and c != "SEQN"), None)
    if cot_col is None:
        print("  Cotinine column not found -- skipping")
        return None
    cot = cot[["SEQN", cot_col]].rename(columns={cot_col: "Z_raw"})
    df = df.merge(cot, on="SEQN", how="inner") if "SEQN" in df.columns else df
    if "SEQN" not in df.columns:
        print("  SEQN not in base metals -- cannot merge cotinine")
        return None
    out = pd.DataFrame()
    out["A"]   = np.log(pd.to_numeric(df["A2"], errors="coerce").clip(0.001))   # log(blood cadmium)
    out["Y"]   = pd.to_numeric(df["Y"], errors="coerce")                         # SBP mmHg
    out["W"]   = np.log(pd.to_numeric(df["A"], errors="coerce").clip(0.001))    # log(blood lead) NCO
    out["Z"]   = np.log(pd.to_numeric(df["Z_raw"], errors="coerce").clip(0.001)) # log(cotinine) NCE
    out["X_age"]  = pd.to_numeric(df["X_age"], errors="coerce")
    out["X_sex"]  = pd.to_numeric(df["X_sex"], errors="coerce")
    out["X_ipr"]  = pd.to_numeric(df["X_ipr"], errors="coerce")
    out["X_race"] = pd.to_numeric(df["X_race"], errors="coerce")
    out = out.dropna(subset=["A", "Y", "W", "Z"])
    return save(out, "nhanes_cadmium",
        "Design B: A=log(blood_cadmium_ug/L), Y=SBP_mmHg, "
        "W=log(blood_lead) NCO: same U_exposure, not caused by cadmium; "
        "Z=log(cotinine) NCE: smoking->cadmium pathway, corr(Z,A)~0.61 log-scale. "
        "N=3695 (wave 2005-2006, adults 20-80). corr(W,Y)~0.24 [W->Y direct lead pathway: CAVEAT].")


def process_nlsy79_spouse():
    nlsy_dir = RAW_DIR / "nlsy79"
    if not nlsy_dir.exists():
        print("[NLSY79] raw directory missing -- skip (expected: raw/nlsy79/)")
        return None
    grade_src  = nlsy_dir / "GRADE.csv"
    cpubid_src = nlsy_dir / "CPUBID.csv"
    if not (grade_src.exists() and cpubid_src.exists()):
        print("[NLSY79] GRADE.csv or CPUBID.csv missing -- skip")
        return None
    print("[NLSY79 Spouse] Processing spouse-proxy design...")
    grade  = pd.read_csv(grade_src)
    cpubid = pd.read_csv(cpubid_src)
    MISS = [-1, -2, -3, -4, -5]
    def clean(df, col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce")
        return s.where(~s.isin(MISS), np.nan)

    out = pd.DataFrame()
    out["A"]         = clean(grade,  "R7007300")   # HGC revised 2000
    out["Y"]         = np.where(clean(cpubid, "R6364600") > 0,
                                np.log(clean(cpubid, "R6364600").clip(1)), np.nan)  # log own wages
    sp_wages         = clean(cpubid, "R6374900")
    out["W"]         = np.where(sp_wages > 0, np.log(sp_wages.clip(1)), np.nan)   # log spouse wages
    z_raw            = clean(grade,  "R6455100")
    out["Z"]         = z_raw.where(z_raw.between(0, 20), np.nan)                   # spouse HGC
    out["X_race"]    = clean(cpubid, "R0214700")
    out["X_sex"]     = clean(cpubid, "R0214800")
    out["X_faminc"]  = clean(cpubid, "R7006500")

    out = out.dropna(subset=["A", "Y", "W", "Z"])
    return save(out, "nlsy79_spouse",
        "SPOUSE design (not sibling -- CPUBID R0007900 not exported): "
        "A=HGC_2000(own years schooling), Y=log(own_wages_1997), "
        "W=log(spouse_wages_1997) NCO: assortative mating on U_family; "
        "Z=spouse_HGC_1998 NCE: corr(Z,A)=0.40 assortative mating. "
        "N=3074. corr(W,Y)~0 (good NCO exclusion). "
        "NOTE: True sibling design needs re-export with R0007900 + individual wages.")


PROCESSORS = {
    "RHC":            process_rhc,
    "Pension_401k":   process_pension,
    "EPA_PM25":       process_epa_pm25,
    "Guns":           process_guns,
    "ETH_ESS":        process_eth_ess,
    "NHANES_Cadmium": process_nhanes_cadmium,
    "NLSY79_Spouse":  process_nlsy79_spouse,
}

if __name__ == "__main__":
    print("=" * 60)
    print("PCI Dataset Processing -- standardise to A/Y/W/Z")
    print("=" * 60)
    results = {}
    for name, fn in PROCESSORS.items():
        print(f"\n[{name}]")
        try:
            dest = fn()
            results[name] = dest is not None
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK  " if ok else "SKIP"
        print(f"  {status}  {name}")
    print(f"\nProcessed files in: {PROC_DIR}")
