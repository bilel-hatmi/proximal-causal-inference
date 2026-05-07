"""
pci_structure_diagnostics.py -- Proximal structure metrics for real datasets.

Computes quantitative diagnostics to evaluate NCO/NCE validity for each dataset.
Outputs per-dataset JSON + final markdown synthesis report.

Run: python -m simulations.archive.utilities.pci_structure_diagnostics
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

PROC_DIR = Path(__file__).parent.parent / "datasets" / "processed"
DIAG_DIR = Path(__file__).parent.parent / "datasets" / "diagnostics"
SUMM_DIR = Path(__file__).parent.parent / "results" / "summaries"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
SUMM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Diagnostic engine
# ---------------------------------------------------------------------------

def partial_corr(x, y, controls):
    """Pearson partial correlation of x~y after regressing out controls."""
    if controls.shape[1] == 0:
        return pearsonr(x, y)[0]
    reg_x = LinearRegression().fit(controls, x)
    reg_y = LinearRegression().fit(controls, y)
    res_x = x - reg_x.predict(controls)
    res_y = y - reg_y.predict(controls)
    r, _ = pearsonr(res_x, res_y)
    return r


def fstat_regression(z, a):
    """F-statistic of regressing A on Z (first-stage strength analogue)."""
    n = len(a)
    reg = LinearRegression().fit(z.reshape(-1, 1), a)
    y_pred = reg.predict(z.reshape(-1, 1))
    ss_res = np.sum((a - y_pred) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    k = 1  # one regressor
    F = (r2 / k) / ((1 - r2) / (n - k - 1))
    return float(F), float(r2)


def run_diagnostics(df, dataset_name, reference=None):
    """
    Compute proximal structure metrics for a dataset.

    Args:
        df: DataFrame with columns A, Y, W, Z (at minimum)
        dataset_name: str identifier
        reference: dict with keys 'estimate', 'naive_ols', 'source' (optional)

    Returns:
        dict of metrics
    """
    A = df["A"].values.astype(float)
    Y = df["Y"].values.astype(float)
    W = df["W"].values.astype(float)
    Z = df["Z"].values.astype(float)
    n = len(A)

    metrics = {"dataset": dataset_name, "N": int(n)}

    # --- Treatment A ---
    metrics["A_mean"]        = float(np.mean(A))
    metrics["A_std"]         = float(np.std(A))
    metrics["A_min"]         = float(np.min(A))
    metrics["A_max"]         = float(np.max(A))
    metrics["A_unique_frac"] = float(len(np.unique(A)) / n)
    metrics["A_is_continuous"] = metrics["A_unique_frac"] > 0.05

    # --- NCE (Z) diagnostics ---
    corr_ZA, pval_ZA = pearsonr(Z, A)
    corr_ZY, pval_ZY = pearsonr(Z, Y)
    F_ZA, R2_ZA      = fstat_regression(Z, A)
    # Partial corr Z~Y controlling for A (should be low if Z only affects Y through U)
    pcorr_ZY_A = partial_corr(Z, Y, A.reshape(-1, 1))
    # Partial corr Z~A controlling for Y (relevance check)
    pcorr_ZA_Y = partial_corr(Z, A, Y.reshape(-1, 1))

    metrics["corr_ZA"]          = round(float(corr_ZA), 4)
    metrics["pval_ZA"]          = round(float(pval_ZA), 6)
    metrics["corr_ZY"]          = round(float(corr_ZY), 4)
    metrics["pval_ZY"]          = round(float(pval_ZY), 6)
    metrics["F_stat_ZA"]        = round(float(F_ZA), 2)
    metrics["R2_ZA"]            = round(float(R2_ZA), 4)
    metrics["partial_corr_ZY_A"] = round(float(pcorr_ZY_A), 4)
    metrics["partial_corr_ZA_Y"] = round(float(pcorr_ZA_Y), 4)

    # NCE verdict: Z useful if |corr_ZA|>0.05 AND |partial_corr_ZY_A|<|corr_ZY|
    metrics["NCE_relevance_ok"]   = abs(corr_ZA) > 0.05
    metrics["NCE_exclusion_signal"] = abs(pcorr_ZY_A) < abs(corr_ZY)

    # --- NCO (W) diagnostics ---
    corr_WY, pval_WY = pearsonr(W, Y)
    corr_WA, pval_WA = pearsonr(W, A)
    pcorr_WA_Y = partial_corr(W, A, Y.reshape(-1, 1))
    pcorr_WY_A = partial_corr(W, Y, A.reshape(-1, 1))

    metrics["corr_WY"]           = round(float(corr_WY), 4)
    metrics["pval_WY"]           = round(float(pval_WY), 6)
    metrics["corr_WA"]           = round(float(corr_WA), 4)
    metrics["pval_WA"]           = round(float(pval_WA), 6)
    metrics["partial_corr_WA_Y"] = round(float(pcorr_WA_Y), 4)
    metrics["partial_corr_WY_A"] = round(float(pcorr_WY_A), 4)

    # NCO verdict: W useful if |corr_WY|>0.05 AND |corr_WA|<|corr_ZA| (exclusion)
    metrics["NCO_relevance_ok"]     = abs(corr_WY) > 0.05
    metrics["NCO_exclusion_signal"] = abs(corr_WA) < 0.30  # soft threshold

    # --- Confounding signal ---
    reg_naive = LinearRegression().fit(A.reshape(-1, 1), Y)
    naive_coeff = float(reg_naive.coef_[0])
    naive_r2    = float(reg_naive.score(A.reshape(-1, 1), Y))
    metrics["naive_ols_coeff"] = round(naive_coeff, 6)
    metrics["naive_ols_R2"]    = round(naive_r2, 4)

    if reference is not None:
        ref_est  = reference.get("estimate", None)
        ref_src  = reference.get("source", "unknown")
        metrics["reference_estimate"] = ref_est
        metrics["reference_source"]   = ref_src
        if ref_est is not None and ref_est != 0:
            gap = abs(naive_coeff - ref_est) / (abs(ref_est) + 1e-10)
            metrics["confounding_gap_pct"] = round(float(gap * 100), 1)
        else:
            metrics["confounding_gap_pct"] = None
    else:
        metrics["reference_estimate"] = None
        metrics["reference_source"]   = "none"
        metrics["confounding_gap_pct"] = None

    # --- Proxy completeness (rank proxy) ---
    # Gram matrix of [Z, W] correlations with Y: should have rank 2 for full PCI ID
    corr_ZW = float(pearsonr(Z, W)[0])
    gram = np.array([
        [1.0,       corr_ZW],
        [corr_ZW,   1.0    ]
    ])
    det_gram = float(np.linalg.det(gram))
    metrics["corr_ZW"]    = round(corr_ZW, 4)
    metrics["det_ZW_gram"] = round(det_gram, 4)
    # det_gram = 1 - corr_ZW^2; should be > 0 (rank 2) and ideally > 0.5
    metrics["proxy_completeness_ok"] = det_gram > 0.1

    # --- Overall PCI score (1-5) ---
    score = 0
    score += 1 if metrics["A_is_continuous"] else 0
    score += 1 if metrics["NCE_relevance_ok"] else 0
    score += 1 if metrics["NCO_relevance_ok"] else 0
    score += 1 if metrics["NCO_exclusion_signal"] else 0
    score += 1 if metrics["proxy_completeness_ok"] else 0
    metrics["pci_score"] = score

    return metrics


# ---------------------------------------------------------------------------
# Per-dataset configurations
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "rhc": {
        "file": "rhc.csv",
        "desc": "RHC/SUPPORT ICU study (Connors 1996)",
        "A_label": "RHC procedure (binary 0/1)",
        "Y_label": "30-day mortality (binary)",
        "W_label": "Serum pH at admission (NCO: severity U, not caused by RHC)",
        "Z_label": "PaO2/FiO2 ratio (NCE: respiratory state proxy for U)",
        "U_label": "Underlying disease severity (ICU patient health status)",
        "pci_papers": [
            "Tchetgen Tchetgen et al. (2020) arXiv:2009.10982",
            "Cui et al. JASA (2023) arXiv:2011.08411",
            "Bennett & Kallus (2022)",
            "Ghassami et al. AISTATS (2022)",
        ],
        "bias_strategy": "B -- IV/observational (RHC increases 30-day mortality by ~14 days; OLS shows opposite)",
        "practitioner": "Critical care: does RHC improve or harm ICU patients? Famous controversy (Connors 1996).",
        "rho_naturalness": "A is binary -- policy bandwidth ρ not applicable directly.",
        "cambridge_access": "Immediate -- free CSV from Vanderbilt Biostatistics.",
        "reference": {"estimate": -0.053, "source": "Tchetgen 2020 proximal g-comp (log-odds scale)"},
    },
    "pension_401k": {
        "file": "pension_401k.csv",
        "desc": "401(k) Pension / SIPP 1991 (Chernozhukov et al.)",
        "A_label": "Income (continuous, $)",
        "Y_label": "Net total financial assets ($)",
        "W_label": "IRA participation (binary, NCO: savings propensity U; not directly caused by income level)",
        "Z_label": "401k eligibility e401 (CAVEAT: closer to IV than NCE; employer type U_employer)",
        "U_label": "Savings propensity / financial discipline (unobserved)",
        "pci_papers": ["Used as DoubleML benchmark (not PCI specifically)"],
        "bias_strategy": "B -- quasi-experimental IV (ATE ~$8k-9k, Chernozhukov 2018 DoubleML paper)",
        "practitioner": "Retirement policy: does 401k access causally increase household wealth?",
        "rho_naturalness": "A=income is continuous; π(a,ρ)=N(income_target, ρ²) is meaningful for income-targeting policies.",
        "cambridge_access": "Immediate -- via doubleml Python package (already installed).",
        "reference": {"estimate": 8000, "source": "DoubleML ATE estimate (p401 on net_tfa)"},
        "caveat": "Z=e401 is IV-style (employer offers 401k → directly increases participation A). "
                  "For pure NCE we would need e.g. sibling employer's eligibility. "
                  "Treat as 'IV-used-as-NCE' with explicit caveat.",
    },
    "epa_pm25_philly": {
        "file": "epa_pm25_philly.csv",
        "desc": "EPA AQS Daily PM2.5 -- Philadelphia 2002 (temporal NCO/NCE design)",
        "A_label": "Daily PM2.5 concentration (μg/m³, continuous)",
        "Y_label": "7-day rolling PM2.5 avg (PROXY -- real Y=daily deaths needs CDC WONDER)",
        "W_label": "PM2.5 previous day (NCO: yesterday's air quality; caused by U=regional conditions; NOT caused by today's A)",
        "Z_label": "PM2.5 next day (NCE: tomorrow's air quality; correlated with A via shared U=weather/industry; NO causal effect on today's Y)",
        "U_label": "Regional atmospheric conditions (weather systems, industrial activity, seasonal patterns)",
        "pci_papers": [
            "Miao & Tchetgen (Statistics & Its Interface, 2024) arXiv:1808.04945 -- Philadelphia/NYC/Boston",
            "Nature Communications (2024) -- Medicare cardiovascular + PM2.5",
        ],
        "bias_strategy": "B -- Miao & Tchetgen (2024) confounding-adjusted estimate near zero (Philadelphia); OLS inflated",
        "practitioner": "Air quality policy: what is the causal effect of PM2.5 on daily mortality? EPA regulatory thresholds.",
        "rho_naturalness": "Excellent: π(a,ρ) = N(EPA_standard, ρ²) directly corresponds to 'average policy targeting 12 μg/m³ with compliance noise ρ'.",
        "cambridge_access": "EPA AQS data: immediate free download. Death data (Y): CDC WONDER free form, ~5 min.",
        "reference": {"estimate": 0.001, "source": "Miao & Tchetgen 2024 confounding-adjusted OLS (Philadelphia)"},
        "caveat": "Y is proxied by rolling PM2.5 avg (not actual mortality). Real application requires CDC WONDER daily death counts.",
    },
    "guns_crime_panel": {
        "file": "guns_crime_panel.csv",
        "desc": "AER Guns Panel (Donohue-Levitt proxy -- lacks abortion_rate variable)",
        "A_label": "Imprisonment rate per 100k (continuous, US state×year panel)",
        "Y_label": "log(murder rate per 100k)",
        "W_label": "log(robbery rate) (NCO: robbery caused by same social conditions U; not caused by imprisonment rate in same mechanism)",
        "Z_label": "log(income per capita) (NCE: income correlates with imprisonment via socioeconomic U; no direct causal path to murder beyond through U)",
        "U_label": "Socioeconomic conditions (poverty, inequality, racial demographics, urbanisation)",
        "pci_papers": [
            "Zhang et al. (ICLR 2024) arXiv:2309.12819 -- Doubly Robust PCI for Continuous Treatments",
            "  NOTE: Zhang et al. use abortion_rate as A, AFDC as Z, beer/guns/prisoners as W",
            "  This file uses Guns panel (proxy) -- abortion_rate requires manual QJE replication data",
        ],
        "bias_strategy": "B -- OLS vs fixed-effects vs IV (Lott & Mustard 1997 vs Donohue-Levitt 2001 debate)",
        "practitioner": "Criminal justice policy: causal effect of mass incarceration on crime rates? Long policy debate.",
        "rho_naturalness": "π(a,ρ) = N(imprisonment_target, ρ²) is interpretable as 'sentencing policy with jurisdiction variation ρ'.",
        "cambridge_access": "Immediate -- AER Guns from R datasets mirror (vincentarelbundock/Rdatasets).",
        "reference": None,
        "caveat": "IMPORTANT: This is the Guns (concealed carry) panel, NOT the Donohue-Levitt abortion-crime panel. "
                  "Abortion_rate variable absent. Full dataset: QJE 2001 replication files or Zhang et al. ICLR 2024 GitHub.",
    },
    "eth_ess_fertilizer": {
        "file": "eth_ess_fertilizer.csv",
        "desc": "Ethiopia ESS Wave 3 2015 -- Fertilizer & Crop Yield",
        "A_label": "Fertilizer applied (kg, continuous; 0=no fertilizer used; 48% zeros)",
        "Y_label": "log(1 + harvest quantity) -- cereal crops (maize/sorghum/teff/wheat/barley)",
        "W_label": "Plot topographic wetness index (TWI) from SRTM (NCO: soil topography caused by U_soil; not caused by fertilizer A)",
        "Z_label": "Village (ea_id) mean fertilizer use share (NCE: extension access + social norms U_access; correlated with A; no direct path to yield)",
        "U_label": "Farm-level productivity, managerial ability, wealth, and soil quality (unobserved)",
        "pci_papers": [
            "Analogous to Miao & Tchetgen (2024) spatial NCO/NCE design",
            "World Bank LSMS-ISA programme (2015 Ethiopia ESS Wave 3)",
            "Adjacent RCT benchmark: Duflo, Kremer & Robinson (2008) AEJ:Applied -- fertilizer subsidies",
        ],
        "bias_strategy": "C -- RCT benchmark from adjacent studies: fertilizer subsidy RCTs (Kenya) show "
                         "25-50% yield increase. OLS here is negative (omitted variable bias: poorer farmers on "
                         "marginal soils use fertilizer more, negative selection). Proximal design should recover "
                         "positive causal effect.",
        "practitioner": "Agricultural development: does subsidised fertilizer causally increase smallholder yields "
                        "in sub-Saharan Africa? Core question for food security (World Bank, IFAD, Gates Foundation).",
        "rho_naturalness": "Excellent: pi(a,rho) = N(recommended_dose, rho^2) directly corresponds to 'input subsidy "
                           "programme targeting 50 kg/plot with farmer adoption noise rho'. Standard recommendation "
                           "= 100 kg DAP/ha on 0.25 ha timad = 25 kg/plot.",
        "cambridge_access": "Registered World Bank Microdata Portal (user supplied ETH_2015_ESS_v03_M_CSV.zip). "
                            "Free with academic registration (~5 min).",
        "reference": {"estimate": 0.25,
                      "source": "Duflo-Kremer-Robinson 2008 AEJ:Applied -- ~25-50% yield increase (Kenya RCT)"},
        "caveat": "A = fertilizer_kg NOT normalised by plot area (GPS area only for 4181/15403 plots). "
                  "Z = village mean fertilizer share (306 EAs; 50 plots/EA avg). "
                  "Y = log(1+harvest_qty) mixes crop-specific units (partially resolved by log). "
                  "TWI is a weak NCO proxy; soil organic matter would be stronger but not collected. "
                  "48% zero-inflation in A is a feature (intent-to-treat proxy via fertilizer access).",
    },
}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

STAR_MAP = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}


def fmt(val, pct=False):
    if val is None:
        return "N/A"
    if pct:
        return f"{val:.1f}%"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def generate_report(all_metrics, all_configs):
    lines = []
    lines.append("# PCI Real-Data Dataset Evaluation")
    lines.append("# Generated by: pci_structure_diagnostics.py")
    lines.append("# Date: 2026-04-30")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Dataset | N | A continuous | NCE corr(Z,A) | NCO corr(W,A) | PCI score | Access |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, cfg in zip(all_metrics, all_configs):
        if m is None:
            continue
        lines.append(
            f"| {cfg['desc'][:35]} | {m['N']:,} | {'Yes' if m['A_is_continuous'] else 'No (binary)'} "
            f"| {m['corr_ZA']:+.3f} | {m['corr_WA']:+.3f} "
            f"| {STAR_MAP[m['pci_score']]} | {cfg['cambridge_access'][:30]} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    for m, cfg in zip(all_metrics, all_configs):
        if m is None:
            continue
        name = m["dataset"]
        lines.append(f"## Dataset: {cfg['desc']}")
        lines.append("")
        lines.append("### Variable mapping")
        lines.append("")
        lines.append("| Role | Variable | Justification |")
        lines.append("|------|----------|---------------|")
        lines.append(f"| A (treatment) | {cfg['A_label']} | Continuous treatment of interest |")
        lines.append(f"| Y (outcome)   | {cfg['Y_label']} | Outcome to model |")
        lines.append(f"| W (NCO)       | {cfg['W_label']} | NCO: caused by U, not by A |")
        lines.append(f"| Z (NCE)       | {cfg['Z_label']} | NCE: corr with A via U, no direct Y effect |")
        lines.append(f"| U (confounder)| {cfg['U_label']} | Unobserved |")
        lines.append("")

        lines.append("### Proximal structure metrics")
        lines.append("")
        lines.append(f"N = {m['N']:,} observations")
        lines.append("")
        lines.append("**Treatment A**")
        lines.append(f"- A_unique_frac = {m['A_unique_frac']:.4f} ({'continuous' if m['A_is_continuous'] else 'BINARY/DISCRETE'})")
        lines.append(f"- A mean = {m['A_mean']:.3f}, std = {m['A_std']:.3f}, range = [{m['A_min']:.3f}, {m['A_max']:.3f}]")
        lines.append("")
        lines.append("**NCE (Z) diagnostics**")
        lines.append(f"- Corr(Z, A)          = {m['corr_ZA']:+.4f}  (relevance; want |r| > 0.05) {'OK' if m['NCE_relevance_ok'] else 'WEAK'}")
        lines.append(f"- Corr(Z, Y)          = {m['corr_ZY']:+.4f}  (completeness signal)")
        lines.append(f"- Partial Corr(Z,Y|A) = {m['partial_corr_ZY_A']:+.4f}  (residual Z-Y after A; should be < |Corr(Z,Y)| if Z acts through U)")
        lines.append(f"- F-stat Z~A          = {m['F_stat_ZA']:.2f}  (first-stage strength; want > 10)")
        lines.append(f"- R2 Z~A              = {m['R2_ZA']:.4f}")
        lines.append("")
        lines.append("**NCO (W) diagnostics**")
        lines.append(f"- Corr(W, Y)          = {m['corr_WY']:+.4f}  (relevance; want |r| > 0.05) {'OK' if m['NCO_relevance_ok'] else 'WEAK'}")
        lines.append(f"- Corr(W, A)          = {m['corr_WA']:+.4f}  (exclusion check; want |r| < 0.30) {'OK' if m['NCO_exclusion_signal'] else 'HIGH -- W may be caused by A'}")
        lines.append(f"- Partial Corr(W,Y|A) = {m['partial_corr_WY_A']:+.4f}  (W captures residual U after controlling A)")
        lines.append("")
        lines.append("**Proxy pair structure**")
        lines.append(f"- Corr(Z, W)          = {m['corr_ZW']:+.4f}  (collinearity check; want < 0.9)")
        lines.append(f"- det([Z,W] gram)     = {m['det_ZW_gram']:.4f}  (proxy completeness; want > 0.1) {'OK' if m['proxy_completeness_ok'] else 'LOW -- Z and W may be collinear'}")
        lines.append("")
        lines.append("**Confounding signal**")
        lines.append(f"- Naive OLS coeff A~Y = {m['naive_ols_coeff']:+.6f}  (naive, ignores U)")
        if m['reference_estimate'] is not None:
            lines.append(f"- Reference estimate  = {m['reference_estimate']}  (source: {m['reference_source']})")
        if m['confounding_gap_pct'] is not None:
            lines.append(f"- Confounding gap     = {m['confounding_gap_pct']:.1f}%  (|OLS - reference| / |reference|)")
        lines.append("")

        lines.append(f"**PCI score: {STAR_MAP[m['pci_score']]} ({m['pci_score']}/5)**")
        lines.append("")
        criteria = [
            ("A is continuous (unique_frac > 5%)", m["A_is_continuous"]),
            ("NCE relevance OK (|corr_ZA| > 0.05)", m["NCE_relevance_ok"]),
            ("NCO relevance OK (|corr_WY| > 0.05)", m["NCO_relevance_ok"]),
            ("NCO exclusion plausible (|corr_WA| < 0.30)", m["NCO_exclusion_signal"]),
            ("Proxy completeness (det > 0.1)", m["proxy_completeness_ok"]),
        ]
        for desc_c, ok in criteria:
            lines.append(f"- {'[x]' if ok else '[ ]'} {desc_c}")
        lines.append("")

        lines.append("### PCI paper usage")
        for ref in cfg["pci_papers"]:
            lines.append(f"- {ref}")
        lines.append("")

        lines.append(f"### Bias reference")
        lines.append(f"{cfg['bias_strategy']}")
        lines.append("")

        lines.append(f"### Practitioner relevance")
        lines.append(cfg["practitioner"])
        lines.append("")

        lines.append(f"### Dataset access")
        lines.append(cfg["cambridge_access"])
        lines.append("")

        lines.append(f"### Policy bandwidth rho naturalness")
        lines.append(cfg["rho_naturalness"])
        lines.append("")

        if "caveat" in cfg:
            lines.append(f"### Caveats")
            lines.append(cfg["caveat"])
            lines.append("")

        lines.append("---")
        lines.append("")

    # Footer
    lines.append("## Datasets needing user action (Tier B)")
    lines.append("")
    lines.append("| Dataset | Status | Action required | Natural A | Natural W | Natural Z |")
    lines.append("|---------|--------|-----------------|-----------|-----------|-----------|")
    lines.append("| NLSY79 siblings | **INCOMPLETE** -- user exported only 4 demographic cols | Re-export from NLS Investigator with HGC (schooling), wages, CPUBID (sibling link) | Years schooling | Sibling's wages | Sibling's schooling |")
    lines.append("| Job Corps | Not downloaded | Register openICPSR https://www.openicpsr.org/openicpsr/project/113269 | Training hours (0-1200h) | Pre-program earnings | Local unemployment |")
    lines.append("| Ethiopia LSMS-ISA | **PROCESSED** (see above) | Already processed as eth_ess_fertilizer.csv | Fertilizer kg (raw) | Plot TWI | Village fert share |")
    lines.append("| AddHealth | Not downloaded -- DUA required | DUA from https://addhealth.cpc.unc.edu/ | Avg peer GPA | Own baseline GPA | Peers-of-peers GPA |")
    lines.append("| MIMIC-IV | Not downloaded | PhysioNet + CITI training (~2h) https://physionet.org/content/mimiciv/ | Vasopressor mg/kg/min | Admission biomarkers | Physician tendency |")
    lines.append("")
    lines.append("## Key references")
    lines.append("")
    lines.append("- arXiv:2309.12819 -- Zhang et al. (ICLR 2024) Doubly Robust Proximal Causal for Continuous Treatments")
    lines.append("  Donohue-Levitt as real-data benchmark (abortion_rate as continuous A)")
    lines.append("- arXiv:1808.04945 -- Miao & Tchetgen (2024) PM2.5/mortality with temporal NCO/NCE design")
    lines.append("- arXiv:2011.08411 -- Cui et al. JASA (2023) RHC as canonical PCI benchmark")
    lines.append("- arXiv:2109.01933 -- Egami & Tchetgen JRSSB (2024) Add Health peer GPA as continuous A")
    lines.append("- arXiv:2512.12038 -- Park et al. (2025) Antibody titer with Modified Treatment Policy")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PCI Structure Diagnostics")
    print("=" * 60)

    all_metrics = []
    all_configs = []

    for key, cfg in DATASET_CONFIGS.items():
        print(f"\n[{key}]")
        fpath = PROC_DIR / cfg["file"]
        if not fpath.exists():
            print(f"  processed file missing: {fpath} -- skip")
            continue
        df = pd.read_csv(fpath)
        required = ["A", "Y", "W", "Z"]
        if not all(c in df.columns for c in required):
            print(f"  missing columns {required} in {fpath.name} -- skip")
            continue

        df = df[required].dropna()
        if len(df) < 30:
            print(f"  too few rows ({len(df)}) -- skip")
            continue

        ref = cfg.get("reference")
        m = run_diagnostics(df, key, reference=ref)

        # Save JSON
        json_path = DIAG_DIR / f"{key}_metrics.json"
        # Convert numpy/python booleans to plain Python types for JSON serialization
        m_serial = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                    for k, v in m.items()}
        with open(json_path, "w") as fj:
            json.dump(m_serial, fj, indent=2)
        print(f"  Saved diagnostics -> {json_path.name}")

        # Quick summary
        print(f"  N={m['N']:,}  A_unique={m['A_unique_frac']:.3f}  "
              f"corr_ZA={m['corr_ZA']:+.3f}  corr_WA={m['corr_WA']:+.3f}  "
              f"score={m['pci_score']}/5")

        all_metrics.append(m)
        all_configs.append(cfg)

    # Generate markdown report
    print("\nGenerating synthesis report...")
    report = generate_report(all_metrics, all_configs)
    report_path = SUMM_DIR / "pci_dataset_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as fr:
        fr.write(report)
    print(f"Report saved -> {report_path}")

    # ASCII summary table
    print("\n" + "=" * 60)
    print("PROXIMAL STRUCTURE SUMMARY")
    print("=" * 60)
    hdr = f"{'Dataset':<25} {'N':>6}  {'A-cont':>6}  {'corr_ZA':>8}  {'corr_WA':>8}  {'score':>6}"
    print(hdr)
    print("-" * len(hdr))
    for m in all_metrics:
        print(
            f"  {m['dataset']:<23} {m['N']:>6,}  "
            f"{'Yes' if m['A_is_continuous'] else 'No ':>6}  "
            f"{m['corr_ZA']:>+8.3f}  "
            f"{m['corr_WA']:>+8.3f}  "
            f"  {m['pci_score']}/5"
        )
