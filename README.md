# Proximal Causal Inference

This repository hosts the code that accompanies my Part III essay in
Mathematical Statistics at the University of Cambridge, *Proximal Causal
Inference: Theory, Methods, and Applications under Unmeasured
Confounding* (DPMMS, submitted May 2026; supervised by Dr P. Zhao and
Prof. Q. Zhao). The essay is public:

- PDF: https://bilelhatmi.vercel.app/docs/part_iii_essay_pci.pdf
- Project page: https://bilelhatmi.vercel.app/projects/part-iii-dissertation

The code stands on its own as a self-contained implementation of the
four estimators studied in the essay, of the diagnostic metrics that
flag their numerical fragility, and of the blind tuning protocol used to
calibrate them in the absence of any oracle reference.

## Scope

Three questions shape the realisation. How does one solve the Fredholm
inversions underlying the proximal bridges in finite samples? Which
feature approximations reduce the cost of the inversions to a tractable
order without distorting them? Which empirical quantities allow the
analyst to decide whether the resulting estimator is well-calibrated?

The library answers these questions through four estimators of the
generalised average causal effect $J(\pi_{a,h}) = \mathbb{E}[Y(\pi(A))]$
under a stochastic-policy formulation, each evaluated against the same
data-generating processes and judged by the same six Monte-Carlo metrics:

- A **linear two-stage with generated regressor** (closed-form OLS),
  the parametric baseline;
- The **KPV plug-in** of [Mastouri et al., 2021], a two-stage RKHS
  estimator that avoids inverting the Fredholm operator explicitly by
  carrying the bridge through a conditional mean embedding;
- The **doubly robust score with cross-fit** of [Cui et al., 2024] and
  [Kallus et al., 2021] on top of the KPV bridge, augmented with the
  IPW-style residual correction;
- The **Bennett RFF DR** estimator of [Bennett et al., 2023], whose
  moment-solve bridge is parameterised by random Fourier features and
  whose policy action admits a closed form through the spectral damping
  factor $\exp(-\omega_A^2 \rho_{\mathrm{pol}}^2 / 2)$.

The four are compared on three synthetic data-generating processes — a
Cobb-Douglas log-linear production model, a Michaelis-Menten
dose-response, a coarsened-Gaussian near-binary intervention — and on
two real-world applications: the NLSY79 cohort for the return to
schooling, and the [Connors et al., 1996] right-heart-catheterisation
trial for the effect of pulmonary artery catheters on 30-day mortality.

A five-stage blind tuning protocol selects hyperparameters using only
in-sample diagnostics, never the truth. The protocol is reported in
two variants: a Bennett-RFF version with an interpolation gate
$\lambda_h \times m_h \geq 10^{-2}$ that prevents the h-bridge from
overfitting, and a kernel DR version with a structural condition-number
penalty.

## Repository layout

The library mirrors the section structure of the essay.

```
pci/
├── bridges/        Bridge functions h(W,A) and q(Z,A,X) as first-class
│                   modular implementations: Oracle, RFF, KPV, Bennett.
├── dgps/           Data-generating processes (Cobb-Douglas,
│                   Michaelis-Menten, coarsened-Gaussian).
├── estimators/     Doubly-robust estimators built from the bridges
│                   (DR-Direct, DR-Kernel, BennettIndepFunctionalDR,
│                   BennettFunctionalDR), plus the linear two-stage and
│                   Kallus minimax baselines.
├── inference/      Variance decomposition and cross-fitting helpers
│                   shared by the DR estimators (V_total / V_reg /
│                   V_corr / V_cov).
├── tuning/         The five-stage blind tuning protocol and the
│                   composite score functions for both Bennett and
│                   DRKPV variants.
├── runner/         Monte-Carlo orchestration (parallel runner, block
│                   checkpoints, bandwidth heuristics, diagnostic
│                   aggregators).
├── metrics/        The six canonical Monte-Carlo metrics: bias,
│                   variance, RMSE, coverage, SER, cv_var.
├── notation.py     Symbol conventions used throughout the package.
└── tests/          Canonical test suite (130 tests, all green).

simulations/
├── experiments/
│   ├── s7_simulations/    Twelve simulation scripts (Section 7).
│   └── s8_applications/   Six real-data scripts (Section 8).
├── analysis/              Figure generators: s7_figures.py and
│                          s8_figures.py.
├── methods/, dgp/         Backward-compatible shims preserving Python
│                          identity for cached pickles.
├── datasets/              Real-data download and processing scripts
│                          (NLSY79, RHC, NHANES, ETH ESS).
├── results/raw/{s7,s8}/   Cached Monte-Carlo outputs in pickle format,
│                          consumed by the figure generators.
├── results/figures/       Output of s7_figures.py and s8_figures.py.
└── archive/               Closed-research artefacts (earlier phases of
                           the campaign retained for reproducibility).
```

## Quick start

```bash
pip install -e .
pytest -q                                    # canonical test suite
python -m simulations.analysis.s7_figures    # regenerate Section 7 figures
python -m simulations.analysis.s8_figures    # regenerate Section 8 figures
```

## Reproducibility

Every Monte-Carlo experiment writes a checkpointed pickle under
`simulations/results/raw/s7/` or `s8/`. The figure generators read
exclusively from those caches without recomputation, so all sixteen
essay figures regenerate in under one minute on a laptop. Re-running
the full Monte-Carlo campaigns from scratch — most notably the
five-stage Bennett blind tuning — is compute-bound, and the shipped
caches let the reader inspect every figure without paying that cost.
A regression suite under `pci/tests/` verifies bit-identity of the
estimators across reruns and the structural integrity of the
canonical layout.

## Datasets

The processed CSVs under `simulations/datasets/processed/` are derived
from publicly available sources: the Bureau of Labor Statistics for
NLSY79; the [Connors et al., 1996] release for the right-heart
catheterisation cohort; the Centers for Disease Control for NHANES; and
the European Social Survey for the ETH ESS extract. The raw downloads
themselves (large) are not bundled in the repository; the
`simulations/datasets/download_datasets.py` and
`simulations/datasets/process_datasets.py` scripts reproduce the
pre-processing chain from the original sources.

## License

MIT (see `LICENSE`).

## References

Bennett, A., Kallus, N., Mao, X., Newey, W., Syrgkanis, V., Uehara, M.
(2023). *Source Condition Double Robust Inference on Functionals of
Inverse Problems*. arXiv:2307.13793.

Connors, A. F. et al. (1996). *The Effectiveness of Right Heart
Catheterization in the Initial Care of Critically Ill Patients*.
JAMA 276(11), 889-897.

Cui, Y., Pu, H., Shi, X., Miao, W., Tchetgen Tchetgen, E. J. (2024).
*Semiparametric Proximal Causal Inference*. JASA 119(546), 1348-1359.

Kallus, N., Mao, X., Uehara, M. (2021). *Causal Inference Under
Unmeasured Confounding With Negative Controls: A Minimax Learning
Approach*. arXiv:2103.14029.

Mastouri, A., Zhu, Y., Gultchin, L., Korba, A., Silva, R., Kusner, M.,
Gretton, A., Muandet, K. (2021). *Proximal Causal Learning with Kernels:
Two-Stage Estimation and Moment Restriction*. ICML.

Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., Miao, W. (2024).
*An Introduction to Proximal Causal Inference*. Statistical Science 39(3).
