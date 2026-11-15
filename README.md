# GatedC G-LHW: Design-Time Learning-Rate Selection for Generalised Bayesian Experimental Design

**Author:** Thijs Missiaen (ORCID: [0009-0007-3678-0810](https://orcid.org/0009-0007-3678-0810))  
**Contact:** thijs.missiaen@gmail.com  
**DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24249688.svg)](https://doi.org/10.5281/zenodo.24249688)  
**Challenge:** Exactory Grand Challenge on design-time learning-rate selection for generalised Bayesian experimental design (anchored on Barlas, Sloman, and Kaski, `arXiv:2511.07671`)

## Overview

This repository provides **GatedC G-LHW**, a frozen design-time and online learning-rate selection rule (`src/gboed_lr/`) for generalised Bayesian optimal experimental design (GBOED) under predictive-centered inverse multi-quadric weighted score matching (PredC IMQ-WSM). The rule relies exclusively on design-time inputs and within-experiment history `(x_{1:t}, y_{1:t})`, with zero data-generating-process leakage or test-scenario tuning.

**1. Design-time scale certification and 50-digit LHW-IMQ calibration (`design_time.py`):** Verifies both parameter homoscedasticity ($\rho _ \theta \le 10^{-6}$) and design-wise scale constancy ($\rho _ \xi \le 10^{-6}$) from prior simulator draws while enforcing $p _ {\mathrm{IMQ}} > 1/2$ (homoscedastic models achieve $\Delta _ \infty = 0$ iff $p _ {\mathrm{IMQ}} > 1/2$, plateau at $p _ {\mathrm{IMQ}} = 1/2$, and diverge when $p _ {\mathrm{IMQ}} < 1/2$), flags variance-heteroscedastic models (on Barlas's pharmacokinetic simulator: `homoscedastic=False`, relative variation `2.50132`, prior-grid lower bound $\Delta _ \infty \ge 71.5635$ and upper bound $\le 949.3513$, rising to $1716.9020$ at $\xi=2\mathrm{h}$ with $p _ {\mathrm{IMQ}} = 1$), and computes exact multivariate Lyddon–Walker–Holmes calibration via `python-flint` `arb` 50-digit ball quadrature:

$$
\omega_0(2\sigma) = 0.5305122672\sigma^2, \qquad \omega_0(5\sigma) = 0.5150662973\sigma^2.
$$

**2. Pure-Bayesian continuation and online chi-squared dispersion gate (`rule.py`):** Eliminates unprincipled regression heuristics via deterministic five-step `AnnealedPredC` scale continuation along $\lambda _ s \in (1.0, 0.5, 0.25, 0.1, 0.0)$ from the Bayesian prior mean $\mu _ 0(\xi)$, then tracks normalized Mahalanobis conflict $\hat{\Delta} _ 0^2$ and posterior-predictive dispersion $S _ t^{\mathrm{pred}}$ (`t >= p + 2`) to set:

$$
\phi(\hat{\Delta}_0^2) = 1 + \frac{\hat{\Delta}_0^4}{\hat{\Delta}_0^4 + 16^2} \in [1, 2], \qquad \tau_{\min}(\hat{\Delta}_0^2) = 1 - \frac{0.75}{1 + (\hat{\Delta}_0^2 / 4)^2} \in [0.25, 1],
$$

keeping $\omega _ t$ unconditionally within $[0.128766574328\sigma^2, 1.061024534381\sigma^2]$.

---

## Verified Benchmark Summary (`results/benchmark_results.json`)

The identical frozen `unified_omega` rule was evaluated against standard log-loss `BOED` ($\omega = 1$) across all nine `(benchmark, scenario)` pairs ($R = 90$ or $270$) on CUDA GPU, reporting mean (SE) and unpaired standardized effect $z _ {\mathrm{uSE}}$ (negative favors `GatedC G-LHW`):

| Benchmark | Scenario | BOED RMSE (SE) | GatedC RMSE ($z$) | BOED MMD (SE) | GatedC MMD ($z$) | BOED NLL (SE) | GatedC NLL ($z$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Location Finding ($T=30, R=90$) | `none` | 0.7276 (0.0015) | 0.7291 ($+0.71$) | 0.0106 (0.0009) | 0.0116 ($+0.78$) | 0.7573 (0.0028) | 0.7608 ($+0.84$) |
| Location Finding ($T=30, R=90$) | `outlier` | 0.7897 (0.0036) | 0.7509 ($-9.51$) | 0.0498 (0.0027) | 0.0189 ($-10.75$) | 0.8837 (0.0091) | 0.7808 ($-10.71$) |
| Location Finding ($T=30, R=90$) | `error-dist` | 0.9657 (0.0013) | 0.9671 ($+0.80$) | 0.0501 (0.0005) | 0.0486 ($-2.23$) | 1.5131 (0.0031) | 1.4994 ($-3.11$) |
| Linear Regression ($T=10, R=270$) | `none` | 1.7065 (0.0257) | 1.5727 ($-4.11$) | 0.1178 (0.0057) | 0.0634 ($-7.97$) | 1.7489 (0.0254) | 1.5897 ($-5.09$) |
| Linear Regression ($T=10, R=270$) | `outlier` | 2.7929 (0.0638) | 2.4616 ($-3.43$) | 0.6134 (0.0196) | 0.4212 ($-6.47$) | 4.1459 (0.1352) | 3.5685 ($-2.65$) |
| Linear Regression ($T=10, R=270$) | `laplace` | 2.0022 (0.0286) | 1.9093 ($-2.46$) | 0.1215 (0.0066) | 0.0828 ($-4.62$) | 2.2418 (0.0304) | 2.1255 ($-2.97$) |
| Nonlinear Radial-Sinusoid ($T=20, R=90$) | `none` | 0.5921 (0.0014) | 0.5888 ($-1.98$) | 0.0177 (0.0017) | 0.0140 ($-1.80$) | 0.5527 (0.0049) | 0.5416 ($-1.91$) |
| Nonlinear Radial-Sinusoid ($T=20, R=90$) | `outlier` | 0.7851 (0.0124) | 0.7077 ($-4.53$) | 0.2507 (0.0147) | 0.0988 ($-7.84$) | 1.4908 (0.0708) | 0.8276 ($-7.81$) |
| Nonlinear Radial-Sinusoid ($T=20, R=90$) | `error-dist` | 1.1405 (0.0064) | 1.1627 ($+2.63$) | 0.1784 (0.0040) | 0.1473 ($-5.83$) | 3.2916 (0.0478) | 2.7646 ($-9.20$) |

Every well-specified benchmark preserves parity under conservative unpaired comparisons ($z _ {\mathrm{uSE}} \le +0.84$), while all six misspecified scenarios significantly improve both MMD ($-2.23$ to $-10.75$ uSE) and predictive NLL ($-2.65$ to $-10.71$ uSE). Because benchmark methods share coupled RNG seeds per replicate, paired standardized differences $z _ {\mathrm{pSE}}$ give $+1.02$ MMD and $+1.08$ NLL on well-specified `Location Finding` (`none`) and $-2.68$ to $-18.60$ MMD / $-3.54$ to $-12.01$ NLL across the six misspecified scenarios. On `Nonlinear Radial-Sinusoid` (`error-dist`, $\kappa = 6.25, T = 20$), sample-pair RMSE rises slightly ($+2.63$ uSE) in exact agreement with Theorem 2.6 ($\partial _ u \mathbb{E}[\mathrm{MSE} _ {\mathrm{pred}} / \sigma^2] \lvert _ {u=0} = +0.375 > 0$).

---

## Reproducibility and Repository Structure

- `src/gboed_lr/`: Frozen design-time calibration (`design_time.py`) and pure-Bayesian continuation plus online gating (`rule.py`).
- `scripts/verify_theorem_and_barlas_audit.py`: Certifies 50-digit `python-flint` `arb` quadrature constants, Stein Fisher expectation (`<= 2.05e-16`), Theorem 2.6 conjugate derivatives (max errors `3.19e-9` NLL and `6.58e-9` MSE), conjugate-vs-BFGS exactness (`3.49e-8` L2 diff, `1.13e-11` FD gradient), Huber saturation (`y=1e6`), Barlas `dm_weighted` `w^3` triple-counting, homoscedastic $p _ {\mathrm{IMQ}}$ tail-gap checks (`p=0.25, 0.5, 1.0`), and PK heteroscedastic tail separation (`results/audit_results.json`).
- `scripts/run_all_benchmarks.py`: Vectorized CUDA GPU sequential experimental design suite and baseline ablations (`results/benchmark_results.json`).
- `scripts/generate_figures.py`: Generates the 300-DPI figures in `paper/figures/`.
- `tests/test_invariants.py`: 14 unit and GPU SMC+MH end-to-end invariant tests.
- `paper/main.tex` and `paper/main.pdf`: 18-page AMS-LaTeX manuscript and compiled PDF.

Run verification locally:

```powershell
pytest -v
python scripts/verify_theorem_and_barlas_audit.py
python scripts/run_all_benchmarks.py
python scripts/generate_figures.py
```
