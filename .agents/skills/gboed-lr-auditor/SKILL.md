---
name: gboed-lr-auditor
description: Design-time learning-rate calibration and epistemic red-team auditing skill for Generalised Bayesian Experimental Design (GBOED) using PredC IMQ-WSM and GatedC G-LHW.
---

# GBOED Learning-Rate Auditor (`gboed-lr-auditor`)

**Author:** Thijs Missiaen (ORCID: [0009-0007-3678-0810](https://orcid.org/0009-0007-3678-0810))

Use this skill when auditing, calibrating, or verifying learning-rate selection rules for Generalised Bayesian Experimental Design (GBOED).

## Design-Time Calibration Invariants

**1. Zero Test-Time Leakage:** Only design-time inputs `(\pi, p(y|\theta,\xi), \ell_B, \mathcal{X})` and within-experiment history `(x_{1:t}, y_{1:t})` may enter the rule. Never accept scenario flags, true parameters, or data-generating noise parameters at inference time.

**2. Homoscedasticity and Design-Constancy Gate:** Certify both parameter homoscedasticity ($\rho _ \theta \le 10^{-6}$) and design-wise scale constancy ($\rho _ \xi \le 10^{-6}$) across prior samples $\theta \sim \pi$ and designs $\xi \in \mathcal{X}$, and enforce $p _ {\mathrm{IMQ}} > 1/2$ for scalar operation. Under homoscedasticity, $\Delta _ \infty(\xi) = 0$ iff $p _ {\mathrm{IMQ}} > 1/2$ ($p _ {\mathrm{IMQ}} = 1/2$ plateaus at $2 c(\xi)\sigma^{-4}\sup|\mu _ {\theta _ 1}-\mu _ {\theta _ 2}|$; $p _ {\mathrm{IMQ}} < 1/2$ diverges to $+\infty$). Flag non-certified heteroscedastic models (`homoscedastic=False`, `xi_constant=False`): at $p _ {\mathrm{IMQ}} = 1$, PredC IMQ-WSM retains a finite strictly positive variance tail bias gap $\Delta _ \infty(\xi) = c(\xi)^2 (\sup _ \theta \sigma _ \theta(\xi)^{-4} - \inf _ \theta \sigma _ \theta(\xi)^{-4}) > 0$ ($\Delta _ \infty = +\infty$ when heteroscedastic $p _ {\mathrm{IMQ}} < 1$).

**3. Exact 50-Digit LHW-IMQ Constants (`p_IMQ = 1`):** Enforce certified `python-flint` `arb` quadrature calibration and unconditional rate envelope:

$$
\omega_0(2\sigma) = 0.5305122672\sigma^2, \qquad \omega_0(5\sigma) = 0.5150662973\sigma^2, \qquad \omega_t \in \bigl[0.128766574328\sigma^2,\thinspace 1.061024534381\sigma^2\bigr].
$$

**4. Pure-Bayesian Continuation and Conflict-Aware Gating:** Initialize predictive centering strictly from the Bayesian prior mean $\mu _ 0(\xi)$ along the deterministic five-step `AnnealedPredC` ladder $(1.0, 0.5, 0.25, 0.1, 0.0)$ rather than non-Bayesian robust regressors, and modulate online rates (`t >= p + 2`) using normalized Mahalanobis conflict `delta0_sq` and dispersion `s_pred` with $\chi^2 _ {t, 0.85}$ (scale threshold), $\chi^2 _ {t, 0.95}$ (tempering trigger), $\phi \in [1, 2]$, and $\tau _ {\min} \in [0.25, 1]$.

## Reference Verification Commands

- Run invariant checks (`14/14`): `pytest -v`
- Run theorem, 50-digit `arb`, and Barlas repository audit: `python scripts/verify_theorem_and_barlas_audit.py`
- Run frozen GPU benchmark suite (`9/9` scenarios): `python scripts/run_all_benchmarks.py`
