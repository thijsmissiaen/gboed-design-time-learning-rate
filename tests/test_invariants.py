"""Unit and invariant tests for GatedC G-LHW design-time calibration and unified_omega."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from gboed_lr import (
    annealed_conjugate_predc_posterior,
    annealed_predc_scales,
    compute_delta0_sq,
    compute_s_pred,
    conjugate_predc_imq_wsm,
    design_time_lhw_imq,
    imq_wsm_quadrature_constants,
    phi_conflict,
    sup_psi_B_exact,
    tau_min_floor,
    unified_omega,
)


def _homo_sigma(theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    return np.full((theta.shape[0], xi.shape[0]), 0.5, dtype=np.float64)


def _hetero_pk_sigma(theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    # Simplified PK heteroscedastic scale: sigma_theta(xi) = sqrt((eps * mu_theta(xi))^2 + nu^2)
    k_a = np.exp(theta[:, 0:1])
    k_e = np.exp(theta[:, 1:2])
    vol = np.exp(theta[:, 2:3])
    t = xi[None, :, 0]
    mu = (400.0 / vol) * (k_a / np.maximum(k_a - k_e, 1e-4)) * (np.exp(-k_e * t) - np.exp(-k_a * t))
    return np.sqrt((0.1 * mu) ** 2 + 0.1)


def test_design_time_signature_isolation() -> None:
    """Verify design_time_lhw_imq and unified_omega expose zero DGP or scenario oracle inputs."""
    forbidden = {"true_theta", "dgp", "scenario", "misspecification", "oracle", "true_sigma"}
    for fn in (design_time_lhw_imq, unified_omega):
        params = set(inspect.signature(fn).parameters.keys())
        assert params.isdisjoint(forbidden), f"Forbidden oracle parameter found in {fn.__name__}: {params & forbidden}"


def test_quadrature_constants_exact() -> None:
    """Verify exact 1D Gaussian quadrature ratios H_B/J_B and exact radical sup_psi_B."""
    c2 = imq_wsm_quadrature_constants(2.0, p_imq=1.0)
    c5 = imq_wsm_quadrature_constants(5.0, p_imq=1.0)

    assert math.isclose(c2.lhw_ratio, 0.5305122671904373, rel_tol=1e-11, abs_tol=1e-12)
    assert math.isclose(c5.lhw_ratio, 0.5150662973154196, rel_tol=1e-11, abs_tol=1e-12)
    assert c2.var_inflation > 1.0
    assert c5.var_inflation > 1.0
    assert c5.var_inflation < c2.var_inflation  # c=5sigma is closer to full Fisher efficiency

    # Closed-form radical for p_IMQ = 1: w_* = sqrt(sqrt(9 r^-4 + 2 r^-2 + 1) - 3 r^-2)
    for r, qc in [(2.0, c2), (5.0, c5)]:
        w_star = math.sqrt(math.sqrt(9.0 / (r ** 4) + 2.0 / (r ** 2) + 1.0) - 3.0 / (r ** 2))
        sup_rad = r * w_star * (1.0 + w_star ** 2 + 2.0 / (r ** 2)) / ((1.0 + w_star ** 2) ** 2)
        assert math.isclose(qc.sup_psi_B, sup_rad, rel_tol=1e-15, abs_tol=1e-15)

    # Bounded influence holds iff p_IMQ >= 1/2
    assert math.isfinite(sup_psi_B_exact(2.0, p_imq=0.5))
    assert math.isinf(sup_psi_B_exact(2.0, p_imq=0.25))


def test_homoscedasticity_certification_and_pk_flag() -> None:
    """Verify homoscedastic passes, PK has finite Delta_inf > 0, and xi-varying sigma is flagged."""
    rng = np.random.default_rng(42)
    theta_homo = rng.normal(0.0, 1.0, size=(64, 4))
    designs = rng.uniform(-4.0, 4.0, size=(20, 2))
    cal_homo = design_time_lhw_imq(_homo_sigma, theta_homo, designs)
    assert cal_homo.homoscedastic is True
    assert cal_homo.xi_constant is True
    assert cal_homo.delta_inf == 0.0
    assert math.isclose(cal_homo.omega0_2, 0.5305122671904373 * 0.25, rel_tol=1e-11)

    theta_pk = np.array([[0.0, math.log(0.1), math.log(20.0)], [0.3, math.log(0.15), math.log(15.0)]])
    t_grid = np.linspace(0.5, 24.0, 12)[:, None]
    cal_pk = design_time_lhw_imq(_hetero_pk_sigma, theta_pk, t_grid, raise_on_heteroscedastic=False)
    assert cal_pk.homoscedastic is False
    # Under p_IMQ=1 and sigma_theta >= sqrt(0.1), Delta_inf is strictly positive finite (<= 100 * c_max^2)
    assert math.isfinite(cal_pk.delta_inf)
    assert cal_pk.delta_inf > 0.0
    max_pk_sig = float(np.max(_hetero_pk_sigma(theta_pk, t_grid)))
    assert cal_pk.delta_inf <= 100.0 * ((2.0 * max_pk_sig) ** 2) + 1e-6

    with pytest.raises(ValueError, match="Heteroscedastic"):
        design_time_lhw_imq(_hetero_pk_sigma, theta_pk, t_grid, raise_on_heteroscedastic=True)

    with pytest.raises(ValueError, match="non-certified heteroscedastic"):
        unified_omega(t=5, p=3, sigma=0.5, delta0_sq=1.0, s_pred=5.0, calibration=cal_pk)

    # Design-dependent homoscedastic noise sigma(xi) = 0.3 + 0.5 * |xi| is independent of theta,
    # so homoscedastic=True (Delta_inf=0), but xi_constant=False and rejected by scalar unified_omega.
    xi_1d = np.linspace(-2.0, 2.0, 15)[:, None]
    sigma_xi_only = lambda th, xi: np.broadcast_to((0.3 + 0.5 * np.abs(xi[:, 0]))[None, :], (th.shape[0], xi.shape[0]))
    cal_xi = design_time_lhw_imq(sigma_xi_only, theta_homo, xi_1d, raise_on_heteroscedastic=False)
    assert cal_xi.homoscedastic is True
    assert cal_xi.delta_inf == 0.0
    assert cal_xi.xi_constant is False
    assert cal_xi.rel_xi_variation > 1.0
    with pytest.raises(ValueError, match="Design-dependent"):
        design_time_lhw_imq(sigma_xi_only, theta_homo, xi_1d, raise_on_heteroscedastic=True)
    with pytest.raises(ValueError, match="design-varying noise scale"):
        unified_omega(t=5, p=2, sigma=0.5, delta0_sq=1.0, s_pred=3.0, calibration=cal_xi)


def test_dimensional_scaling_omega_proportional_to_sigma_sq() -> None:
    """Verify omega_0(c) and omega_t scale strictly as sigma^2."""
    rng = np.random.default_rng(0)
    theta = rng.normal(0.0, 1.0, size=(16, 2))
    designs = np.linspace(-4.0, 4.0, 10)[:, None]

    cal_1 = design_time_lhw_imq(lambda th, xi: np.full((th.shape[0], xi.shape[0]), 0.5), theta, designs)
    cal_2 = design_time_lhw_imq(lambda th, xi: np.full((th.shape[0], xi.shape[0]), 1.5), theta, designs)

    scale = (1.5 / 0.5) ** 2
    assert math.isclose(cal_2.omega0_2 / cal_1.omega0_2, scale, rel_tol=1e-12)
    assert math.isclose(cal_2.omega0_5 / cal_1.omega0_5, scale, rel_tol=1e-12)

    d1 = unified_omega(t=6, p=2, sigma=0.5, delta0_sq=2.0, s_pred=4.0, calibration=cal_1)
    d2 = unified_omega(t=6, p=2, sigma=1.5, delta0_sq=2.0, s_pred=4.0, calibration=cal_2)
    assert math.isclose(d2.omega_t / d1.omega_t, scale, rel_tol=1e-12)
    assert math.isclose(d2.c_t / d1.c_t, 3.0, rel_tol=1e-12)


def test_phi_and_tau_min_asymptotics_and_bounds() -> None:
    """Verify phi(0)=1, lim phi=2, tau_min(0)=0.25, lim tau_min=1, and monotonicity."""
    assert math.isclose(phi_conflict(0.0), 1.0, abs_tol=1e-15)
    assert math.isclose(tau_min_floor(0.0), 0.25, abs_tol=1e-15)
    assert math.isclose(phi_conflict(1e8), 2.0, abs_tol=1e-12)
    assert math.isclose(tau_min_floor(1e8), 1.0, abs_tol=1e-12)

    grid = np.array([0.0, 0.5, 2.0, 8.0, 32.0, 128.0, 1e4])
    phis = phi_conflict(grid)
    taus = tau_min_floor(grid)
    assert np.all(np.diff(phis) > 0.0)
    assert np.all(np.diff(taus) > 0.0)


def test_unified_omega_gates_and_warmup() -> None:
    """Verify warmup phase t < p+2 and chi2 thresholds at t >= p+2."""
    theta = np.zeros((8, 2))
    designs = np.zeros((5, 1))
    cal = design_time_lhw_imq(lambda th, xi: np.full((th.shape[0], xi.shape[0]), 1.0), theta, designs)

    # Warmup: t=3 < p+2=4 always selects c=2*sigma, tau=1
    d_warm = unified_omega(t=3, p=2, sigma=1.0, delta0_sq=0.0, s_pred=100.0, calibration=cal)
    assert math.isclose(d_warm.c_t, 2.0)
    assert math.isclose(d_warm.tau_t, 1.0)
    assert math.isclose(d_warm.omega_t, cal.omega0_2)

    # Gated well-specified: s_pred <= chi2_{t, 0.85} -> c=5*sigma, tau=1
    t = 10
    q85 = float(chi2.ppf(0.85, df=t))
    q95 = float(chi2.ppf(0.95, df=t))
    d_in = unified_omega(t=t, p=2, sigma=1.0, delta0_sq=1.0, s_pred=0.5 * q85, calibration=cal)
    assert math.isclose(d_in.c_t, 5.0)
    assert math.isclose(d_in.tau_t, 1.0)

    # Gated mild dispersion: chi2_{t,0.85} < s_pred <= chi2_{t,0.95} -> c=2*sigma, tau=1
    d_mid = unified_omega(t=t, p=2, sigma=1.0, delta0_sq=1.0, s_pred=0.5 * (q85 + q95), calibration=cal)
    assert math.isclose(d_mid.c_t, 2.0)
    assert math.isclose(d_mid.tau_t, 1.0)

    # Gated severe dispersion: s_pred > chi2_{t,0.95} -> c=2*sigma, tau=max(tau_min, t/s_pred)
    d_sev = unified_omega(t=t, p=2, sigma=1.0, delta0_sq=0.0, s_pred=1000.0, calibration=cal)
    assert math.isclose(d_sev.c_t, 2.0)
    assert math.isclose(d_sev.tau_t, 0.25)  # floor tau_min(0)=0.25 > 10/1000


def test_param_a_prior_trap_lower_bound() -> None:
    """Verify exact Param A prior-trap inequality: 0.5*||theta*-mu0||^2 > omega*T*Delta_lmax traps at mu0."""
    # 1D Gaussian mean under Param A IMQ-WSM: ell_A(theta; y) = m^2 s^2 + 2*(m^2 s' + (m^2)' s)
    # where s = -(y - theta)/sigma^2, s' = -1/sigma^2, m^2 = (1 + (y-theta)^2 / c^2)^(-1)
    sigma = 1.0
    c = 2.0 * sigma
    omega = 0.530512 * (sigma ** 2)
    T = 5

    def ell_param_a(theta: float, y: np.ndarray) -> np.ndarray:
        r = y - theta
        m2 = 1.0 / (1.0 + (r / c) ** 2)
        dm2_dy = -2.0 * r / (c ** 2) * (m2 ** 2)
        s = -r / (sigma ** 2)
        ds_dy = -1.0 / (sigma ** 2)
        return m2 * (s ** 2) + 2.0 * (m2 * ds_dy + dm2_dy * s)

    # Exact global oscillation bound Delta_lmax = sup_r ell_A - inf_r ell_A
    r_grid = np.linspace(-200.0, 200.0, 200001)
    vals = ell_param_a(0.0, r_grid)
    delta_lmax = float(np.max(vals) - np.min(vals))

    # Choose theta* violating the recovery condition: 0.5 * theta*^2 > omega * T * delta_lmax
    theta_star = 12.0  # 0.5 * 144 = 72 >> 0.53 * 5 * ~6.5 = ~17.2
    assert 0.5 * (theta_star ** 2) > omega * T * delta_lmax

    y_data = np.full(T, theta_star)
    log_post_at_mu0 = -0.5 * (0.0 ** 2) - omega * float(np.sum(ell_param_a(0.0, y_data)))
    log_post_at_star = -0.5 * (theta_star ** 2) - omega * float(np.sum(ell_param_a(theta_star, y_data)))

    # Posterior density at the misspecified prior mean mu0=0 strictly dominates theta*
    assert log_post_at_mu0 > log_post_at_star


def test_delta0_and_s_pred_helpers() -> None:
    """Smoke check for compute_delta0_sq and compute_s_pred."""
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 1.0, size=(5000, 3))
    d0 = compute_delta0_sq(samples, np.zeros(3), np.eye(3))
    assert 0.90 < d0 < 1.10

    s_val = compute_s_pred(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.0, 0.0]), sigma=0.5)
    assert s_val == 0.0


def test_overflow_and_nan_safety() -> None:
    """Verify overflow-safe phi_conflict/tau_min_floor and NaN rejection in unified_omega."""
    assert math.isclose(phi_conflict(1e200), 2.0, abs_tol=1e-12)
    assert math.isclose(phi_conflict(float("inf")), 2.0, abs_tol=1e-12)
    assert math.isclose(tau_min_floor(1e200), 1.0, abs_tol=1e-12)
    assert math.isclose(tau_min_floor(float("inf")), 1.0, abs_tol=1e-12)

    theta = np.zeros((4, 2))
    designs = np.zeros((3, 1))
    cal = design_time_lhw_imq(lambda th, xi: np.full((th.shape[0], xi.shape[0]), 1.0), theta, designs)

    with pytest.raises(ValueError, match="NaN"):
        unified_omega(t=5, p=2, sigma=1.0, delta0_sq=float("nan"), s_pred=2.0, calibration=cal)
    with pytest.raises(ValueError, match="NaN"):
        unified_omega(t=5, p=2, sigma=1.0, delta0_sq=1.0, s_pred=float("nan"), calibration=cal)

    d_inf_s = unified_omega(t=5, p=2, sigma=1.0, delta0_sq=0.0, s_pred=float("inf"), calibration=cal)
    assert math.isclose(d_inf_s.tau_t, 0.25, abs_tol=1e-12)
    assert math.isclose(d_inf_s.c_t, 2.0, abs_tol=1e-12)

    with pytest.raises(ValueError, match="at least 2"):
        design_time_lhw_imq(lambda th, xi: np.ones((th.shape[0], xi.shape[0])), np.zeros((1, 2)), designs)


def test_stein_fisher_consistency_and_conjugate_vs_bfgs() -> None:
    """Verify exact Stein Fisher consistency at fixed mu_hat != mu* and conjugate vs BFGS."""
    sigma = 0.8
    c = 2.0 * sigma
    mu_star = 3.2
    for mu_hat in (mu_star - 7.3 * sigma, mu_star, mu_star + 5.1 * sigma):
        def d_mu_ell(y: float) -> float:
            r = y - mu_hat
            u = 1.0 + (r / c) ** 2
            m2 = 1.0 / u
            dm2 = -2.0 * r / (c * c * u * u)
            return 2.0 * m2 * (mu_star - y) / (sigma ** 4) + 2.0 * dm2 / (sigma ** 2)

        exp_score, _ = quad(lambda y: d_mu_ell(y) * norm.pdf(y, loc=mu_star, scale=sigma), -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13)
        assert abs(exp_score) < 1e-13

    # Verify closed-form conjugate update matches numerical minimization of Gibbs free energy
    rng = np.random.default_rng(2026)
    x = np.linspace(-3.0, 3.0, 9)
    phi = np.column_stack([np.ones_like(x), x, x ** 2])
    beta_true = np.array([2.5, -1.7, 0.4])
    y = phi @ beta_true + rng.normal(0.0, sigma, size=len(x))
    mu0 = np.array([0.5, -0.3, 0.1])
    prec0 = np.diag([0.8, 1.5, 2.2])
    mu_hat = phi @ np.array([2.1, -1.5, 0.3])
    omega = 0.53 * (sigma ** 2)

    mean_conj, cov_conj = conjugate_predc_imq_wsm(phi, y, mu_hat, sigma=sigma, c=c, omega=omega, mu0=mu0, sigma0_inv=prec0)

    def objective(th: np.ndarray) -> float:
        d = th - mu0
        prior_term = 0.5 * float(d @ prec0 @ d)
        dy = y - mu_hat
        u = 1.0 + (dy / c) ** 2
        m2 = 1.0 / u
        dm2 = -2.0 * dy / (c * c * u * u)
        mu_th = phi @ th
        s = (mu_th - y) / (sigma ** 2)
        ell = m2 * (s ** 2) + 2.0 * (-m2 / (sigma ** 2) + dm2 * s)
        return prior_term + omega * float(np.sum(ell))

    opt = minimize(objective, np.zeros(3), method="BFGS", options={"gtol": 1e-7})
    assert opt.success
    assert np.linalg.norm(mean_conj - opt.x) < 1e-7
    # Also check central finite-difference gradient and Hessian directly at mean_conj
    h = 1e-4
    grad_fd = np.zeros(3)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        grad_fd[k] = (objective(mean_conj + e) - objective(mean_conj - e)) / (2.0 * h)
    assert np.linalg.norm(grad_fd) < 1e-10
    assert objective(mean_conj) <= objective(opt.x) + 1e-14
    assert np.all(np.linalg.eigvalsh(cov_conj) > 0.0)


def test_annealed_predc_unbiasedness_and_huber_bounded_shift() -> None:
    """Verify pure-Bayesian AnnealedPredC recovers signal under outliers and has bounded outlier shift."""
    rng = np.random.default_rng(123)
    x = np.linspace(-4.0, 4.0, 12)
    phi = np.column_stack([np.ones(12), x])
    beta_true = np.array([10.0, -7.0])
    sigma = 0.5
    y = beta_true[0] + beta_true[1] * x + rng.normal(0.0, sigma, size=12)
    y_corrupt = y.copy()
    y_corrupt[[1, 5, 9]] -= 40.0

    scales = annealed_predc_scales(y_corrupt, np.zeros(12), sigma=sigma)
    assert scales[0] > 2.0 * sigma
    assert math.isclose(scales[-1], 2.0 * sigma, abs_tol=1e-12)

    mean_post, cov_post, _ = annealed_conjugate_predc_posterior(
        phi,
        y_corrupt,
        sigma=sigma,
        c_target=2.0 * sigma,
        omega_target=0.530512267 * (sigma ** 2),
    )
    assert np.linalg.norm(mean_post - beta_true) < 1.0
    assert np.all(np.linalg.eigvalsh(cov_post) > 0.0)

    # Huber bounded gross-error sensitivity as |y_0| -> 1e6 vs unbounded BOED divergence
    y_base = beta_true[0] + beta_true[1] * x
    norms_gated = []
    norms_boed = []
    for y_out in (1e2, 1e4, 1e6):
        y_test = y_base.copy()
        y_test[0] = y_out
        m_g, _, _ = annealed_conjugate_predc_posterior(phi, y_test, sigma=sigma, c_target=2.0 * sigma, omega_target=0.5305 * (sigma ** 2))
        prec_b = np.eye(2) + (phi.T @ phi) / (sigma ** 2)
        m_b = np.linalg.solve(prec_b, (phi.T @ y_test) / (sigma ** 2))
        norms_gated.append(float(np.linalg.norm(m_g)))
        norms_boed.append(float(np.linalg.norm(m_b)))
    assert max(norms_gated) < 25.0
    assert abs(norms_gated[-1] - norms_gated[-2]) < 1e-3
    assert norms_boed[-1] > 5e4


def test_unconditional_rate_envelope_bounds() -> None:
    """Verify omega_t stays inside [0.25*omega0(5*sigma), 2*omega0(2*sigma)] for all inputs."""
    cal = design_time_lhw_imq(lambda th, xi: np.ones((th.shape[0], xi.shape[0])), np.zeros((4, 2)), np.zeros((3, 1)))
    lo = 0.25 * cal.const_5.lhw_ratio
    hi = 2.0 * cal.const_2.lhw_ratio
    for t in (1, 3, 4, 10, 30):
        for d0 in (0.0, 1.0, 16.0, 1e6):
            for sp in (0.0, float(t), 10.0 * t, 1e9):
                dec = unified_omega(t=t, p=2, sigma=1.0, delta0_sq=d0, s_pred=sp, calibration=cal)
                assert lo - 1e-12 <= dec.omega_t <= hi + 1e-12


def test_smc_gpu_end_to_end_smoke() -> None:
    """Fast end-to-end smoke test of run_smc_sequential_gpu on small R, T, M."""
    import sys
    from pathlib import Path
    import torch

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _smc_gpu import run_smc_sequential_gpu
    from run_nonlinear_benchmark import nonlin_candidate_fn, nonlin_mu_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma = 0.4
    cal = design_time_lhw_imq(
        sigma_fn=lambda th, xi: np.full((th.shape[0], xi.shape[0]), sigma, dtype=np.float64),
        theta_samples=np.zeros((8, 3)),
        design_grid=np.zeros((4, 2)),
    )
    gen = torch.Generator(device=device).manual_seed(77)
    theta_true = torch.randn((4, 3), device=device, generator=gen)
    test_designs = -3.0 + 6.0 * torch.rand((4, 40, 2), device=device, generator=gen)

    def dgp_fn(th: torch.Tensor, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        mu = nonlin_mu_fn(th.unsqueeze(1), x.unsqueeze(1)).squeeze(1).squeeze(1)
        return mu + sigma * torch.randn(mu.shape, device=device, generator=g)

    out1 = run_smc_sequential_gpu(
        method="GatedC_G_LHW",
        R=4,
        T=6,
        M=64,
        p=3,
        d_xi=2,
        bound_lo=-3.0,
        bound_hi=3.0,
        sigma=sigma,
        mu_fn=nonlin_mu_fn,
        theta_true=theta_true,
        dgp_sample_fn=dgp_fn,
        calibration=cal,
        candidate_fn=nonlin_candidate_fn,
        test_designs=test_designs,
        sigma_eval=sigma,
        seed=99,
        n_mh_steps=4,
        device=device,
    )
    assert out1["mmd"].shape == (4,)
    assert out1["nll"].shape == (4,)
    assert np.all(np.isfinite(out1["mmd"]))
    assert np.all(np.isfinite(out1["nll"]))
    assert len(out1["omega_trace"]) == 6
    assert len(out1["c_trace"]) == 6


def test_p_imq_tail_gap_threshold_and_t_ge_3_annealed_bound() -> None:
    """Verify homoscedastic tail gap vanishes iff p_imq > 1/2 and t >= 3 annealed outlier bound."""
    rng = np.random.default_rng(17)
    theta = rng.normal(0.0, 1.0, size=(24, 2))
    designs = np.linspace(-2.0, 2.0, 7)[:, None]

    def homo_sig(th: np.ndarray, xi: np.ndarray) -> np.ndarray:
        return np.ones((th.shape[0], xi.shape[0]), dtype=np.float64)

    def linear_mu(th: np.ndarray, xi: np.ndarray) -> np.ndarray:
        return th[:, 0:1] + th[:, 1:2] * xi[None, :, 0]

    # p_imq < 0.5: infinite homoscedastic tail gap, rejected by unified_omega
    cal_025 = design_time_lhw_imq(homo_sig, theta, designs, p_imq=0.25, mu_fn=linear_mu)
    assert math.isinf(cal_025.delta_inf)
    with pytest.raises(ValueError, match="p_imq="):
        unified_omega(t=5, p=2, sigma=1.0, delta0_sq=0.0, s_pred=5.0, calibration=cal_025)
    with pytest.raises(ValueError, match="p_imq="):
        design_time_lhw_imq(homo_sig, theta, designs, p_imq=0.25, raise_on_heteroscedastic=True)

    # p_imq == 0.5: exact boundary plateau 2 * c / sigma^4 * (mu_max - mu_min)
    cal_050 = design_time_lhw_imq(homo_sig, theta, designs, p_imq=0.5, mu_fn=linear_mu)
    mus = linear_mu(theta, designs)
    expected_gap = float(np.max(2.0 * 2.0 * (np.max(mus, axis=0) - np.min(mus, axis=0))))
    assert math.isclose(cal_050.delta_inf, expected_gap, rel_tol=1e-12)
    with pytest.raises(ValueError, match="p_imq="):
        unified_omega(t=5, p=2, sigma=1.0, delta0_sq=0.0, s_pred=5.0, calibration=cal_050)

    # p_imq > 0.5: homoscedastic tail gap vanishes
    cal_100 = design_time_lhw_imq(homo_sig, theta, designs, p_imq=1.0, mu_fn=linear_mu)
    assert cal_100.delta_inf == 0.0

    # Minimal majority sample size t = 3 with 1 corrupted observation (1 < 3/2): bounded shift
    x3 = np.array([-1.0, 0.0, 1.0])
    phi3 = np.column_stack([np.ones(3), x3])
    beta_true = np.array([4.0, -2.5])
    sigma = 0.5
    omega = cal_100.const_2.lhw_ratio * (sigma ** 2)
    y_clean = phi3 @ beta_true
    m_clean, _, _ = annealed_conjugate_predc_posterior(phi3, y_clean, sigma=sigma, c_target=2.0 * sigma, omega_target=omega)
    shifts = []
    for y_out in (1e4, 1e6, 1e8):
        y_corr = y_clean.copy()
        y_corr[0] = y_out
        m_corr, _, _ = annealed_conjugate_predc_posterior(phi3, y_corr, sigma=sigma, c_target=2.0 * sigma, omega_target=omega)
        shifts.append(float(np.linalg.norm(m_corr - m_clean)))
    assert max(shifts) < 15.0
    assert abs(shifts[-1] - shifts[-2]) < 1e-3
