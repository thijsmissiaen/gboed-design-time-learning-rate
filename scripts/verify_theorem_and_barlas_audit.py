"""Rigorous ball-arithmetic (flint.arb) quadrature, analytical theorem verification,
and forensic audit of Barlas et al. (arXiv:2511.07671, yasirbarlas/GBOED @ c2b9c63).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import flint
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from gboed_lr.design_time import (
    design_time_lhw_imq,
    imq_wsm_quadrature_constants,
)
from gboed_lr.rule import (
    annealed_conjugate_predc_posterior,
    conjugate_predc_imq_wsm,
)


def pk_mu_and_sigma(
    theta_log: np.ndarray,
    xi_t: np.ndarray,
    D_v: float = 400.0,
    eps_scale: float = 0.01,
    nu_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Barlas Appendix E.6 Pharmacokinetics mean and heteroscedastic std."""
    theta = np.exp(theta_log)
    k_a = theta[..., 0:1]
    k_e = theta[..., 1:2]
    vol = theta[..., 2:3]
    mu = (D_v / vol) * (k_a / np.maximum(k_a - k_e, 1e-6)) * (
        np.exp(-k_e * xi_t[None, :]) - np.exp(-k_a * xi_t[None, :])
    )
    sd = np.sqrt((mu ** 2) * eps_scale + nu_scale)
    return mu, sd


def pk_wsm_loss(
    theta_log: np.ndarray,
    xi_t: float,
    y: np.ndarray,
    mu_hat: float,
    c: float,
    p_imq: float = 1.0,
) -> np.ndarray:
    """Evaluate PredC IMQ-WSM loss on Barlas PK model across theta_log [M, 3] and y [N]."""
    mu, sd = pk_mu_and_sigma(theta_log, np.array([xi_t], dtype=np.float64))
    mu = mu[:, 0][:, None]
    sd2 = (sd[:, 0] ** 2)[:, None]
    y_b = y[None, :]

    dy_hat = y_b - mu_hat
    u = 1.0 + (dy_hat ** 2) / (c ** 2)
    m2 = u ** (-p_imq)
    dm2_dy = (-2.0 * p_imq * dy_hat / (c ** 2)) * (u ** (-p_imq - 1.0))

    s = (mu - y_b) / sd2
    ds_dy = -1.0 / sd2
    return m2 * (s ** 2) + 2.0 * (m2 * ds_dy + dm2_dy * s)


def certify_quadrature_arb(c_ratio: int, p_imq: int) -> dict:
    """Compute guaranteed flint.arb ball-arithmetic enclosures for H_B, J_B, omega0/s2, sup|psi_B|."""
    flint.ctx.dps = 50
    inv_sqrt_2pi = 1 / (2 * flint.arb.pi()).sqrt()
    R_int = flint.arb(16)
    r_acb = flint.acb(c_ratio)
    r_arb = flint.arb(c_ratio)

    def integrand_H(z: flint.acb, _analytic: bool) -> flint.acb:
        m2 = (1 + (z / r_acb) ** 2) ** (-p_imq)
        return 2 * m2 * (-z * z / 2).exp() * inv_sqrt_2pi

    def integrand_J(z: flint.acb, _analytic: bool) -> flint.acb:
        u = 1 + (z / r_acb) ** 2
        psi = z * (u ** (-p_imq)) + (2 * p_imq * z / (r_acb ** 2)) * (u ** (-p_imq - 1))
        return 4 * (psi ** 2) * (-z * z / 2).exp() * inv_sqrt_2pi

    H_core = flint.acb.integral(integrand_H, -16, 16, rel_tol=1e-40).real
    J_core = flint.acb.integral(integrand_J, -16, 16, rel_tol=1e-40).real

    gauss_R = (-R_int * R_int / 2).exp() * inv_sqrt_2pi
    tail_z0 = 2 * gauss_R / R_int
    tail_z2 = 2 * (R_int + 1 / R_int) * gauss_R
    H_arb = H_core + flint.arb(0, float(2 * tail_z0))
    coeff_J = 4 * (1 + 2 * p_imq / (r_arb ** 2)) ** 2
    J_arb = J_core + flint.arb(0, float(coeff_J * tail_z2))

    ratio_arb = H_arb / J_arb
    eff_arb = (H_arb ** 2) / J_arb

    # Exact stationary equation in flint.arb for p_imq >= 1
    a = flint.arb(1 - 2 * p_imq)
    b = 2 * flint.arb(p_imq) * (1 - flint.arb(2 * p_imq + 1) / (r_arb ** 2))
    c_coef = 4 * flint.arb(p_imq * (p_imq + 1)) / (r_arb ** 2)
    u_star = (b + (b * b - 4 * a * c_coef).sqrt()) / (2 * (2 * flint.arb(p_imq) - 1))
    w_star = (u_star - 1).sqrt()
    sup_arb = r_arb * w_star * (u_star ** (-p_imq) + (2 * p_imq / (r_arb ** 2)) * (u_star ** (-p_imq - 1)))

    qc_scipy = imq_wsm_quadrature_constants(float(c_ratio), p_imq=float(p_imq))
    return {
        "p_imq": float(p_imq),
        "c_ratio": float(c_ratio),
        "H_B": float(H_arb),
        "J_B": float(J_arb),
        "omega0_over_sigma2": float(ratio_arb),
        "H_B2_over_J_B": float(eff_arb),
        "J_B_over_H_B2": float(J_arb / (H_arb ** 2)),
        "sup_abs_psi_B": float(sup_arb),
        "arb_intervals": {
            "H_B": H_arb.str(22),
            "J_B": J_arb.str(22),
            "omega0_over_sigma2": ratio_arb.str(22),
            "H_B2_over_J_B": eff_arb.str(22),
            "sup_abs_psi_B": sup_arb.str(22),
        },
        "scipy_cross_check_max_diff": float(
            max(
                abs(float(H_arb) - qc_scipy.H_B),
                abs(float(J_arb) - qc_scipy.J_B),
                abs(float(ratio_arb) - qc_scipy.lhw_ratio),
                abs(float(sup_arb) - qc_scipy.sup_psi_B),
            )
        ),
        "quad_err_H_scipy": qc_scipy.quad_err_H,
        "quad_err_J_scipy": qc_scipy.quad_err_J,
    }


def main() -> None:
    out_path = Path("results/audit_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Certified flint.arb quadrature constants + unconditional rate envelope
    quad_table = {}
    for p_imq in (1, 2):
        for c_ratio in (2, 5):
            quad_table[f"p{p_imq}_c{c_ratio}"] = certify_quadrature_arb(c_ratio, p_imq)

    rate_envelope = {
        "omega_min_over_sigma2": 0.25 * quad_table["p1_c5"]["omega0_over_sigma2"],
        "omega_max_over_sigma2": 2.0 * quad_table["p1_c2"]["omega0_over_sigma2"],
        "gamma_inf_min": 0.25,
        "gamma_inf_max": 2.0,
    }

    # 2. Stein Fisher consistency at arbitrary fixed mu_hat != mu* + Conjugate vs BFGS
    sigma_stein = 0.8
    c_stein = 2.0 * sigma_stein
    mu_star = 3.2
    stein_checks = []
    for shift_sig in (-10.0, -3.0, -0.5, 0.0, 2.0, 8.0):
        mu_hat = mu_star + shift_sig * sigma_stein

        def score_integrand(y: float) -> float:
            r = y - mu_hat
            u = 1.0 + (r / c_stein) ** 2
            m2 = 1.0 / u
            dm2 = -2.0 * r / (c_stein ** 2 * u * u)
            d_mu_ell = 2.0 * m2 * (mu_star - y) / (sigma_stein ** 4) + 2.0 * dm2 / (sigma_stein ** 2)
            return d_mu_ell * float(norm.pdf(y, loc=mu_star, scale=sigma_stein))

        val, err = quad(score_integrand, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13)
        stein_checks.append({"mu_hat_shift_sigma": shift_sig, "expected_score": float(val), "quad_err": float(err)})

    rng_bf = np.random.default_rng(2026)
    x_bf = np.linspace(-3.0, 3.0, 9)
    phi_bf = np.column_stack([np.ones_like(x_bf), x_bf, x_bf ** 2])
    y_bf = phi_bf @ np.array([2.5, -1.7, 0.4]) + rng_bf.normal(0.0, sigma_stein, size=9)
    mu0_bf = np.array([0.5, -0.3, 0.1])
    prec0_bf = np.diag([0.8, 1.5, 2.2])
    mu_hat_bf = phi_bf @ np.array([2.1, -1.5, 0.3])
    w_bf = quad_table["p1_c2"]["omega0_over_sigma2"] * (sigma_stein ** 2)
    m_conj, _ = conjugate_predc_imq_wsm(phi_bf, y_bf, mu_hat_bf, sigma=sigma_stein, c=c_stein, omega=w_bf, mu0=mu0_bf, sigma0_inv=prec0_bf)

    def gibbs_obj(th: np.ndarray) -> float:
        d = th - mu0_bf
        dy = y_bf - mu_hat_bf
        u = 1.0 + (dy / c_stein) ** 2
        m2 = 1.0 / u
        dm2 = -2.0 * dy / (c_stein ** 2 * u * u)
        s = (phi_bf @ th - y_bf) / (sigma_stein ** 2)
        ell = m2 * (s ** 2) + 2.0 * (-m2 / (sigma_stein ** 2) + dm2 * s)
        return 0.5 * float(d @ prec0_bf @ d) + w_bf * float(np.sum(ell))

    opt_bf = minimize(gibbs_obj, np.zeros(3), method="BFGS", options={"gtol": 1e-8})
    h_fd = 1e-4
    grad_fd = np.array([(gibbs_obj(m_conj + h_fd * np.eye(3)[k]) - gibbs_obj(m_conj - h_fd * np.eye(3)[k])) / (2.0 * h_fd) for k in range(3)])
    conjugate_vs_bfgs = {
        "param_diff_l2": float(np.linalg.norm(m_conj - opt_bf.x)),
        "central_fd_grad_norm": float(np.linalg.norm(grad_fd)),
    }

    # 3. 1D conjugate conflict-floor Theorem 2.6: BOTH NLL and sample-pair MSE derivatives
    deriv_checks = []
    r0 = 1.0
    delta0_sq = 1.0
    eps = 1e-7
    for kappa in (1.0, 1.5, 2.0, 4.0, 6.25):
        for T in (20, 50, 200, 1000):
            def enll(u: float) -> float:
                mse_over_s2 = ((r0 * u) ** 2) * delta0_sq + ((1.0 - r0 * u) ** 2) * kappa / float(T)
                return 0.5 * math.log(2.0 * math.pi) + 0.5 * math.log(1.0 + u) + (kappa + mse_over_s2) / (2.0 * (1.0 + u))

            def emse_over_s2(u: float) -> float:
                return 1.0 + u + kappa + ((r0 * u) ** 2) * delta0_sq + ((1.0 - r0 * u) ** 2) * kappa / float(T)

            num_nll = (enll(eps) - enll(-eps)) / (2.0 * eps)
            exact_nll = 0.5 * (1.0 - kappa) - kappa * (1.0 + 2.0 * r0) / (2.0 * float(T))
            num_mse = (emse_over_s2(eps) - emse_over_s2(-eps)) / (2.0 * eps)
            exact_mse = 1.0 - (2.0 * r0 * kappa) / float(T)
            deriv_checks.append({
                "kappa": kappa,
                "T": T,
                "nll_num_deriv": num_nll,
                "nll_exact_deriv": exact_nll,
                "nll_abs_err": abs(num_nll - exact_nll),
                "mse_over_s2_num_deriv": num_mse,
                "mse_over_s2_exact_deriv": exact_mse,
                "mse_abs_err": abs(num_mse - exact_mse),
            })

    taus = np.linspace(0.1, 2.0, 77)
    T_curve = 20
    nll_curves = {}
    for kappa in (1.0, 2.0, 4.0):
        for d0 in (0.0, 4.0, 16.0, 64.0):
            vals = []
            for tau in taus:
                u = 1.0 / (r0 + tau * T_curve)
                mse_over_s2 = ((r0 * u) ** 2) * d0 + ((1.0 - r0 * u) ** 2) * kappa / float(T_curve)
                vals.append(0.5 * math.log(2.0 * math.pi) + 0.5 * math.log(1.0 + u) + (kappa + mse_over_s2) / (2.0 * (1.0 + u)))
            nll_curves[f"kappa{kappa}_d0_{int(d0)}"] = vals

    # 4. PredC Attenuation & Huber Gross-Error Outlier Saturation
    delta_pred_sig = 30.0
    m2_attenuation = 1.0 / (1.0 + (delta_pred_sig / 2.0) ** 2)
    x_hub = np.linspace(-4.0, 4.0, 12)
    phi_hub = np.column_stack([np.ones(12), x_hub])
    beta_hub = np.array([10.0, -7.0])
    sig_hub = 0.5
    y_clean_hub = phi_hub @ beta_hub
    w_hub = quad_table["p1_c2"]["omega0_over_sigma2"] * (sig_hub ** 2)
    huber_shift_sweep = []
    for y_out in (1e1, 1e2, 1e4, 1e6):
        y_tmp = y_clean_hub.copy()
        y_tmp[0] = y_out
        m_g, _, _ = annealed_conjugate_predc_posterior(phi_hub, y_tmp, sigma=sig_hub, c_target=2.0 * sig_hub, omega_target=w_hub)
        prec_b = np.eye(2) + (phi_hub.T @ phi_hub) / (sig_hub ** 2)
        m_b = np.linalg.solve(prec_b, (phi_hub.T @ y_tmp) / (sig_hub ** 2))
        huber_shift_sweep.append({
            "y_outlier": float(y_out),
            "gatedc_norm": float(np.linalg.norm(m_g)),
            "gatedc_err_vs_clean": float(np.linalg.norm(m_g - beta_hub)),
            "boed_norm": float(np.linalg.norm(m_b)),
        })

    # 5. Forensic audit of yasirbarlas/GBOED @ c2b9c63dd8c623320d4d43aefde366ca671948f0
    w_samples = {}
    for w_in in (0.2, 0.4):
        w_samples[f"w_{w_in}"] = {
            "input_w": w_in,
            "barlas_dm_weighted_internal_factor_w2": w_in ** 2,
            "barlas_log_potential_external_factor_w": w_in,
            "barlas_effective_multiplier_w3": w_in ** 3,
            "attenuation_ratio_vs_intended": w_in ** 2,
        }
    barlas_repo_audit = {
        "commit_sha": "c2b9c63dd8c623320d4d43aefde366ca671948f0",
        "predc_already_present_in_repo": {
            "file": "utils/loss_functions.py",
            "lines": [83, 144, 222],
            "note": "mean = torch.mean(predictive, dim=0) across regression, location, and pharmacokinetic classes",
        },
        "triple_counting_files": {
            "utils/loss_functions.py": [90, 92, 150, 151, 228, 229],
            "regression.py": [340],
            "location.py": [406],
            "pharmacokinetic.py": [419],
            "utils/gibbs_eig.py": [59, 92],
        },
        "numerical_examples": w_samples,
    }

    # 6. Heteroscedastic PK finite positive tail gap & blow-up
    rng_pk = np.random.default_rng(123)
    mu_pk_prior = np.log(np.array([1.0, 0.1, 20.0]))
    theta_pk_samples = rng_pk.normal(loc=mu_pk_prior, scale=math.sqrt(0.05), size=(128, 3))
    theta_pk_samples = theta_pk_samples[np.exp(theta_pk_samples[:, 0]) > np.exp(theta_pk_samples[:, 1])]
    xi_grid = np.linspace(0.5, 24.0, 24)[:, None]

    pk_cal = design_time_lhw_imq(
        sigma_fn=lambda th, xi: pk_mu_and_sigma(th, xi[:, 0])[1],
        theta_samples=theta_pk_samples,
        design_grid=xi_grid,
        raise_on_heteroscedastic=False,
    )
    _, pk_sig_grid = pk_mu_and_sigma(theta_pk_samples, xi_grid[:, 0])
    pk_c_grid = 2.0 * np.median(pk_sig_grid, axis=0)
    pk_prior_grid_upper = float(np.max(100.0 * (pk_c_grid ** 2)))

    theta_star = np.log(np.array([[1.5, 0.15, 15.0]]))
    theta_alt = np.log(np.array([[2.2, 0.08, 10.0]]))
    th_pair = np.vstack([theta_star, theta_alt])
    xi_eval = 2.0
    mu_pair, sd_pair = pk_mu_and_sigma(th_pair, np.array([xi_eval]))
    mu_hat_pk = float(mu_pair[0, 0])
    c_pk = 2.0 * float(sd_pair[0, 0])
    y_sweep = np.array([5.0, 20.0, 50.0, 100.0, 500.0, 2000.0, 10000.0])

    pk_tail_p1 = pk_wsm_loss(th_pair, xi_eval, y_sweep, mu_hat_pk, c_pk, p_imq=1.0)
    pk_tail_p2 = pk_wsm_loss(th_pair, xi_eval, y_sweep, mu_hat_pk, c_pk, p_imq=2.0)
    asym_lim = (c_pk ** 2) / (sd_pair[:, 0] ** 4)

    y_out_pk = np.array([mu_hat_pk - 5.0])
    loss_prior_p1 = pk_wsm_loss(theta_pk_samples, xi_eval, y_out_pk, mu_hat_pk, c_pk, p_imq=1.0)[:, 0]
    loss_star_p1 = float(pk_wsm_loss(theta_star, xi_eval, y_out_pk, mu_hat_pk, c_pk, p_imq=1.0)[0, 0])
    loss_prior_p2 = pk_wsm_loss(theta_pk_samples, xi_eval, y_out_pk, mu_hat_pk, c_pk, p_imq=2.0)[:, 0]
    loss_star_p2 = float(pk_wsm_loss(theta_star, xi_eval, y_out_pk, mu_hat_pk, c_pk, p_imq=2.0)[0, 0])

    pk_audit = {
        "design_time_homoscedastic": pk_cal.homoscedastic,
        "design_time_xi_constant": pk_cal.xi_constant,
        "design_time_rel_sigma_variation": pk_cal.rel_sigma_variation,
        "design_time_delta_inf_finite_lower_bound": pk_cal.delta_inf,
        "design_time_delta_inf_prior_grid_upper_bound": pk_prior_grid_upper,
        "xi_eval_delta_inf_upper_bound_100_c_pk_sq": float(100.0 * (c_pk ** 2)),
        "xi_eval": xi_eval,
        "mu_star": float(mu_pair[0, 0]),
        "sigma_star": float(sd_pair[0, 0]),
        "mu_alt": float(mu_pair[1, 0]),
        "sigma_alt": float(sd_pair[1, 0]),
        "c_pk": c_pk,
        "y_sweep": y_sweep.tolist(),
        "p1_loss_star": pk_tail_p1[0].tolist(),
        "p1_loss_alt": pk_tail_p1[1].tolist(),
        "p1_tail_gap_star_minus_alt": (pk_tail_p1[0] - pk_tail_p1[1]).tolist(),
        "p1_analytic_limit_c2_over_sd4_star": float(asym_lim[0]),
        "p1_analytic_limit_c2_over_sd4_alt": float(asym_lim[1]),
        "p2_loss_star": pk_tail_p2[0].tolist(),
        "p2_loss_alt": pk_tail_p2[1].tolist(),
        "p1_loss_gap_at_y_minus_5_max_bias": float(loss_star_p1 - np.min(loss_prior_p1)),
        "p2_loss_gap_at_y_minus_5_max_bias": float(loss_star_p2 - np.min(loss_prior_p2)),
    }

    # 7. Homoscedastic p_IMQ tail gap checks across p_imq in {0.25, 0.5, 1.0}
    homoscedastic_p_imq_checks = {}
    sig_gap, c_gap, mu_hat_gap, mu1_gap, mu2_gap = 1.0, 2.0, 0.0, 0.0, 1.0
    ys_gap = np.array([1e2, 1e4, 1e6], dtype=np.float64)
    for p_val in (0.25, 0.5, 1.0):
        def _loss_diff(y_arr: np.ndarray, p_cur: float = p_val) -> np.ndarray:
            r = y_arr - mu_hat_gap
            u = 1.0 + (r / c_gap) ** 2
            m2 = u ** (-p_cur)
            dm2 = (-2.0 * p_cur * r / (c_gap ** 2)) * (u ** (-p_cur - 1.0))
            s1 = (mu1_gap - y_arr) / (sig_gap ** 2)
            s2 = (mu2_gap - y_arr) / (sig_gap ** 2)
            ds = -1.0 / (sig_gap ** 2)
            l1 = m2 * (s1 ** 2) + 2.0 * (m2 * ds + dm2 * s1)
            l2 = m2 * (s2 ** 2) + 2.0 * (m2 * ds + dm2 * s2)
            return np.abs(l1 - l2)

        diffs_gap = _loss_diff(ys_gap)
        homoscedastic_p_imq_checks[f"p_{p_val}"] = {
            "y_values": ys_gap.tolist(),
            "abs_loss_diff": diffs_gap.tolist(),
            "analytic_limit": float(2.0 * c_gap * abs(mu1_gap - mu2_gap) / (sig_gap ** 4)) if p_val == 0.5 else (0.0 if p_val > 0.5 else "inf"),
        }

    payload = {
        "quadrature_constants": quad_table,
        "rate_envelope": rate_envelope,
        "stein_fisher_consistency": stein_checks,
        "conjugate_vs_bfgs": conjugate_vs_bfgs,
        "theorem26_derivative_checks": deriv_checks,
        "nll_finite_sample_curves": {
            "T": T_curve,
            "taus": taus.tolist(),
            "curves": nll_curves,
        },
        "predc_attenuation_m2_at_30sigma": m2_attenuation,
        "huber_gross_error_saturation": huber_shift_sweep,
        "barlas_repo_audit": barlas_repo_audit,
        "pk_heteroscedastic_audit": pk_audit,
        "homoscedastic_p_imq_tail_gap_checks": homoscedastic_p_imq_checks,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== flint.arb Certified Quadrature Constants ===")
    for k, v in quad_table.items():
        print(
            f"  {k}: w0/s2={v['omega0_over_sigma2']:.12f}, eff={v['H_B2_over_J_B']:.9f}, "
            f"sup|psi|={v['sup_abs_psi_B']:.9f}, scipy_diff={v['scipy_cross_check_max_diff']:.2e}"
        )
    print(f"=== Rate Envelope p=1: [{rate_envelope['omega_min_over_sigma2']:.9f}, {rate_envelope['omega_max_over_sigma2']:.9f}] * sigma^2 ===")
    print(f"=== Stein Fisher max |E[score]|: {max(abs(c['expected_score']) for c in stein_checks):.3e} ===")
    print(f"=== Conjugate vs BFGS: param_diff={conjugate_vs_bfgs['param_diff_l2']:.3e}, fd_grad={conjugate_vs_bfgs['central_fd_grad_norm']:.3e} ===")
    print(
        f"=== Theorem 2.6 max errors: NLL={max(d['nll_abs_err'] for d in deriv_checks):.3e}, "
        f"MSE={max(d['mse_abs_err'] for d in deriv_checks):.3e} ==="
    )
    print(f"=== PK finite Delta_inf lower bound={pk_cal.delta_inf:.4f}, tail at y=10000: {pk_tail_p1[0, -1]:.6f} vs {asym_lim[0]:.6f} ===")
    print(f"Saved audit results to {out_path}")


if __name__ == "__main__":
    main()
