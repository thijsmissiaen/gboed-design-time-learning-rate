"""Frozen online conflict & chi-squared dispersion gate (unified_omega).

All hyperparameters are fixed a priori and identical across every benchmark.
Only design-time calibration and within-experiment quantities (x_{1:t}, y_{1:t}, q_t)
are consumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .design_time import DesignTimeCalibration


@dataclass(frozen=True)
class RuleDecision:
    """Output of the frozen unified_omega gate at sequential step t."""

    omega_t: float
    c_t: float
    omega0_t: float
    phi: float
    tau_t: float
    tau_min: float
    delta0_sq: float
    s_pred: float
    chi2_85: float | None
    chi2_95: float | None


def phi_conflict(delta0_sq: float | np.ndarray) -> float | np.ndarray:
    """Conflict-aware bias-relief multiplier phi(Delta_0^2) = 1 + D^4 / (D^4 + 16^2)."""
    raw = np.asarray(delta0_sq, dtype=np.float64)
    if np.any(np.isnan(raw)):
        raise ValueError("delta0_sq must not be NaN")
    d = np.maximum(raw, 0.0)
    # Overflow-safe rational evaluation: D^4 / (D^4 + 256) = 1 / (1 + (16/D^2)^2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.where(d > 1e-150, 1.0 / (1.0 + (16.0 / d) ** 2), 0.0)
        res = np.where(np.isposinf(d), 2.0, 1.0 + ratio)
    return float(res) if np.ndim(delta0_sq) == 0 else res


def tau_min_floor(delta0_sq: float | np.ndarray) -> float | np.ndarray:
    """Conflict-aware variance-inflation floor tau_min(Delta_0^2) = 1 - 0.75 / (1 + (D/4)^2)."""
    raw = np.asarray(delta0_sq, dtype=np.float64)
    if np.any(np.isnan(raw)):
        raise ValueError("delta0_sq must not be NaN")
    d = np.maximum(raw, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        res = np.where(np.isposinf(d), 1.0, 1.0 - 0.75 / (1.0 + (d / 4.0) ** 2))
    return float(res) if np.ndim(delta0_sq) == 0 else res


def compute_delta0_sq(
    theta_samples: np.ndarray,
    mu0: np.ndarray,
    sigma0_inv: np.ndarray,
) -> float:
    """Compute prior-data conflict index Delta_0^2 = (1/p) E_q[(theta-mu0)^T Sigma0^-1 (theta-mu0)].

    Args:
        theta_samples: Posterior samples of shape (M, p).
        mu0: Prior mean of shape (p,).
        sigma0_inv: Prior precision matrix of shape (p, p).
    """
    theta = np.asarray(theta_samples, dtype=np.float64)
    mu = np.asarray(mu0, dtype=np.float64)
    prec = np.asarray(sigma0_inv, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError(f"Expected theta_samples of shape (M, p), got {theta.shape}")
    p = theta.shape[1]
    diff = theta - mu[None, :]
    quad_forms = np.einsum("mi,ij,mj->m", diff, prec, diff)
    return float(np.mean(quad_forms) / float(p))


def compute_s_pred(
    y_obs: np.ndarray,
    pred_mean: np.ndarray,
    pred_var: np.ndarray,
    sigma: float,
) -> float:
    """Compute standardized posterior-predictive dispersion statistic S_t^pred."""
    y = np.asarray(y_obs, dtype=np.float64)
    mu = np.asarray(pred_mean, dtype=np.float64)
    var = np.maximum(np.asarray(pred_var, dtype=np.float64), 0.0)
    denom = (float(sigma) ** 2) + var
    return float(np.sum(((y - mu) ** 2) / denom))


def unified_omega(
    t: int,
    p: int,
    sigma: float,
    delta0_sq: float,
    s_pred: float,
    calibration: DesignTimeCalibration,
    allow_heteroscedastic_override: bool = False,
) -> RuleDecision:
    """Evaluate the frozen GatedC G-LHW learning-rate and scale-threshold gate.

    Parameters are strictly frozen across all benchmarks:
      - Warmup t < p + 2: c_t = 2*sigma, omega_t = omega0(2*sigma) * phi(delta0_sq)
      - Gated  t >= p + 2:
          tau_t = 1 if S_t^pred <= chi2_{t, 0.95} else max(tau_min(delta0_sq), t / S_t^pred)
          (c_t, omega0_t) = (5*sigma, omega0(5*sigma)) if S_t^pred <= chi2_{t, 0.85} else (2*sigma, omega0(2*sigma))
          omega_t = omega0_t * phi(delta0_sq) * tau_t
    """
    if not allow_heteroscedastic_override:
        if not calibration.homoscedastic:
            raise ValueError(
                "Cannot apply unified_omega to a non-certified heteroscedastic model "
                f"(non-vanishing tail gap Delta_infty={calibration.delta_inf:.4g})."
            )
        if not calibration.xi_constant:
            raise ValueError(
                "Cannot apply scalar unified_omega to a model with design-varying noise scale sigma(xi) "
                f"(rel_xi_variation={calibration.rel_xi_variation:.3e})."
            )
        if calibration.p_imq <= 0.5 + 1e-12:
            raise ValueError(
                f"Cannot apply unified_omega with p_imq={calibration.p_imq} <= 0.5 (non-vanishing tail gap)."
            )
    if t < 1:
        raise ValueError(f"Step t must be >= 1, got {t}")
    if p < 1:
        raise ValueError(f"Dimension p must be >= 1, got {p}")
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be finite and positive, got {sigma}")
    if math.isnan(delta0_sq):
        raise ValueError("delta0_sq must not be NaN")
    if math.isnan(s_pred):
        raise ValueError("s_pred must not be NaN")

    phi = float(phi_conflict(delta0_sq))
    t_min = float(tau_min_floor(delta0_sq))

    omega0_2 = calibration.const_2.lhw_ratio * (float(sigma) ** 2)
    omega0_5 = calibration.const_5.lhw_ratio * (float(sigma) ** 2)

    if t < p + 2:
        c_t = 2.0 * float(sigma)
        omega0_t = omega0_2
        tau_t = 1.0
        omega_t = omega0_t * phi
        return RuleDecision(
            omega_t=float(omega_t),
            c_t=float(c_t),
            omega0_t=float(omega0_t),
            phi=phi,
            tau_t=tau_t,
            tau_min=t_min,
            delta0_sq=float(delta0_sq),
            s_pred=float(s_pred),
            chi2_85=None,
            chi2_95=None,
        )

    q85 = float(chi2.ppf(0.85, df=t))
    q95 = float(chi2.ppf(0.95, df=t))
    s_val = float(s_pred)

    if s_val <= q95:
        tau_t = 1.0
    else:
        tau_t = max(t_min, float(t) / max(s_val, 1e-12))

    if s_val <= q85:
        c_t = 5.0 * float(sigma)
        omega0_t = omega0_5
    else:
        c_t = 2.0 * float(sigma)
        omega0_t = omega0_2

    omega_t = omega0_t * phi * tau_t
    return RuleDecision(
        omega_t=float(omega_t),
        c_t=float(c_t),
        omega0_t=float(omega0_t),
        phi=phi,
        tau_t=float(tau_t),
        tau_min=t_min,
        delta0_sq=float(delta0_sq),
        s_pred=s_val,
        chi2_85=q85,
        chi2_95=q95,
    )


def annealed_predc_scales(
    y_obs: np.ndarray,
    mu_init: np.ndarray,
    sigma: float,
    lambdas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.0),
) -> np.ndarray:
    """Compute the deterministic PredC scale-continuation ladder c^(s) down to 2*sigma.

    Prevents prior-center attenuation (Proposition 2.5) without external robust initializers
    by starting from c_max = max(2*sigma, median_j |y_j - mu_init_j| + 2*sigma).
    """
    y = np.asarray(y_obs, dtype=np.float64)
    mu0 = np.asarray(mu_init, dtype=np.float64)
    if y.shape != mu0.shape or y.ndim != 1:
        raise ValueError(f"Expected matching 1D arrays y_obs and mu_init, got {y.shape} and {mu0.shape}")
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be finite positive, got {sigma}")
    c_floor = 2.0 * float(sigma)
    med_abs = float(np.median(np.abs(y - mu0))) if y.size > 0 else 0.0
    c_max = max(c_floor, med_abs + c_floor)
    lam_arr = np.asarray(lambdas, dtype=np.float64)
    return c_floor + lam_arr * (c_max - c_floor)


def conjugate_predc_imq_wsm(
    design_matrix: np.ndarray,
    y_obs: np.ndarray,
    mu_hat: np.ndarray,
    sigma: float,
    c: float,
    omega: float,
    mu0: np.ndarray | None = None,
    sigma0_inv: np.ndarray | None = None,
    p_imq: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact closed-form Gaussian Gibbs posterior for PredC IMQ-WSM in linear models.

    Computes N(post_mean, post_cov) minimizing (1/2)||theta - mu0||_{Sigma0^-1}^2
    + omega * sum_j ell_B(Phi_j theta; y_j, mu_hat_j).
    """
    phi = np.asarray(design_matrix, dtype=np.float64)
    y = np.asarray(y_obs, dtype=np.float64)
    mh = np.asarray(mu_hat, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError(f"Expected 2D design_matrix (t, p), got {phi.shape}")
    t, p = phi.shape
    if y.shape != (t,) or mh.shape != (t,):
        raise ValueError(f"Expected y_obs and mu_hat of shape ({t},), got {y.shape} and {mh.shape}")
    mu_prior = np.zeros(p, dtype=np.float64) if mu0 is None else np.asarray(mu0, dtype=np.float64)
    prec_prior = np.eye(p, dtype=np.float64) if sigma0_inv is None else np.asarray(sigma0_inv, dtype=np.float64)

    dy = y - mh
    u = 1.0 + (dy * dy) / (float(c) ** 2)
    m2 = u ** (-float(p_imq))
    dm2_dy = (-2.0 * float(p_imq) * dy / (float(c) ** 2)) * (u ** (-float(p_imq) - 1.0))

    s2 = float(sigma) ** 2
    scale = 2.0 * float(omega) / (s2 * s2)
    prec = prec_prior + scale * ((phi.T * m2) @ phi)
    rhs = prec_prior @ mu_prior + scale * (phi.T @ (m2 * y - s2 * dm2_dy))
    cov = np.linalg.inv(prec)
    mean = np.linalg.solve(prec, rhs)
    return mean, cov


def annealed_conjugate_predc_posterior(
    design_matrix: np.ndarray,
    y_obs: np.ndarray,
    sigma: float,
    c_target: float,
    omega_target: float,
    mu0: np.ndarray | None = None,
    sigma0_inv: np.ndarray | None = None,
    lambdas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.0),
    p_imq: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Bayesian PredC conjugate posterior via deterministic scale continuation.

    Starts from the prior predictive center Phi * mu0 and anneals c from c_max down to 2*sigma,
    then evaluates at (c_target, omega_target).
    """
    phi = np.asarray(design_matrix, dtype=np.float64)
    p = phi.shape[1]
    mu_prior = np.zeros(p, dtype=np.float64) if mu0 is None else np.asarray(mu0, dtype=np.float64)
    mu_init = phi @ mu_prior
    scales = annealed_predc_scales(y_obs, mu_init, sigma, lambdas=lambdas)

    mu_hat = mu_init
    mean = mu_prior.copy()
    cov = np.linalg.inv(np.eye(p, dtype=np.float64) if sigma0_inv is None else np.asarray(sigma0_inv, dtype=np.float64))
    for c_s in scales:
        mean, cov = conjugate_predc_imq_wsm(
            phi,
            y_obs,
            mu_hat,
            sigma=sigma,
            c=float(c_s),
            omega=omega_target,
            mu0=mu_prior,
            sigma0_inv=sigma0_inv,
            p_imq=p_imq,
        )
        mu_hat = phi @ mean

    if abs(float(c_target) - 2.0 * float(sigma)) > 1e-12:
        mean, cov = conjugate_predc_imq_wsm(
            phi,
            y_obs,
            mu_hat,
            sigma=sigma,
            c=float(c_target),
            omega=omega_target,
            mu0=mu_prior,
            sigma0_inv=sigma0_inv,
            p_imq=p_imq,
        )
    return mean, cov, mu_hat
