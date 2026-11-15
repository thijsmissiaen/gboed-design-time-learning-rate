"""Design-time homoscedasticity certification and LHW-IMQ quadrature calibration.

This module uses ONLY design-time specifications (prior samples, likelihood
model sigma_theta(xi), and design space X). It never accesses the true data-
generating process or scenario labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


@dataclass(frozen=True)
class QuadratureConstants:
    """Exact 1D Gaussian quadrature constants for PredC IMQ-WSM at ratio c/sigma."""

    c_ratio: float
    p_imq: float
    H_B: float
    J_B: float
    lhw_ratio: float  # H_B / J_B
    var_inflation: float  # J_B / H_B^2
    sup_psi_B: float
    quad_err_H: float
    quad_err_J: float


@dataclass(frozen=True)
class DesignTimeCalibration:
    """Frozen design-time calibration object for GatedC G-LHW."""

    homoscedastic: bool
    xi_constant: bool
    sigma_ref: float
    rel_sigma_variation: float
    rel_xi_variation: float
    delta_inf: float
    p_imq: float
    const_2: QuadratureConstants
    const_5: QuadratureConstants
    omega0_2: float
    omega0_5: float


def _m2(z: np.ndarray | float, c_ratio: float, p_imq: float = 1.0) -> np.ndarray | float:
    return (1.0 + (z / c_ratio) ** 2) ** (-p_imq)


def _dm2_dz(z: np.ndarray | float, c_ratio: float, p_imq: float = 1.0) -> np.ndarray | float:
    return (-2.0 * p_imq * z / (c_ratio ** 2)) * (1.0 + (z / c_ratio) ** 2) ** (-p_imq - 1.0)


def psi_B_standardized(z: np.ndarray | float, c_ratio: float, p_imq: float = 1.0) -> np.ndarray | float:
    """Standardized influence function psi_B(z) = z * m0(z)^2 - (m0^2)'(z)."""
    return z * _m2(z, c_ratio, p_imq) - _dm2_dz(z, c_ratio, p_imq)


def sup_psi_B_exact(c_ratio: float, p_imq: float = 1.0) -> float:
    """Exact closed-form global supremum sup_{z in R} |psi_B(z; c_ratio, p_imq)|.

    For z = r * w and u = 1 + w^2, stationary points of psi_B(r*w) solve
    (1 - 2p) u^2 + 2p(1 - (2p+1)/r^2) u + 4p(p+1)/r^2 = 0.
    The influence function is uniformly bounded if and only if p_imq >= 1/2.
    """
    r = float(c_ratio)
    p = float(p_imq)
    if r <= 0.0:
        raise ValueError(f"c_ratio must be positive, got {r}")
    if p <= 0.0:
        raise ValueError(f"p_imq must be positive, got {p}")
    if p < 0.5 - 1e-14:
        return float("inf")

    r2 = r * r
    if abs(p - 0.5) <= 1e-14:
        # Linear stationary equation: (1 - 2/r^2) u + 3/r^2 = 0, tail limit |psi_B| -> r
        if r2 >= 2.0 - 1e-14:
            return r
        u_star = 3.0 / (2.0 - r2)
        w_star = math.sqrt(u_star - 1.0)
        peak = r * w_star * (u_star ** (-p) + (2.0 * p / r2) * (u_star ** (-p - 1.0)))
        return max(r, peak)

    a = 1.0 - 2.0 * p
    b = 2.0 * p * (1.0 - (2.0 * p + 1.0) / r2)
    c_coef = 4.0 * p * (p + 1.0) / r2
    disc = b * b - 4.0 * a * c_coef
    u_star = (b + math.sqrt(disc)) / (2.0 * (2.0 * p - 1.0))
    w_star = math.sqrt(u_star - 1.0)
    return float(r * w_star * (u_star ** (-p) + (2.0 * p / r2) * (u_star ** (-p - 1.0))))


def imq_wsm_quadrature_constants(c_ratio: float, p_imq: float = 1.0) -> QuadratureConstants:
    """Compute exact 1D Gaussian quadrature integrals for PredC IMQ-WSM at z ~ N(0, 1)."""
    if c_ratio <= 0.0:
        raise ValueError(f"c_ratio must be positive, got {c_ratio}")
    if p_imq <= 0.0:
        raise ValueError(f"p_imq must be positive, got {p_imq}")

    h_val, h_err = quad(
        lambda z: 2.0 * _m2(z, c_ratio, p_imq) * norm.pdf(z),
        -np.inf,
        np.inf,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=200,
    )
    j_val, j_err = quad(
        lambda z: 4.0 * (psi_B_standardized(z, c_ratio, p_imq) ** 2) * norm.pdf(z),
        -np.inf,
        np.inf,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=200,
    )

    sup_psi = sup_psi_B_exact(c_ratio, p_imq)

    return QuadratureConstants(
        c_ratio=float(c_ratio),
        p_imq=float(p_imq),
        H_B=float(h_val),
        J_B=float(j_val),
        lhw_ratio=float(h_val / j_val),
        var_inflation=float(j_val / (h_val ** 2)),
        sup_psi_B=sup_psi,
        quad_err_H=float(h_err),
        quad_err_J=float(j_err),
    )


def _evaluate_sigma_grid(
    sigma_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_samples: np.ndarray,
    design_grid: np.ndarray,
    rtol: float = 1e-6,
) -> tuple[bool, bool, float, float, float, np.ndarray]:
    sigmas = np.asarray(sigma_fn(theta_samples, design_grid), dtype=np.float64)
    if sigmas.ndim < 2:
        raise ValueError(f"Expected sigma_fn output of shape (K, D, ...), got {sigmas.shape}")
    if sigmas.shape[0] < 2:
        raise ValueError(f"Need at least 2 prior theta samples to certify homoscedasticity, got {sigmas.shape[0]}")
    if not np.all(np.isfinite(sigmas)):
        raise ValueError("Observation noise scale sigma_theta(xi) must be finite.")
    sigma_min_per_xi = np.min(sigmas, axis=0)
    sigma_max_per_xi = np.max(sigmas, axis=0)
    if np.any(sigma_min_per_xi <= 0.0):
        raise ValueError("Observation noise scale sigma_theta(xi) must be strictly positive.")

    rel_theta_var = float(np.max((sigma_max_per_xi - sigma_min_per_xi) / sigma_min_per_xi))
    med_per_xi = np.median(sigmas, axis=0)
    rel_xi_var = float((np.max(med_per_xi) - np.min(med_per_xi)) / np.min(med_per_xi))
    is_theta_homo = rel_theta_var <= rtol
    is_xi_const = rel_xi_var <= rtol
    sigma_ref = float(np.median(sigmas))
    return is_theta_homo, is_xi_const, sigma_ref, rel_theta_var, rel_xi_var, sigmas


def certify_scale_invariance(
    sigma_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_samples: np.ndarray,
    design_grid: np.ndarray,
    rtol: float = 1e-6,
) -> tuple[bool, bool, float, float, float]:
    """Certify both theta-independence and design-constancy of sigma_theta(xi).

    Returns:
        (is_theta_homoscedastic, is_xi_constant, sigma_ref, rel_theta_variation, rel_xi_variation)
    """
    is_homo, is_xi, sigma_ref, rel_th, rel_xi, _ = _evaluate_sigma_grid(
        sigma_fn=sigma_fn,
        theta_samples=theta_samples,
        design_grid=design_grid,
        rtol=rtol,
    )
    return is_homo, is_xi, sigma_ref, rel_th, rel_xi


def certify_homoscedasticity(
    sigma_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_samples: np.ndarray,
    design_grid: np.ndarray,
    rtol: float = 1e-6,
) -> tuple[bool, float, float]:
    """Certify whether model observation scale sigma_theta(xi) is theta-independent.

    Args:
        sigma_fn: Callable (theta_batch [K, ...], xi_batch [D, ...]) -> sigma [K, D].
        theta_samples: Prior parameter samples [K, ...].
        design_grid: Candidate designs [D, ...].
        rtol: Maximum relative variation across theta allowed for homoscedasticity.

    Returns:
        (is_homoscedastic, sigma_ref, max_rel_theta_variation)
    """
    is_homo, _, sigma_ref, rel_th, _, _ = _evaluate_sigma_grid(
        sigma_fn=sigma_fn,
        theta_samples=theta_samples,
        design_grid=design_grid,
        rtol=rtol,
    )
    return is_homo, sigma_ref, rel_th


def design_time_lhw_imq(
    sigma_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_samples: np.ndarray,
    design_grid: np.ndarray,
    p_imq: float = 1.0,
    rtol: float = 1e-6,
    raise_on_heteroscedastic: bool = False,
    mu_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> DesignTimeCalibration:
    """Perform design-time certification and exact LHW-IMQ quadrature calibration.

    Checks both theta-independence of sigma_theta(xi) (non-vanishing tail gap Delta_infty > 0
    when p_imq == 1 or homoscedastic p_imq <= 0.5) and xi-constancy required for scalar calibration.
    """
    is_homo, xi_const, sigma_ref, rel_th_var, rel_xi_var, sigmas = _evaluate_sigma_grid(
        sigma_fn=sigma_fn,
        theta_samples=theta_samples,
        design_grid=design_grid,
        rtol=rtol,
    )
    if is_homo:
        if p_imq > 0.5 + 1e-12:
            delta_inf = 0.0
        elif p_imq < 0.5 - 1e-12:
            delta_inf = float("inf")
        else:
            if mu_fn is not None:
                mus = np.asarray(mu_fn(theta_samples, design_grid), dtype=np.float64)
                mu_span = np.max(mus, axis=0) - np.min(mus, axis=0)
                s_med = np.median(sigmas, axis=0)
                c_xi = 2.0 * s_med
                delta_inf = float(np.max(2.0 * c_xi * mu_span / (s_med ** 4)))
            else:
                delta_inf = float("nan")
    elif p_imq < 1.0 - 1e-12:
        delta_inf = float("inf")
    elif abs(p_imq - 1.0) <= 1e-12:
        # Exact finite tail gap Delta_infty(xi) = c(xi)^2 * (sigma_min^-4 - sigma_max^-4)
        s_min = np.min(sigmas, axis=0)
        s_max = np.max(sigmas, axis=0)
        c_xi = 2.0 * np.median(sigmas, axis=0)
        delta_inf = float(np.max((c_xi ** 2) * (s_min ** (-4) - s_max ** (-4))))
    else:
        delta_inf = 0.0

    if raise_on_heteroscedastic and not is_homo:
        raise ValueError(
            f"Heteroscedastic likelihood across theta detected (rel_sigma_variation={rel_th_var:.3e} > {rtol:.1e}): "
            f"PredC IMQ-WSM at p_imq={p_imq} has non-vanishing tail bias gap Delta_inf={delta_inf:.6f}."
        )
    if raise_on_heteroscedastic and not xi_const:
        raise ValueError(
            f"Design-dependent noise scale sigma(xi) detected (rel_xi_variation={rel_xi_var:.3e} > {rtol:.1e}): "
            "scalar LHW calibration requires per-design noise normalization."
        )
    if raise_on_heteroscedastic and p_imq <= 0.5 + 1e-12:
        raise ValueError(
            f"Kernel exponent p_imq={p_imq} <= 0.5 violates homoscedastic tail unbiasedness "
            f"(Delta_infty={delta_inf}); select p_imq > 0.5."
        )

    const_2 = imq_wsm_quadrature_constants(2.0, p_imq=p_imq)
    const_5 = imq_wsm_quadrature_constants(5.0, p_imq=p_imq)

    return DesignTimeCalibration(
        homoscedastic=is_homo,
        xi_constant=xi_const,
        sigma_ref=sigma_ref,
        rel_sigma_variation=rel_th_var,
        rel_xi_variation=rel_xi_var,
        delta_inf=delta_inf,
        p_imq=float(p_imq),
        const_2=const_2,
        const_5=const_5,
        omega0_2=const_2.lhw_ratio * (sigma_ref ** 2),
        omega0_5=const_5.lhw_ratio * (sigma_ref ** 2),
    )
