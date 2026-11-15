"""Vectorized CUDA GPU SMC+MH engine, Gibbs NMC EIG acquisition, and Barlas predictive metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.stats import chi2

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from gboed_lr import DesignTimeCalibration, unified_omega


def systematic_resample_batch(weights: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Vectorized systematic resampling over batch dimension (R, M) -> indices (R, M)."""
    R, M = weights.shape
    device = weights.device
    cdf = torch.cumsum(weights, dim=1)
    cdf[:, -1] = 1.0
    u0 = torch.rand((R, 1), device=device, generator=generator, dtype=weights.dtype)
    u = (u0 + torch.arange(M, device=device, dtype=weights.dtype).unsqueeze(0)) / float(M)
    return torch.searchsorted(cdf, u, right=False).clamp(0, M - 1)


def weighted_mean_and_cov(particles: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted mean (R, p) and covariance (R, p, p) from (R, M, p) and (R, M)."""
    mean = torch.einsum("rm,rmp->rp", weights, particles)
    diff = particles - mean.unsqueeze(1)
    cov = torch.einsum("rm,rmp,rmq->rpq", weights, diff, diff)
    return mean, cov


def predc_imq_wsm_loss(
    mu_theta: torch.Tensor,
    y_obs: torch.Tensor,
    mu_hat: torch.Tensor,
    sigma: float,
    c: torch.Tensor,
    p_imq: float = 1.0,
) -> torch.Tensor:
    """Compute PredC IMQ-WSM loss ell_B(theta; y, x) vectorized over (R, M, t)."""
    inv_var = 1.0 / (sigma * sigma)
    c2 = c * c
    if c2.ndim == 1:
        c2 = c2.unsqueeze(-1)
    dy_hat = (y_obs - mu_hat).unsqueeze(1)
    c2_b = c2.unsqueeze(1)
    u = 1.0 + (dy_hat * dy_hat) / c2_b
    m2 = u.pow(-p_imq)
    dm2_dy = (-2.0 * p_imq * dy_hat / c2_b) * u.pow(-p_imq - 1.0)

    s = (mu_theta - y_obs.unsqueeze(1)) * inv_var
    ds_dy = -inv_var
    return m2 * (s * s) + 2.0 * (m2 * ds_dy + dm2_dy * s)


def _sample_laplace_torch(
    shape: tuple[int, ...],
    scale: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    u = torch.rand(shape, device=device, generator=generator).clamp(1e-7, 1.0 - 1e-7)
    shift = u - 0.5
    return -float(scale) * torch.sign(shift) * torch.log1p(-2.0 * torch.abs(shift))


@torch.no_grad()
def compute_barlas_metrics_gpu(
    post_particles: torch.Tensor,
    mu_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    theta_true: torch.Tensor,
    test_designs: torch.Tensor,
    sigma_model: float,
    sigma_true: float,
    true_noise_dist: str = "normal",
    n_eval: int = 1000,
    m_nll: int = 256,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Barlas Appendix F RMSE, MMD^2 (median heuristic), and NLL per replication."""
    if true_noise_dist not in ("normal", "laplace"):
        raise ValueError(f"true_noise_dist must be 'normal' or 'laplace', got {true_noise_dist}")

    R, M, p = post_particles.shape
    D = test_designs.shape[1]
    device = post_particles.device

    idx_pred = torch.randint(0, M, (R, n_eval), device=device, generator=generator)
    th_pred = torch.gather(post_particles, 1, idx_pred.unsqueeze(-1).expand(-1, -1, p))
    mu_pred = mu_fn(th_pred, test_designs)  # (R, n_eval, D)
    y_pred = mu_pred + sigma_model * torch.randn(mu_pred.shape, device=device, generator=generator)

    mu_true = mu_fn(theta_true.unsqueeze(1), test_designs)  # (R, 1, D)
    if true_noise_dist == "normal":
        y_true = mu_true + sigma_true * torch.randn((R, n_eval, D), device=device, generator=generator)
    else:
        y_true = mu_true + _sample_laplace_torch((R, n_eval, D), sigma_true, device=device, generator=generator)

    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=1)).mean(dim=1)  # (R,)

    mmd_accum = torch.zeros(R, device=device, dtype=torch.float32)
    batch_r = min(15, R)
    chunk_d = 10
    bw_n = min(250, n_eval)
    tri_i, tri_j = torch.triu_indices(2 * bw_n, 2 * bw_n, offset=1, device=device)

    for r0 in range(0, R, batch_r):
        r1 = min(R, r0 + batch_r)
        for d0 in range(0, D, chunk_d):
            d1 = min(D, d0 + chunk_d)
            xp = y_pred[r0:r1, :, d0:d1].transpose(1, 2)
            yt = y_true[r0:r1, :, d0:d1].transpose(1, 2)

            xy_sub = torch.cat([xp[:, :, :bw_n], yt[:, :, :bw_n]], dim=2)
            diffs_bw = xy_sub[:, :, tri_i] - xy_sub[:, :, tri_j]
            h_n = torch.median(diffs_bw * diffs_bw, dim=-1).values.clamp(min=1e-12)
            bw2 = h_n

            inv_2bw2 = (1.0 / bw2).unsqueeze(-1).unsqueeze(-1)
            d_xx = (xp.unsqueeze(-1) - xp.unsqueeze(-2)) ** 2
            d_yy = (yt.unsqueeze(-1) - yt.unsqueeze(-2)) ** 2
            d_xy = (xp.unsqueeze(-1) - yt.unsqueeze(-2)) ** 2

            sum_xx = (torch.exp(-d_xx * inv_2bw2).sum(dim=(-1, -2)) - float(n_eval)) / (n_eval * (n_eval - 1))
            sum_yy = (torch.exp(-d_yy * inv_2bw2).sum(dim=(-1, -2)) - float(n_eval)) / (n_eval * (n_eval - 1))
            sum_xy = 2.0 * torch.exp(-d_xy * inv_2bw2).mean(dim=(-1, -2))
            mmd_accum[r0:r1] += (sum_xx + sum_yy - sum_xy).sum(dim=1)

    mmd = mmd_accum / float(D)

    idx_nll = torch.randint(0, M, (R, m_nll), device=device, generator=generator)
    th_nll = torch.gather(post_particles, 1, idx_nll.unsqueeze(-1).expand(-1, -1, p))
    mu_nll = mu_fn(th_nll, test_designs)
    log_norm = -0.5 * math.log(2.0 * math.pi * sigma_model * sigma_model)
    inv_2s2 = 0.5 / (sigma_model * sigma_model)

    ll_accum = torch.zeros(R, device=device, dtype=torch.float32)
    for r0 in range(0, R, batch_r):
        r1 = min(R, r0 + batch_r)
        for d0 in range(0, D, chunk_d):
            d1 = min(D, d0 + chunk_d)
            diff = y_true[r0:r1, :, d0:d1].unsqueeze(1) - mu_nll[r0:r1, :, d0:d1].unsqueeze(2)
            lp = log_norm - inv_2s2 * (diff * diff)
            log_pred = torch.logsumexp(lp, dim=1) - math.log(float(m_nll))
            ll_accum[r0:r1] += log_pred.mean(dim=1).sum(dim=1)

    nll = -ll_accum / float(D)
    return rmse, mmd, nll


@torch.no_grad()
def select_eig_design_gpu(
    particles: torch.Tensor,
    mu_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    candidates: torch.Tensor,
    sigma: float,
    w_vec: torch.Tensor | None = None,
    n_outer: int = 256,
    m_inner: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select max-EIG design from candidates (R, C, d_xi) using vectorized (optionally tempered) NMC EIG."""
    R, M, p = particles.shape
    device = particles.device
    idx_out = torch.randint(0, M, (R, n_outer), device=device, generator=generator)
    idx_in = torch.randint(0, M, (R, m_inner), device=device, generator=generator)
    th_out = torch.gather(particles, 1, idx_out.unsqueeze(-1).expand(-1, -1, p))
    th_in = torch.gather(particles, 1, idx_in.unsqueeze(-1).expand(-1, -1, p))

    mu_out = mu_fn(th_out, candidates)  # (R, n_outer, C)
    mu_in = mu_fn(th_in, candidates)    # (R, m_inner, C)
    y_sim = mu_out + sigma * torch.randn(mu_out.shape, device=device, generator=generator)

    inv_2s2 = 0.5 / (sigma * sigma)
    log_norm = -0.5 * math.log(2.0 * math.pi * sigma * sigma)
    lp_cond = log_norm - inv_2s2 * ((y_sim - mu_out) ** 2)  # (R, n_outer, C)
    if w_vec is not None:
        w3 = w_vec.view(R, 1, 1)
        w4 = w_vec.view(R, 1, 1, 1)
    else:
        w3 = None
        w4 = None

    C = candidates.shape[1]
    eig = torch.empty((R, C), device=device, dtype=torch.float32)
    chunk_c = 32
    for c0 in range(0, C, chunk_c):
        c1 = min(C, c0 + chunk_c)
        diff = y_sim[:, :, c0:c1].unsqueeze(2) - mu_in[:, :, c0:c1].unsqueeze(1)
        lp_in = log_norm - inv_2s2 * (diff * diff)
        if w4 is not None:
            lp_in = w4 * lp_in
            lp_c = w3 * lp_cond[:, :, c0:c1]
        else:
            lp_c = lp_cond[:, :, c0:c1]
        lp_marg = torch.logsumexp(lp_in, dim=2) - math.log(float(m_inner))
        eig[:, c0:c1] = (lp_c - lp_marg).mean(dim=1)

    best_idx = torch.argmax(eig, dim=1)
    return torch.gather(candidates, 1, best_idx.view(R, 1, 1).expand(-1, -1, candidates.shape[-1])).squeeze(1)


@torch.no_grad()
def select_gibbs_eig_design_gpu(
    particles: torch.Tensor,
    mu_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    candidates: torch.Tensor,
    sigma: float,
    c_vec: torch.Tensor,
    w_vec: torch.Tensor,
    n_outer: int = 256,
    m_inner: int = 256,
    p_imq: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select max Gibbs-NMC EIG design from candidates (R, C, d_xi) under PredC IMQ-WSM."""
    R, M, p = particles.shape
    device = particles.device
    idx_out = torch.randint(0, M, (R, n_outer), device=device, generator=generator)
    idx_in = torch.randint(0, M, (R, m_inner), device=device, generator=generator)
    th_out = torch.gather(particles, 1, idx_out.unsqueeze(-1).expand(-1, -1, p))
    th_in = torch.gather(particles, 1, idx_in.unsqueeze(-1).expand(-1, -1, p))

    mu_out = mu_fn(th_out, candidates)  # (R, n_outer, C)
    mu_in = mu_fn(th_in, candidates)    # (R, m_inner, C)
    mu_hat_cand = mu_in.mean(dim=1)     # (R, C) PredC predictive mean
    y_sim = mu_out + sigma * torch.randn(mu_out.shape, device=device, generator=generator)

    inv_var = 1.0 / (sigma * sigma)
    c2 = (c_vec * c_vec).view(R, 1, 1)
    w_b3 = w_vec.view(R, 1, 1)
    w_b4 = w_vec.view(R, 1, 1, 1)

    C = candidates.shape[1]
    eig = torch.empty((R, C), device=device, dtype=torch.float32)
    chunk_c = 16
    for c0 in range(0, C, chunk_c):
        c1 = min(C, c0 + chunk_c)
        dy_c = y_sim[:, :, c0:c1] - mu_hat_cand[:, None, c0:c1]  # (R, n_outer, cc)
        u = 1.0 + (dy_c * dy_c) / c2
        m2 = u.pow(-p_imq)
        dm2_dy = (-2.0 * p_imq * dy_c / c2) * u.pow(-p_imq - 1.0)

        s_cond = (mu_out[:, :, c0:c1] - y_sim[:, :, c0:c1]) * inv_var
        loss_cond = m2 * (s_cond * s_cond) + 2.0 * (-inv_var * m2 + dm2_dy * s_cond)
        log_num = -w_b3 * loss_cond

        s_in = (mu_in[:, None, :, c0:c1] - y_sim[:, :, None, c0:c1]) * inv_var
        loss_in = m2[:, :, None, :] * (s_in * s_in) + 2.0 * (
            -inv_var * m2[:, :, None, :] + dm2_dy[:, :, None, :] * s_in
        )
        log_den = torch.logsumexp(-w_b4 * loss_in, dim=2) - math.log(float(m_inner))
        eig[:, c0:c1] = (log_num - log_den).mean(dim=1)

    best_idx = torch.argmax(eig, dim=1)
    return torch.gather(candidates, 1, best_idx.view(R, 1, 1).expand(-1, -1, candidates.shape[-1])).squeeze(1)


@torch.no_grad()
def run_smc_sequential_gpu(
    method: str,
    R: int,
    T: int,
    M: int,
    p: int,
    d_xi: int,
    bound_lo: float,
    bound_hi: float,
    sigma: float,
    mu_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    theta_true: torch.Tensor,
    dgp_sample_fn: Callable[[torch.Tensor, torch.Tensor, torch.Generator], torch.Tensor],
    calibration: DesignTimeCalibration | None,
    candidate_fn: Callable[[torch.Tensor, int, torch.Generator], torch.Tensor],
    test_designs: torch.Tensor,
    sigma_eval: float,
    seed: int,
    n_mh_steps: int = 14,
    perm_swap_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device = torch.device("cuda"),
    true_noise_dist: str = "normal",
    mu0: torch.Tensor | None = None,
    prec0: torch.Tensor | None = None,
    dgp_seed: int | None = None,
    eval_seed: int | None = None,
    barlas_params: tuple[float, float, float, float] = (1.8, 0.2, 0.06, 0.4),
) -> dict:
    """Run vectorized sequential BOED or GBOED baselines with SMC+MH on GPU.

    Supported methods:
      - BOED: Standard Bayesian log-loss + NMC EIG.
      - Tempered_LogLoss: Log-loss with chi2_{t,0.95} dispersion tempering tau_t + tempered NMC EIG.
      - Fixed_LHW: PredC IMQ-WSM at fixed c_t=2*sigma, omega_t=omega0(2*sigma) + Gibbs NMC EIG.
      - Barlas_ExpDecay: Exponential-decay c(t)=q1*exp(-b*(t-1))+q2 at fixed w + Gibbs NMC EIG.
      - GatedC_G_LHW: Frozen unified_omega (PredC IMQ-WSM) + Gibbs NMC EIG.
    """
    valid_methods = {"BOED", "Tempered_LogLoss", "Fixed_LHW", "Barlas_ExpDecay", "GatedC_G_LHW"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(valid_methods)}")

    gen = torch.Generator(device=device).manual_seed(seed)
    gen_dgp = torch.Generator(device=device).manual_seed(dgp_seed if dgp_seed is not None else seed + 7777)
    gen_eval = torch.Generator(device=device).manual_seed(eval_seed if eval_seed is not None else seed + 9999)

    mu0_t = torch.zeros(p, device=device, dtype=torch.float32) if mu0 is None else mu0.to(device=device, dtype=torch.float32)
    prec0_t = torch.eye(p, device=device, dtype=torch.float32) if prec0 is None else prec0.to(device=device, dtype=torch.float32)
    cov0_t = torch.linalg.inv(prec0_t)
    L0_t = torch.linalg.cholesky(cov0_t)
    particles = mu0_t.view(1, 1, p) + torch.einsum("ij,rmj->rmi", L0_t, torch.randn((R, M, p), device=device, generator=gen))
    log_w = torch.full((R, M), -math.log(float(M)), device=device)

    x_hist = torch.empty((R, 0, d_xi), device=device)
    y_hist = torch.empty((R, 0), device=device)

    def log_prior(th: torch.Tensor) -> torch.Tensor:
        d = th - mu0_t.view(1, 1, p)
        return -0.5 * torch.einsum("rmi,ij,rmj->rm", d, prec0_t, d)

    def log_target_boed(th: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, w_vec: torch.Tensor | None = None) -> torch.Tensor:
        lp = log_prior(th)
        if xs.shape[1] > 0:
            mu_th = mu_fn(th, xs)
            diff = ys.unsqueeze(1) - mu_th
            ll = (-0.5 * math.log(2.0 * math.pi * sigma * sigma) - 0.5 * (diff * diff) / (sigma * sigma)).sum(dim=-1)
            if w_vec is not None:
                lp = lp + w_vec.unsqueeze(-1) * ll
            else:
                lp = lp + ll
        return lp

    def log_target_gibbs(
        th: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        mu_hat: torch.Tensor,
        c_vec: torch.Tensor,
        w_vec: torch.Tensor,
    ) -> torch.Tensor:
        lp = log_prior(th)
        if xs.shape[1] > 0:
            mu_th = mu_fn(th, xs)
            losses = predc_imq_wsm_loss(mu_th, ys, mu_hat, sigma, c_vec, p_imq=1.0)
            lp = lp - w_vec.unsqueeze(-1) * losses.sum(dim=-1)
        return lp

    def mh_rejuvenate(
        th: torch.Tensor,
        target_fn: Callable[[torch.Tensor], torch.Tensor],
        cov_est: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        reg = 1e-4 * torch.eye(p, device=device).unsqueeze(0)
        cov_reg = cov_est + reg
        try:
            L = torch.linalg.cholesky(cov_reg)
        except Exception:
            L = 0.25 * torch.eye(p, device=device).unsqueeze(0).expand(R, -1, -1)

        cur_lp = target_fn(th)
        base_scale = 2.38 / math.sqrt(float(p))
        for step_idx in range(steps):
            scale = base_scale if (step_idx % 2 == 0) else 0.5 * base_scale
            eps = torch.randn(th.shape, device=device, generator=gen)
            prop = th + scale * torch.einsum("rij,rmj->rmi", L, eps)
            prop_lp = target_fn(prop)
            u = torch.log(torch.rand((R, M), device=device, generator=gen).clamp(min=1e-12))
            acc = u < (prop_lp - cur_lp)
            th = torch.where(acc.unsqueeze(-1), prop, th)
            cur_lp = torch.where(acc, prop_lp, cur_lp)
            if perm_swap_fn is not None and (step_idx == steps // 2):
                prop_s = perm_swap_fn(th)
                prop_slp = target_fn(prop_s)
                u_s = torch.log(torch.rand((R, M), device=device, generator=gen).clamp(min=1e-12))
                acc_s = u_s < (prop_slp - cur_lp)
                th = torch.where(acc_s.unsqueeze(-1), prop_s, th)
                cur_lp = torch.where(acc_s, prop_slp, cur_lp)
        return th

    q1_b, q2_b, b_rate, w_barlas = barlas_params
    prev_cov = cov0_t.unsqueeze(0).expand(R, -1, -1).clone()
    if method == "Barlas_ExpDecay":
        prev_c = torch.full((R,), q1_b + q2_b, device=device)
        prev_w = torch.full((R,), w_barlas, device=device)
    elif method in ("Fixed_LHW", "GatedC_G_LHW"):
        assert calibration is not None
        prev_c = torch.full((R,), 2.0 * sigma, device=device)
        prev_w = torch.full((R,), calibration.omega0_2, device=device)
    else:
        prev_c = torch.full((R,), 2.0 * sigma, device=device)
        prev_w = torch.ones((R,), device=device)

    prev_mu_hat = torch.empty((R, 0), device=device)
    omega_trace: list[float] = []
    c_trace: list[float] = []

    for step in range(1, T + 1):
        w_cur = torch.softmax(log_w, dim=1)
        idx_u = systematic_resample_batch(w_cur, generator=gen)
        unweighted_prev = torch.gather(particles, 1, idx_u.unsqueeze(-1).expand(-1, -1, p))

        cands = candidate_fn(unweighted_prev, step, gen)
        if method == "BOED":
            x_next = select_eig_design_gpu(unweighted_prev, mu_fn, cands, sigma, n_outer=256, m_inner=256, generator=gen)
        elif method == "Tempered_LogLoss":
            x_next = select_eig_design_gpu(unweighted_prev, mu_fn, cands, sigma, w_vec=prev_w, n_outer=256, m_inner=256, generator=gen)
        else:
            x_next = select_gibbs_eig_design_gpu(
                unweighted_prev, mu_fn, cands, sigma, prev_c, prev_w, n_outer=256, m_inner=256, generator=gen
            )

        y_next = dgp_sample_fn(theta_true, x_next, gen_dgp)
        x_hist = torch.cat([x_hist, x_next.unsqueeze(1)], dim=1)
        y_hist = torch.cat([y_hist, y_next.unsqueeze(1)], dim=1)

        if method == "BOED":
            mu_step = mu_fn(particles, x_next.unsqueeze(1)).squeeze(-1)
            d_lw = -0.5 * math.log(2.0 * math.pi * sigma * sigma) - 0.5 * ((y_next.unsqueeze(1) - mu_step) ** 2) / (sigma * sigma)
            log_w = log_w + d_lw
            log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w = torch.exp(log_w)
            ess = 1.0 / torch.sum(w * w, dim=1)
            _, cov_w = weighted_mean_and_cov(particles, w)
            cov_use = torch.where((ess < 16.0).view(R, 1, 1), 0.5 * cov_w + 0.5 * prev_cov, cov_w)
            prev_cov = cov_use.clone()

            need_resamp = ess < (0.65 * M)
            if need_resamp.any():
                idx_r = systematic_resample_batch(w, generator=gen)
                particles_r = torch.gather(particles, 1, idx_r.unsqueeze(-1).expand(-1, -1, p))
                particles_mh = mh_rejuvenate(particles_r, lambda th: log_target_boed(th, x_hist, y_hist), cov_use, steps=n_mh_steps)
                particles = torch.where(need_resamp.view(R, 1, 1), particles_mh, particles)
                log_w = torch.where(need_resamp.unsqueeze(1), torch.full_like(log_w, -math.log(float(M))), log_w)
            omega_trace.append(1.0)
            c_trace.append(float("inf"))

        elif method == "Tempered_LogLoss":
            lp_old = log_target_boed(particles, x_hist[:, :-1, :], y_hist[:, :-1], w_vec=prev_w)
            lp_pil = log_target_boed(particles, x_hist, y_hist, w_vec=prev_w)
            log_wp = log_w + (lp_pil - lp_old)
            log_wp = log_wp - torch.logsumexp(log_wp, dim=1, keepdim=True)
            wp = torch.exp(log_wp)
            ess_p = 1.0 / torch.sum(wp * wp, dim=1)
            _, cov_p = weighted_mean_and_cov(particles, wp)
            cov_p_use = torch.where((ess_p < 16.0).view(R, 1, 1), 0.5 * cov_p + 0.5 * prev_cov, cov_p)
            if (ess_p < (0.55 * M)).any():
                idx_rp = systematic_resample_batch(wp, generator=gen)
                pr_p = torch.gather(particles, 1, idx_rp.unsqueeze(-1).expand(-1, -1, p))
                pmh_p = mh_rejuvenate(pr_p, lambda th: log_target_boed(th, x_hist, y_hist, w_vec=prev_w), cov_p_use, steps=max(8, n_mh_steps // 2))
                need_p = ess_p < (0.55 * M)
                particles = torch.where(need_p.view(R, 1, 1), pmh_p, particles)
                log_wp = torch.where(need_p.unsqueeze(1), torch.full_like(log_wp, -math.log(float(M))), log_wp)
                wp = torch.exp(log_wp)

            mu_all_p = mu_fn(particles, x_hist)
            mu_hat = torch.einsum("rm,rmt->rt", wp, mu_all_p)
            var_hat = torch.einsum("rm,rmt->rt", wp, (mu_all_p - mu_hat.unsqueeze(1)) ** 2)
            s_pred_vec = torch.sum(((y_hist - mu_hat) ** 2) / (sigma * sigma + var_hat), dim=1)
            if step < p + 2:
                tau_new = torch.ones((R,), device=device)
            else:
                q95 = float(chi2.ppf(0.95, df=step))
                sp = s_pred_vec.clamp(min=1e-12)
                tau_new = torch.where(sp <= q95, torch.ones_like(sp), torch.clamp(float(step) / sp, min=0.25, max=1.0))

            lp_fin = log_target_boed(particles, x_hist, y_hist, w_vec=tau_new)
            log_w = log_wp + (lp_fin - lp_pil)
            log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w = torch.exp(log_w)
            ess = 1.0 / torch.sum(w * w, dim=1)
            _, cov_w = weighted_mean_and_cov(particles, w)
            cov_use = torch.where((ess < 16.0).view(R, 1, 1), 0.5 * cov_w + 0.5 * cov_p_use, cov_w)
            prev_cov = cov_use.clone()
            need_resamp = ess < (0.65 * M)
            if need_resamp.any():
                idx_r = systematic_resample_batch(w, generator=gen)
                particles_r = torch.gather(particles, 1, idx_r.unsqueeze(-1).expand(-1, -1, p))
                particles_mh = mh_rejuvenate(particles_r, lambda th: log_target_boed(th, x_hist, y_hist, w_vec=tau_new), cov_use, steps=n_mh_steps)
                particles = torch.where(need_resamp.view(R, 1, 1), particles_mh, particles)
                log_w = torch.where(need_resamp.unsqueeze(1), torch.full_like(log_w, -math.log(float(M))), log_w)
            prev_w = tau_new.detach()
            omega_trace.append(float(tau_new.mean().item()))
            c_trace.append(float("inf"))

        else:
            # GBOED IMQ-WSM family: GatedC_G_LHW, Fixed_LHW, Barlas_ExpDecay
            w_prev = torch.exp(log_w)
            mu_all_prev = mu_fn(particles, x_hist)
            mu_hat_pilot = torch.einsum("rm,rmt->rt", w_prev, mu_all_prev)

            if method == "Fixed_LHW":
                assert calibration is not None
                c_pilot = torch.full((R,), 2.0 * sigma, device=device)
                w_pilot = torch.full((R,), calibration.omega0_2, device=device)
            elif method == "Barlas_ExpDecay":
                c_val_b = q1_b * math.exp(-b_rate * float(step - 1)) + q2_b
                c_pilot = torch.full((R,), c_val_b, device=device)
                w_pilot = torch.full((R,), w_barlas, device=device)
            else:
                assert calibration is not None
                diff0 = particles - mu0_t.view(1, 1, p)
                delta0_prev_t = torch.einsum("rmp,pq,rmq->rm", diff0, prec0_t, diff0)
                delta0_prev = torch.einsum("rm,rm->r", w_prev, delta0_prev_t) / float(p)
                phi_prev = 1.0 + 1.0 / (1.0 + (16.0 / delta0_prev.clamp(min=1e-12)) ** 2)
                c_pilot = torch.full((R,), 2.0 * sigma, device=device)
                w_pilot = calibration.omega0_2 * phi_prev

            lp_old = log_target_gibbs(particles, x_hist[:, :-1, :], y_hist[:, :-1], prev_mu_hat, prev_c, prev_w)
            lp_pilot = log_target_gibbs(particles, x_hist, y_hist, mu_hat_pilot, c_pilot, w_pilot)
            log_w_p = log_w + (lp_pilot - lp_old)
            log_w_p = log_w_p - torch.logsumexp(log_w_p, dim=1, keepdim=True)
            wp = torch.exp(log_w_p)
            ess_p = 1.0 / torch.sum(wp * wp, dim=1)
            _, cov_p = weighted_mean_and_cov(particles, wp)
            cov_p_use = torch.where((ess_p < 16.0).view(R, 1, 1), 0.5 * cov_p + 0.5 * prev_cov, cov_p)

            need_p = ess_p < (0.55 * M)
            if need_p.any():
                idx_rp = systematic_resample_batch(wp, generator=gen)
                particles_rp = torch.gather(particles, 1, idx_rp.unsqueeze(-1).expand(-1, -1, p))
                particles_p_mh = mh_rejuvenate(
                    particles_rp,
                    lambda th: log_target_gibbs(th, x_hist, y_hist, mu_hat_pilot, c_pilot, w_pilot),
                    cov_p_use,
                    steps=max(8, n_mh_steps // 2),
                )
                particles = torch.where(need_p.view(R, 1, 1), particles_p_mh, particles)
                log_w_p = torch.where(need_p.unsqueeze(1), torch.full_like(log_w_p, -math.log(float(M))), log_w_p)
                wp = torch.exp(log_w_p)

            mu_all_p = mu_fn(particles, x_hist)
            mu_hat = torch.einsum("rm,rmt->rt", wp, mu_all_p)
            var_hat = torch.einsum("rm,rmt->rt", wp, (mu_all_p - mu_hat.unsqueeze(1)) ** 2)

            if method == "Fixed_LHW":
                c_new, w_new = c_pilot, w_pilot
                omega_trace.append(float(w_new[0].item()))
                c_trace.append(float(c_new[0].item()))
            elif method == "Barlas_ExpDecay":
                c_new, w_new = c_pilot, w_pilot
                omega_trace.append(float(w_new[0].item()))
                c_trace.append(float(c_new[0].item()))
            else:
                assert calibration is not None
                diff_now = particles - mu0_t.view(1, 1, p)
                d0_t = torch.einsum("rmp,pq,rmq->rm", diff_now, prec0_t, diff_now)
                delta0_vec = torch.einsum("rm,rm->r", wp, d0_t) / float(p)
                s_pred_vec = torch.sum(((y_hist - mu_hat) ** 2) / (sigma * sigma + var_hat), dim=1)
                d0_np = delta0_vec.detach().cpu().numpy()
                sp_np = s_pred_vec.detach().cpu().numpy()
                c_new_list, w_new_list = [], []
                for r in range(R):
                    dec = unified_omega(step, p, sigma, float(d0_np[r]), float(sp_np[r]), calibration)
                    c_new_list.append(dec.c_t)
                    w_new_list.append(dec.omega_t)
                c_new = torch.tensor(c_new_list, device=device, dtype=torch.float32)
                w_new = torch.tensor(w_new_list, device=device, dtype=torch.float32)
                omega_trace.append(float(np.mean(w_new_list)))
                c_trace.append(float(np.mean(c_new_list)))

            lp_pilot_cur = log_target_gibbs(particles, x_hist, y_hist, mu_hat_pilot, c_pilot, w_pilot)
            lp_final = log_target_gibbs(particles, x_hist, y_hist, mu_hat, c_new, w_new)
            log_w = log_w_p + (lp_final - lp_pilot_cur)
            log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w = torch.exp(log_w)
            ess = 1.0 / torch.sum(w * w, dim=1)
            _, cov_w = weighted_mean_and_cov(particles, w)
            cov_use = torch.where((ess < 16.0).view(R, 1, 1), 0.5 * cov_w + 0.5 * cov_p_use, cov_w)
            prev_cov = cov_use.clone()

            need_resamp = ess < (0.65 * M)
            if need_resamp.any():
                idx_r = systematic_resample_batch(w, generator=gen)
                particles_r = torch.gather(particles, 1, idx_r.unsqueeze(-1).expand(-1, -1, p))
                particles_mh = mh_rejuvenate(
                    particles_r,
                    lambda th: log_target_gibbs(th, x_hist, y_hist, mu_hat, c_new, w_new),
                    cov_use,
                    steps=n_mh_steps,
                )
                particles = torch.where(need_resamp.view(R, 1, 1), particles_mh, particles)
                log_w = torch.where(need_resamp.unsqueeze(1), torch.full_like(log_w, -math.log(float(M))), log_w)

            prev_mu_hat = mu_hat.detach()
            prev_c = c_new.detach()
            prev_w = w_new.detach()

    w_final = torch.softmax(log_w, dim=1)
    idx_f = systematic_resample_batch(w_final, generator=gen)
    final_particles = torch.gather(particles, 1, idx_f.unsqueeze(-1).expand(-1, -1, p))

    rmse, mmd, nll = compute_barlas_metrics_gpu(
        final_particles,
        mu_fn,
        theta_true,
        test_designs,
        sigma_model=sigma,
        sigma_true=sigma_eval,
        true_noise_dist=true_noise_dist,
        n_eval=1000,
        m_nll=256,
        generator=gen_eval,
    )
    return {
        "rmse": rmse.detach().cpu().numpy(),
        "mmd": mmd.detach().cpu().numpy(),
        "nll": nll.detach().cpu().numpy(),
        "omega_trace": omega_trace,
        "c_trace": c_trace,
    }


def summarize_benchmark_scenario(
    method_outputs: dict[str, dict],
    baseline_key: str = "BOED",
    primary_key: str = "GatedC_G_LHW",
    elapsed_sec: float = 0.0,
) -> dict:
    """Summarize RMSE/MMD/NLL means, SEs, and unpaired/paired differences vs BOED."""
    out: dict = {}
    R = len(next(iter(method_outputs.values()))["mmd"])
    for method, res in method_outputs.items():
        rmse_arr = np.asarray(res["rmse"], dtype=np.float64)
        mmd_arr = np.asarray(res["mmd"], dtype=np.float64)
        nll_arr = np.asarray(res["nll"], dtype=np.float64)
        entry = {
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_se": float(np.std(rmse_arr, ddof=1) / np.sqrt(R)),
            "mmd_mean": float(np.mean(mmd_arr)),
            "mmd_se": float(np.std(mmd_arr, ddof=1) / np.sqrt(R)),
            "nll_mean": float(np.mean(nll_arr)),
            "nll_se": float(np.std(nll_arr, ddof=1) / np.sqrt(R)),
        }
        if "omega_trace" in res:
            entry["omega_trace"] = [float(x) for x in res["omega_trace"]]
        if "c_trace" in res:
            entry["c_trace"] = [float(x) for x in res["c_trace"]]
        out[method] = entry

    b_raw = method_outputs[baseline_key]
    b_sum = out[baseline_key]
    comparisons: dict = {}
    for method, res in method_outputs.items():
        if method == baseline_key:
            continue
        m_sum = out[method]
        dmmd = np.asarray(res["mmd"], dtype=np.float64) - np.asarray(b_raw["mmd"], dtype=np.float64)
        dnll = np.asarray(res["nll"], dtype=np.float64) - np.asarray(b_raw["nll"], dtype=np.float64)
        drmse = np.asarray(res["rmse"], dtype=np.float64) - np.asarray(b_raw["rmse"], dtype=np.float64)
        mmd_use = float(np.hypot(m_sum["mmd_se"], b_sum["mmd_se"]))
        nll_use = float(np.hypot(m_sum["nll_se"], b_sum["nll_se"]))
        rmse_use = float(np.hypot(m_sum["rmse_se"], b_sum["rmse_se"]))
        comparisons[method] = {
            "delta_rmse": float(np.mean(drmse)),
            "delta_rmse_unpaired_se": rmse_use,
            "rmse_z_unpaired": float(np.mean(drmse) / rmse_use) if rmse_use > 0 else 0.0,
            "delta_mmd": float(np.mean(dmmd)),
            "delta_mmd_paired_se": float(np.std(dmmd, ddof=1) / np.sqrt(R)),
            "delta_mmd_unpaired_se": mmd_use,
            "mmd_z_unpaired": float(np.mean(dmmd) / mmd_use) if mmd_use > 0 else 0.0,
            "delta_nll": float(np.mean(dnll)),
            "delta_nll_paired_se": float(np.std(dnll, ddof=1) / np.sqrt(R)),
            "delta_nll_unpaired_se": nll_use,
            "nll_z_unpaired": float(np.mean(dnll) / nll_use) if nll_use > 0 else 0.0,
        }

    prim = dict(comparisons[primary_key])
    prim["elapsed_sec"] = float(elapsed_sec)
    out["comparison"] = prim
    out["comparisons_all"] = comparisons
    return out

