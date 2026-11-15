"""Benchmark 2 (Barlas Appendix E.4): Bayesian Linear Regression (T=10, R=270) on CUDA.

Uses ONLY pure-Bayesian AnnealedPredC scale continuation from prior mean 0 (zero Theil-Sen/RANSAC),
vectorized Gibbs NMC EIG design acquisition across GBOED methods, true Laplace test draws,
and 7 methods including full ablations and Barlas exponential-decay baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
for _p in (_SRC_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _smc_gpu import (
    compute_barlas_metrics_gpu,
    select_eig_design_gpu,
    select_gibbs_eig_design_gpu,
    summarize_benchmark_scenario,
)
from gboed_lr import (
    annealed_conjugate_predc_posterior,
    compute_s_pred,
    conjugate_predc_imq_wsm,
    design_time_lhw_imq,
    phi_conflict,
    unified_omega,
)


def reg_mu_fn(theta: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """theta: (R, M, 2), xi: (R, D, 1) -> (R, M, D)."""
    x = xi.squeeze(-1)
    b0 = theta[..., 0:1]
    b1 = theta[..., 1:2]
    return b0 + b1 * x.unsqueeze(1)


def _sample_gaussian_particles_gpu(
    means_np: np.ndarray,
    covs_np: np.ndarray,
    m_particles: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    mt = torch.tensor(means_np, device=device, dtype=torch.float32)
    ct = torch.tensor(covs_np, device=device, dtype=torch.float32)
    Lt = torch.linalg.cholesky(ct + 1e-6 * torch.eye(2, device=device).unsqueeze(0))
    eps = torch.randn((mt.shape[0], m_particles, 2), device=device, generator=generator)
    return mt.unsqueeze(1) + torch.einsum("rij,rmj->rmi", Lt, eps)


def run_benchmark(R: int = 270, T: int = 10, seed: int = 2000) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_np = np.linspace(-4.0, 4.0, 100, dtype=np.float64)
    phi_grid = np.column_stack([np.ones(100), grid_np])  # (100, 2)

    configs = [
        (np.array([10.0, -7.0], dtype=np.float64), 1.2),
        (np.array([-3.0, 8.0], dtype=np.float64), 0.8),
        (np.array([9.0, 9.0], dtype=np.float64), 1.0),
    ]
    r_per_cfg = R // 3
    rem = R - 3 * r_per_cfg
    counts = [r_per_cfg + (1 if i < rem else 0) for i in range(3)]

    methods = [
        "BOED",
        "Tempered_LogLoss",
        "Fixed_LHW",
        "Unannealed_GatedC",
        "Barlas_ExpDecay_Transferred",
        "Barlas_ExpDecay_Regression",
        "GatedC_G_LHW",
    ]

    results = {
        "benchmark": "bayesian_linear_regression",
        "protocol": "Barlas Appendix E.4 (alpha=0.30 outliers, true Laplace test draws, Gibbs NMC EIG)",
        "R": R,
        "T": T,
        "prior": "N(0, I_2) [severely misspecified vs Barlas true_beta]",
        "device": str(device),
        "methods": methods,
        "scenarios": {},
    }

    for scen_idx, scenario in enumerate(["none", "outlier", "laplace"]):
        t0 = time.time()
        true_noise_dist = "laplace" if scenario == "laplace" else "normal"
        acc_raw: dict[str, dict[str, list[np.ndarray]]] = {
            m: {"rmse": [], "mmd": [], "nll": []} for m in methods
        }

        for cfg_idx, ((beta_star, sigma), r_cfg) in enumerate(zip(configs, counts)):
            if r_cfg == 0:
                continue
            cal = design_time_lhw_imq(
                sigma_fn=lambda th, xi, s=sigma: np.full((th.shape[0], xi.shape[0]), s, dtype=np.float64),
                theta_samples=np.zeros((8, 2), dtype=np.float64),
                design_grid=grid_np[:, None],
                raise_on_heteroscedastic=True,
            )

            # Pre-generate coupled per-(rep, step) DGP shocks so every method faces identical shocks
            rng_dgp = np.random.default_rng(seed + 1000 * scen_idx + 100 * cfg_idx)
            if scenario == "none":
                shocks = rng_dgp.normal(0.0, sigma, size=(r_cfg, T))
            elif scenario == "outlier":
                base_n = rng_dgp.normal(0.0, sigma, size=(r_cfg, T))
                mask = (rng_dgp.random((r_cfg, T)) < 0.30).astype(np.float64)
                out_mag = rng_dgp.uniform(3.0 * sigma, 9.0 * sigma, size=(r_cfg, T))
                shocks = base_n - mask * out_mag
            else:
                shocks = rng_dgp.laplace(0.0, sigma, size=(r_cfg, T))

            cands_gpu = torch.tensor(grid_np, device=device, dtype=torch.float32).view(1, 100, 1).expand(r_cfg, -1, -1)
            theta_true_t = torch.tensor(beta_star, device=device, dtype=torch.float32).unsqueeze(0).expand(r_cfg, -1)

            for method in methods:
                gen_m = torch.Generator(device=device).manual_seed(seed + 1000 * scen_idx + 100 * cfg_idx + 31)
                means = np.zeros((r_cfg, 2), dtype=np.float64)
                covs = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], r_cfg, axis=0)
                c_cur = np.full(r_cfg, 2.0 * sigma, dtype=np.float64)
                w_cur = np.full(r_cfg, cal.omega0_2, dtype=np.float64)
                if method == "Tempered_LogLoss":
                    w_cur = np.ones(r_cfg, dtype=np.float64)
                elif method == "Barlas_ExpDecay_Transferred":
                    c_cur = np.full(r_cfg, 2.0, dtype=np.float64)
                    w_cur = np.full(r_cfg, 0.4, dtype=np.float64)
                elif method == "Barlas_ExpDecay_Regression":
                    c_cur = np.full(r_cfg, 10.0, dtype=np.float64)
                    w_cur = np.full(r_cfg, 0.4, dtype=np.float64)

                xs_hist = np.zeros((r_cfg, T), dtype=np.float64)
                ys_hist = np.zeros((r_cfg, T), dtype=np.float64)

                for t in range(1, T + 1):
                    # 1. Batched design selection on GPU
                    if method == "BOED":
                        # Exact Gaussian predictive variance argmax == NMC EIG limit
                        quad_v = np.einsum("gp,rpq,gq->rg", phi_grid, covs, phi_grid)
                        best_idx = np.argmax(0.5 * np.log1p(quad_v / (sigma * sigma)), axis=1)
                    elif method == "Tempered_LogLoss":
                        parts = _sample_gaussian_particles_gpu(means, covs, 256, device, gen_m)
                        x_sel = select_eig_design_gpu(
                            parts,
                            reg_mu_fn,
                            cands_gpu,
                            sigma,
                            w_vec=torch.tensor(w_cur, device=device, dtype=torch.float32),
                            n_outer=128,
                            m_inner=128,
                            generator=gen_m,
                        )
                        x_sel_np = x_sel[:, 0].cpu().numpy()
                        best_idx = np.argmin(np.abs(grid_np[None, :] - x_sel_np[:, None]), axis=1)
                    else:
                        parts = _sample_gaussian_particles_gpu(means, covs, 256, device, gen_m)
                        x_sel = select_gibbs_eig_design_gpu(
                            parts,
                            reg_mu_fn,
                            cands_gpu,
                            sigma,
                            c_vec=torch.tensor(c_cur, device=device, dtype=torch.float32),
                            w_vec=torch.tensor(w_cur, device=device, dtype=torch.float32),
                            n_outer=128,
                            m_inner=128,
                            generator=gen_m,
                        )
                        x_sel_np = x_sel[:, 0].cpu().numpy()
                        best_idx = np.argmin(np.abs(grid_np[None, :] - x_sel_np[:, None]), axis=1)

                    x_step = grid_np[best_idx]
                    y_step = beta_star[0] + beta_star[1] * x_step + shocks[:, t - 1]
                    xs_hist[:, t - 1] = x_step
                    ys_hist[:, t - 1] = y_step

                    # 2. Conjugate / PredC update per replication
                    for rep in range(r_cfg):
                        xa = xs_hist[rep, :t]
                        ya = ys_hist[rep, :t]
                        phi_t = np.column_stack([np.ones(t), xa])

                        if method == "BOED":
                            pr = np.eye(2) + (phi_t.T @ phi_t) / (sigma * sigma)
                            covs[rep] = np.linalg.inv(pr)
                            means[rep] = covs[rep] @ ((phi_t.T @ ya) / (sigma * sigma))

                        elif method == "Tempered_LogLoss":
                            pr_prev = np.eye(2) + (w_cur[rep] / (sigma * sigma)) * (phi_t.T @ phi_t)
                            cov_p = np.linalg.inv(pr_prev)
                            mu_p = cov_p @ ((w_cur[rep] / (sigma * sigma)) * (phi_t.T @ ya))
                            if t < 4:
                                tau_t = 1.0
                            else:
                                s_p = compute_s_pred(ya, phi_t @ mu_p, np.sum((phi_t @ cov_p) * phi_t, axis=1), sigma)
                                tau_t = 1.0 if s_p <= float(chi2.ppf(0.95, df=t)) else max(0.25, float(t) / max(s_p, 1e-12))
                            w_cur[rep] = tau_t
                            pr = np.eye(2) + (tau_t / (sigma * sigma)) * (phi_t.T @ phi_t)
                            covs[rep] = np.linalg.inv(pr)
                            means[rep] = covs[rep] @ ((tau_t / (sigma * sigma)) * (phi_t.T @ ya))

                        elif method == "Fixed_LHW":
                            mp, cp, _ = annealed_conjugate_predc_posterior(
                                phi_t, ya, sigma=sigma, c_target=2.0 * sigma, omega_target=cal.omega0_2
                            )
                            means[rep], covs[rep] = mp, cp

                        elif method == "Unannealed_GatedC":
                            # Single-pass PredC from prior predictive mean mu_hat=0 (demonstrates Proposition 2.5 Attenuation)
                            mu0_hat = np.zeros(t, dtype=np.float64)
                            mp_pil, cp_pil = conjugate_predc_imq_wsm(phi_t, ya, mu0_hat, sigma=sigma, c=2.0 * sigma, omega=cal.omega0_2)
                            d0_sq = float((np.sum(mp_pil * mp_pil) + np.trace(cp_pil)) / 2.0)
                            s_p = compute_s_pred(ya, phi_t @ mp_pil, np.sum((phi_t @ cp_pil) * phi_t, axis=1), sigma)
                            dec = unified_omega(t=t, p=2, sigma=sigma, delta0_sq=d0_sq, s_pred=s_p, calibration=cal)
                            c_cur[rep], w_cur[rep] = dec.c_t, dec.omega_t
                            means[rep], covs[rep] = conjugate_predc_imq_wsm(phi_t, ya, mu0_hat, sigma=sigma, c=dec.c_t, omega=dec.omega_t)

                        elif method == "Barlas_ExpDecay_Transferred":
                            c_b = 1.8 * math.exp(-0.06 * float(t - 1)) + 0.2
                            w_b = 0.4
                            c_cur[rep], w_cur[rep] = c_b, w_b
                            mu_hat_prev = phi_t @ means[rep]
                            means[rep], covs[rep] = conjugate_predc_imq_wsm(phi_t, ya, mu_hat_prev, sigma=sigma, c=c_b, omega=w_b)

                        elif method == "Barlas_ExpDecay_Regression":
                            c_b = 9.0 * math.exp(-0.3 * float(t - 1)) + 1.0
                            w_b = 0.4
                            c_cur[rep], w_cur[rep] = c_b, w_b
                            mu_hat_prev = phi_t @ means[rep]
                            means[rep], covs[rep] = conjugate_predc_imq_wsm(phi_t, ya, mu_hat_prev, sigma=sigma, c=c_b, omega=w_b)

                        else:
                            # Pure-Bayesian GatedC G-LHW with AnnealedPredC scale continuation
                            mp_0, cp_0, _ = annealed_conjugate_predc_posterior(
                                phi_t, ya, sigma=sigma, c_target=2.0 * sigma, omega_target=cal.omega0_2
                            )
                            d0_init = float((np.sum(mp_0 * mp_0) + np.trace(cp_0)) / 2.0)
                            w0_pilot = cal.omega0_2 * float(phi_conflict(d0_init))
                            mp_pil, cp_pil, _ = annealed_conjugate_predc_posterior(
                                phi_t, ya, sigma=sigma, c_target=2.0 * sigma, omega_target=w0_pilot
                            )
                            d0_sq = float((np.sum(mp_pil * mp_pil) + np.trace(cp_pil)) / 2.0)
                            s_pred = compute_s_pred(ya, phi_t @ mp_pil, np.sum((phi_t @ cp_pil) * phi_t, axis=1), sigma)
                            dec = unified_omega(t=t, p=2, sigma=sigma, delta0_sq=d0_sq, s_pred=s_pred, calibration=cal)
                            c_cur[rep], w_cur[rep] = dec.c_t, dec.omega_t
                            means[rep], covs[rep], _ = annealed_conjugate_predc_posterior(
                                phi_t, ya, sigma=sigma, c_target=dec.c_t, omega_target=dec.omega_t
                            )

                # Coupled GPU predictive metric evaluation
                gen_ev = torch.Generator(device=device).manual_seed(seed + 1000 * scen_idx + 100 * cfg_idx + 7)
                post_p = _sample_gaussian_particles_gpu(means, covs, 1024, device, gen_ev)
                rmse, mmd, nll = compute_barlas_metrics_gpu(
                    post_p,
                    reg_mu_fn,
                    theta_true_t,
                    cands_gpu,
                    sigma_model=sigma,
                    sigma_true=sigma,
                    true_noise_dist=true_noise_dist,
                    n_eval=1000,
                    m_nll=256,
                    generator=gen_ev,
                )
                acc_raw[method]["rmse"].append(rmse.detach().cpu().numpy())
                acc_raw[method]["mmd"].append(mmd.detach().cpu().numpy())
                acc_raw[method]["nll"].append(nll.detach().cpu().numpy())

        method_outputs = {
            m: {k: np.concatenate(acc_raw[m][k]) for k in ("rmse", "mmd", "nll")}
            for m in methods
        }
        scen_data = summarize_benchmark_scenario(
            method_outputs,
            baseline_key="BOED",
            primary_key="GatedC_G_LHW",
            elapsed_sec=time.time() - t0,
        )
        results["scenarios"][scenario] = scen_data
        b = scen_data["BOED"]
        g = scen_data["GatedC_G_LHW"]
        comp = scen_data["comparison"]
        print(
            f"[Reg | {scenario:8s}] "
            f"BOED MMD={b['mmd_mean']:.4f}({b['mmd_se']:.4f}) NLL={b['nll_mean']:.4f}({b['nll_se']:.4f}) RMSE={b['rmse_mean']:.4f} | "
            f"GatedC MMD={g['mmd_mean']:.4f}({g['mmd_se']:.4f}) NLL={g['nll_mean']:.4f}({g['nll_se']:.4f}) RMSE={g['rmse_mean']:.4f} | "
            f"uSE(MMD={comp['mmd_z_unpaired']:+.2f}, NLL={comp['nll_z_unpaired']:+.2f}) | {comp['elapsed_sec']:.1f}s"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=270)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    res = run_benchmark(R=args.R, T=args.T, seed=args.seed)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
