"""Benchmark 1 (Barlas Appendix E.7): Location Finding (d=4, K=2, T=30, R=90, M=1024) on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
for _p in (_SRC_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _smc_gpu import run_smc_sequential_gpu, summarize_benchmark_scenario
from gboed_lr import design_time_lhw_imq


def loc_mu_fn(theta: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """theta: (R, M, 8) -> (R, M, 2, 4), xi: (R, C, 4) -> mu (R, M, C)."""
    R, M, _ = theta.shape
    th = theta.view(R, M, 2, 4)
    diff = th.unsqueeze(2) - xi.unsqueeze(1).unsqueeze(3)  # (R, M, C, 2, 4)
    dist2 = (diff * diff).sum(dim=-1)  # (R, M, C, 2)
    intensity = 0.1 + (1.0 / (1e-4 + dist2)).sum(dim=-1)  # (R, M, C)
    return torch.log(intensity)


def loc_perm_swap(theta: torch.Tensor) -> torch.Tensor:
    """Exact K=2 label-switching symmetric MH move."""
    return torch.cat([theta[..., 4:8], theta[..., 0:4]], dim=-1)


def loc_candidate_fn(particles: torch.Tensor, step: int, gen: torch.Generator) -> torch.Tensor:
    """Generate 384 candidate designs in [-4, 4]^4: 256 uniform + 128 posterior-guided."""
    R, M, _ = particles.shape
    device = particles.device
    c_unif = -4.0 + 8.0 * torch.rand((R, 256, 4), device=device, generator=gen)

    idx_m = torch.randint(0, M, (R, 128), device=device, generator=gen)
    th_pick = torch.gather(particles, 1, idx_m.unsqueeze(-1).expand(-1, -1, 8)).view(R, 128, 2, 4)
    k_pick = torch.randint(0, 2, (R, 128, 1, 1), device=device, generator=gen).expand(-1, -1, 1, 4)
    src = torch.gather(th_pick, 2, k_pick).squeeze(2)
    scales = torch.where(
        torch.arange(128, device=device) < 64,
        torch.tensor(0.2, device=device),
        torch.tensor(0.6, device=device),
    ).view(1, 128, 1)
    c_local = (src + scales * torch.randn((R, 128, 4), device=device, generator=gen)).clamp(-4.0, 4.0)
    return torch.cat([c_unif, c_local], dim=1)


def run_benchmark(R: int = 90, T: int = 30, M: int = 1024, seed: int = 1000) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma = 0.5
    p = 8
    d_xi = 4

    # Design-time certification (only prior samples, sigma_fn, design space)
    rng_np = np.random.default_rng(0)
    theta_prior_np = rng_np.normal(0.0, 1.0, size=(64, p))
    design_grid_np = rng_np.uniform(-4.0, 4.0, size=(64, d_xi))
    calibration = design_time_lhw_imq(
        sigma_fn=lambda th, xi: np.full((th.shape[0], xi.shape[0]), sigma, dtype=np.float64),
        theta_samples=theta_prior_np,
        design_grid=design_grid_np,
        raise_on_heteroscedastic=True,
    )

    methods = ["BOED", "Fixed_LHW", "Barlas_ExpDecay", "GatedC_G_LHW"]
    results = {
        "benchmark": "location_finding_d4_k2",
        "protocol": "Barlas Appendix E.7 exact spec (theta* ~ N(0, I_8) per rep; error-dist sigma_true=0.8)",
        "R": R,
        "T": T,
        "M": M,
        "d": 4,
        "K": 2,
        "sigma": sigma,
        "error_dist_sigma_true": 0.8,
        "device": str(device),
        "methods": methods,
        "scenarios": {},
    }

    for scenario_idx, scenario in enumerate(["none", "outlier", "error-dist"]):
        t_start = time.time()
        gen_setup = torch.Generator(device=device).manual_seed(seed + 100 * scenario_idx)
        # Exact Barlas Appendix E.7 spec: prior-drawn true locations per replication
        theta_true = torch.randn((R, p), device=device, generator=gen_setup)
        test_designs = -4.0 + 8.0 * torch.rand((R, 500, d_xi), device=device, generator=gen_setup)
        sigma_eval = 0.8 if scenario == "error-dist" else sigma

        if scenario == "none":
            def dgp_fn(th: torch.Tensor, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
                mu = loc_mu_fn(th.unsqueeze(1), x.unsqueeze(1)).squeeze(1).squeeze(1)
                return mu + sigma * torch.randn(mu.shape, device=device, generator=g)
        elif scenario == "outlier":
            def dgp_fn(th: torch.Tensor, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
                mu = loc_mu_fn(th.unsqueeze(1), x.unsqueeze(1)).squeeze(1).squeeze(1)
                y = mu + sigma * torch.randn(mu.shape, device=device, generator=g)
                mask = (torch.rand(mu.shape, device=device, generator=g) < 0.30).float()
                out = 3.0 * sigma + 4.0 * sigma * torch.rand(mu.shape, device=device, generator=g)
                return y - mask * out
        else:
            # Single error-dist scale sigma_true = 0.8 across all R replications (Barlas E.7)
            def dgp_fn(th: torch.Tensor, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
                mu = loc_mu_fn(th.unsqueeze(1), x.unsqueeze(1)).squeeze(1).squeeze(1)
                return mu + 0.8 * torch.randn(mu.shape, device=device, generator=g)

        method_outputs = {}
        for method in methods:
            method_outputs[method] = run_smc_sequential_gpu(
                method=method,
                R=R,
                T=T,
                M=M,
                p=p,
                d_xi=d_xi,
                bound_lo=-4.0,
                bound_hi=4.0,
                sigma=sigma,
                mu_fn=loc_mu_fn,
                theta_true=theta_true,
                dgp_sample_fn=dgp_fn,
                calibration=calibration if method in ("Fixed_LHW", "GatedC_G_LHW") else None,
                candidate_fn=loc_candidate_fn,
                test_designs=test_designs,
                sigma_eval=sigma_eval,
                seed=seed + 100 * scenario_idx + 1,
                n_mh_steps=14,
                perm_swap_fn=loc_perm_swap,
                device=device,
                dgp_seed=seed + 100 * scenario_idx + 5000,
                eval_seed=seed + 100 * scenario_idx + 9000,
                barlas_params=(1.8, 0.2, 0.06, 0.4),
            )

        scen_metrics = summarize_benchmark_scenario(
            method_outputs,
            baseline_key="BOED",
            primary_key="GatedC_G_LHW",
            elapsed_sec=time.time() - t_start,
        )
        results["scenarios"][scenario] = scen_metrics
        b = scen_metrics["BOED"]
        g = scen_metrics["GatedC_G_LHW"]
        comp = scen_metrics["comparison"]
        print(
            f"[Loc d=4 | {scenario:10s}] "
            f"BOED MMD={b['mmd_mean']:.4f}({b['mmd_se']:.4f}) NLL={b['nll_mean']:.4f}({b['nll_se']:.4f}) RMSE={b['rmse_mean']:.4f} | "
            f"GatedC MMD={g['mmd_mean']:.4f}({g['mmd_se']:.4f}) NLL={g['nll_mean']:.4f}({g['nll_se']:.4f}) RMSE={g['rmse_mean']:.4f} | "
            f"uSE(MMD={comp['mmd_z_unpaired']:+.2f}, NLL={comp['nll_z_unpaired']:+.2f}) | {comp['elapsed_sec']:.1f}s"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=90)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    res = run_benchmark(R=args.R, T=args.T, M=args.M, seed=args.seed)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
