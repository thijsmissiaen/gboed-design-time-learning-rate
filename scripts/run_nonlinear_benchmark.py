"""Benchmark 3: Well-specified-prior 3D Nonlinear Radial-Sinusoidal Sequential Design (T=20, R=90, M=1024) on CUDA."""

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


def nonlin_mu_fn(theta: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """theta: (R, M, 3), xi: (R, C, 2) -> (R, M, C)."""
    th12 = theta[..., :2].unsqueeze(2)
    th3 = theta[..., 2].unsqueeze(2)
    xi_e = xi.unsqueeze(1)
    r2 = ((xi_e - th12) ** 2).sum(dim=-1)
    harmonic = torch.sin(0.8 * (xi[..., 0] + xi[..., 1])).unsqueeze(1)
    return 2.0 * torch.exp(-r2 / (2.0 * 1.44)) + th3 * harmonic


def nonlin_candidate_fn(particles: torch.Tensor, step: int, gen: torch.Generator) -> torch.Tensor:
    """Generate 320 candidate designs in [-3, 3]^2: 224 uniform + 96 posterior-guided."""
    R, M, _ = particles.shape
    device = particles.device
    c_unif = -3.0 + 6.0 * torch.rand((R, 224, 2), device=device, generator=gen)
    idx_m = torch.randint(0, M, (R, 96), device=device, generator=gen)
    th_pick = torch.gather(particles, 1, idx_m.unsqueeze(-1).expand(-1, -1, 3))[..., :2]
    c_local = (th_pick + 0.5 * torch.randn((R, 96, 2), device=device, generator=gen)).clamp(-3.0, 3.0)
    return torch.cat([c_unif, c_local], dim=1)


def run_benchmark(R: int = 90, T: int = 20, M: int = 1024, seed: int = 3000) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma = 0.4
    p = 3
    d_xi = 2

    rng_np = np.random.default_rng(0)
    calibration = design_time_lhw_imq(
        sigma_fn=lambda th, xi: np.full((th.shape[0], xi.shape[0]), sigma, dtype=np.float64),
        theta_samples=rng_np.normal(0.0, 1.0, size=(64, p)),
        design_grid=rng_np.uniform(-3.0, 3.0, size=(64, d_xi)),
        raise_on_heteroscedastic=True,
    )

    methods = ["BOED", "Fixed_LHW", "Tempered_LogLoss", "GatedC_G_LHW"]
    results = {
        "benchmark": "nonlinear_radial_sinusoidal_well_specified_prior",
        "R": R,
        "T": T,
        "M": M,
        "p": p,
        "d_xi": d_xi,
        "sigma": sigma,
        "error_dist_sigma_true": 1.0,
        "device": str(device),
        "methods": methods,
        "scenarios": {},
    }

    for scen_idx, scenario in enumerate(["none", "outlier", "error-dist"]):
        t0 = time.time()
        gen_setup = torch.Generator(device=device).manual_seed(seed + 100 * scen_idx)
        theta_true = torch.randn((R, p), device=device, generator=gen_setup)
        test_designs = -3.0 + 6.0 * torch.rand((R, 500, d_xi), device=device, generator=gen_setup)
        sigma_eval = sigma if scenario != "error-dist" else 1.0

        if scenario == "none":
            def dgp_fn(th: torch.Tensor, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
                mu = nonlin_mu_fn(th.unsqueeze(1), x.unsqueeze(1)).squeeze(1).squeeze(1)
                return mu + sigma * torch.randn(mu.shape, device=device, generator=g)
        elif scenario == "outlier":
            def dgp_fn(th: torch.Tensor, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
                mu = nonlin_mu_fn(th.unsqueeze(1), x.unsqueeze(1)).squeeze(1).squeeze(1)
                y = mu + sigma * torch.randn(mu.shape, device=device, generator=g)
                mask = (torch.rand(mu.shape, device=device, generator=g) < 0.30).float()
                out = 3.0 * sigma + 4.0 * sigma * torch.rand(mu.shape, device=device, generator=g)
                return y - mask * out
        else:
            def dgp_fn(th: torch.Tensor, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
                mu = nonlin_mu_fn(th.unsqueeze(1), x.unsqueeze(1)).squeeze(1).squeeze(1)
                return mu + 1.0 * torch.randn(mu.shape, device=device, generator=g)

        method_outputs = {}
        for method in methods:
            method_outputs[method] = run_smc_sequential_gpu(
                method=method,
                R=R,
                T=T,
                M=M,
                p=p,
                d_xi=d_xi,
                bound_lo=-3.0,
                bound_hi=3.0,
                sigma=sigma,
                mu_fn=nonlin_mu_fn,
                theta_true=theta_true,
                dgp_sample_fn=dgp_fn,
                calibration=calibration if method in ("Fixed_LHW", "GatedC_G_LHW") else None,
                candidate_fn=nonlin_candidate_fn,
                test_designs=test_designs,
                sigma_eval=sigma_eval,
                seed=seed + 100 * scen_idx + 1,
                n_mh_steps=14,
                perm_swap_fn=None,
                device=device,
                dgp_seed=seed + 100 * scen_idx + 5000,
                eval_seed=seed + 100 * scen_idx + 9000,
            )

        scen_metrics = summarize_benchmark_scenario(
            method_outputs,
            baseline_key="BOED",
            primary_key="GatedC_G_LHW",
            elapsed_sec=time.time() - t0,
        )
        results["scenarios"][scenario] = scen_metrics
        b = scen_metrics["BOED"]
        g = scen_metrics["GatedC_G_LHW"]
        comp = scen_metrics["comparison"]
        print(
            f"[Nonlin | {scenario:10s}] "
            f"BOED MMD={b['mmd_mean']:.4f}({b['mmd_se']:.4f}) NLL={b['nll_mean']:.4f}({b['nll_se']:.4f}) RMSE={b['rmse_mean']:.4f} | "
            f"GatedC MMD={g['mmd_mean']:.4f}({g['mmd_se']:.4f}) NLL={g['nll_mean']:.4f}({g['nll_se']:.4f}) RMSE={g['rmse_mean']:.4f} | "
            f"uSE(MMD={comp['mmd_z_unpaired']:+.2f}, NLL={comp['nll_z_unpaired']:+.2f}) | {comp['elapsed_sec']:.1f}s"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=90)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    res = run_benchmark(R=args.R, T=args.T, M=args.M, seed=args.seed)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
