"""Generate 300-DPI English publication figures strictly from verified JSON outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from gboed_lr.design_time import psi_B_standardized
from gboed_lr.rule import phi_conflict, tau_min_floor


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def make_fig1_forest(bench: dict, out_path: Path) -> None:
    """Forest plot of unpaired z-scores (GatedC - BOED) / SE_unpaired across all benchmarks."""
    rows = [
        ("Loc. Finding | Well-spec.", "location_finding", "none", True),
        ("Loc. Finding | Outlier", "location_finding", "outlier", False),
        ("Loc. Finding | Error-dist", "location_finding", "error-dist", False),
        ("Lin. Regr. | Prior conflict", "linear_regression", "none", True),
        ("Lin. Regr. | Outlier", "linear_regression", "outlier", False),
        ("Lin. Regr. | Laplace", "linear_regression", "laplace", False),
        ("Nonlin. Rad. | Well-spec.", "nonlinear_radial_sinusoidal", "none", True),
        ("Nonlin. Rad. | Outlier", "nonlinear_radial_sinusoidal", "outlier", False),
        ("Nonlin. Rad. | Error-dist", "nonlinear_radial_sinusoidal", "error-dist", False),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True, constrained_layout=True)
    y_pos = np.arange(len(rows))[::-1]

    for ax, key, title in [
        (ax1, "mmd_z_unpaired", r"Predictive MMD $z$-score ($\Delta / \mathrm{SE}_{\mathrm{unpaired}}$)"),
        (ax2, "nll_z_unpaired", r"Predictive NLL $z$-score ($\Delta / \mathrm{SE}_{\mathrm{unpaired}}$)"),
    ]:
        zs = [float(bench["benchmarks"][bkey]["scenarios"][scen]["comparison"][key]) for _, bkey, scen, _ in rows]
        left_lim = min(min(zs) - 2.8, -13.5)

        ax.axvspan(-1.0, 1.0, color="#d9e8f5", alpha=0.6, label=r"Parity band $[-1, +1]$")
        ax.axvspan(left_lim, -1.0, color="#e6f5ea", alpha=0.5, label=r"Robust gain ($z < -1$)")
        ax.axvline(0.0, color="#4a4a4a", linestyle="-", linewidth=0.9)
        ax.axvline(-1.0, color="#1b7837", linestyle="--", linewidth=1.1)
        ax.axvline(1.0, color="#3b6998", linestyle="--", linewidth=1.0)

        for idx, (label, bkey, scen, is_none) in enumerate(rows):
            z = zs[idx]
            color = "#2b5c8f" if is_none else "#1b7837"
            marker = "s" if is_none else "o"
            ax.plot([0.0, z], [y_pos[idx], y_pos[idx]], color=color, linewidth=1.6, alpha=0.8)
            ax.scatter([z], [y_pos[idx]], color=color, marker=marker, s=55, zorder=4)
            offset = -0.35 if z < 0 else 0.35
            ha = "right" if z < 0 else "left"
            ax.text(z + offset, y_pos[idx], f"{z:+.2f}", va="center", ha=ha, fontsize=7.8, color=color, fontweight="bold")

        ax.set_xlim(left_lim, 2.5)
        ax.set_xlabel(title)
        ax.grid(axis="x", linestyle=":", alpha=0.45)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r[0] for r in rows])
    ax1.set_title("MMD Unpaired Standardized Effect vs BOED", fontweight="bold")
    ax2.set_title("NLL Unpaired Standardized Effect vs BOED", fontweight="bold")
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=True, facecolor="white", edgecolor="#cccccc", loc="outside lower center", ncol=2)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_fig2_conflict_floor(audit: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)

    taus = np.array(audit["nll_finite_sample_curves"]["taus"])
    curves = audit["nll_finite_sample_curves"]["curves"]
    styles = [
        ("kappa1.0_d0_0", r"$\kappa=1, \Delta_0^2=0$ (well-spec.)", "#2b5c8f", "-"),
        ("kappa2.0_d0_0", r"$\kappa=2, \Delta_0^2=0$ (overdisp., no conflict)", "#1b9e77", "--"),
        ("kappa2.0_d0_16", r"$\kappa=2, \Delta_0^2=16$ (overdisp., moderate conflict)", "#d95f02", "-."),
        ("kappa2.0_d0_64", r"$\kappa=2, \Delta_0^2=64$ (overdisp., severe conflict)", "#b2182b", ":"),
    ]
    for key, label, color, ls in styles:
        ax1.plot(taus, curves[key], label=label, color=color, linestyle=ls, linewidth=2.0)
    ax1.axvline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_xlabel(r"Tempering factor $\tau$")
    ax1.set_ylabel(r"Expected NLL $\mathbb{E}[\mathrm{NLL}(\tau)]$ ($T=20$)")
    ax1.set_title("1D Conjugate Conflict-Floor Tradeoff", fontweight="bold")
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.legend(frameon=False, loc="upper right", fontsize=7.8)

    d_grid = np.logspace(-2, 3, 400)
    ax2.semilogx(d_grid, phi_conflict(d_grid), label=r"Bias relief $\phi(\hat{\Delta}_0^2) \in [1, 2]$", color="#2b5c8f", linewidth=2.2)
    ax2.semilogx(d_grid, tau_min_floor(d_grid), label=r"Floor $\tau_{\min}(\hat{\Delta}_0^2) \in [0.25, 1]$", color="#d95f02", linewidth=2.2)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)
    ax2.axhline(2.0, color="#2b5c8f", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.axhline(0.25, color="#d95f02", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel(r"Prior-data conflict index $\hat{\Delta}_0^2$")
    ax2.set_ylabel("Gate multiplier value")
    ax2.set_title("Frozen Conflict Relief and Tempering Floor", fontweight="bold")
    ax2.set_ylim(0.15, 2.15)
    ax2.grid(linestyle=":", alpha=0.5)
    ax2.legend(frameon=False, loc="center right")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_fig3_pk_and_ablations(audit: dict, bench: dict, out_path: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    pk = audit["pk_heteroscedastic_audit"]
    ys = np.array(pk["y_sweep"])
    ax1.loglog(ys, np.abs(pk["p1_loss_star"]), "o-", color="#b2182b", linewidth=1.8, label=r"$p_{\mathrm{IMQ}}=1, \theta^*$ ($\sigma^*=" + f"{pk['sigma_star']:.2f}" + "$)")
    ax1.loglog(ys, np.abs(pk["p1_loss_alt"]), "s--", color="#2b5c8f", linewidth=1.8, label=r"$p_{\mathrm{IMQ}}=1, \theta_{\mathrm{alt}}$ ($\sigma_{\mathrm{alt}}=" + f"{pk['sigma_alt']:.2f}" + "$)")
    ax1.axhline(pk["p1_analytic_limit_c2_over_sd4_star"], color="#b2182b", linestyle=":", linewidth=1.1, label=r"Limit $c^2/\sigma(\theta^*)^4$")
    ax1.axhline(pk["p1_analytic_limit_c2_over_sd4_alt"], color="#2b5c8f", linestyle=":", linewidth=1.1, label=r"Limit $c^2/\sigma(\theta_{\mathrm{alt}})^4$")
    ax1.set_xlabel(r"Outlier $y$ ($\xi = 2$h)")
    ax1.set_ylabel(r"$|\ell_B(\theta; y, \xi)|$")
    ax1.set_title("PK Heteroscedastic Tail Gap", fontweight="bold")
    ax1.grid(which="both", linestyle=":", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, loc="center right", fontsize=7.2)

    z = np.linspace(-10.0, 10.0, 500)
    ax2.plot(z, z, linestyle=":", color="gray", linewidth=1.3, label=r"Gaussian score $z$")
    ax2.plot(z, psi_B_standardized(z, 2.0, 1.0), color="#d95f02", linewidth=2.0, label=r"PredC IMQ, $c/\sigma=2$")
    ax2.plot(z, psi_B_standardized(z, 5.0, 1.0), color="#1b9e77", linewidth=2.0, label=r"PredC IMQ, $c/\sigma=5$")
    ax2.set_ylim(-3.6, 3.6)
    ax2.set_xlabel(r"Standardized residual $z$")
    ax2.set_ylabel(r"Influence $\psi_B(z)$")
    ax2.set_title("Bounded Homoscedastic Influence", fontweight="bold")
    ax2.grid(linestyle=":", alpha=0.5)
    ax2.legend(frameon=False, loc="upper left", fontsize=7.5)

    reg_scens = bench["benchmarks"]["linear_regression"]["scenarios"]
    methods_ab = [
        ("BOED", "BOED", "#5b7c99"),
        ("Tempered_LogLoss", "TempLog", "#984ea3"),
        ("Unannealed_GatedC", "Unann.", "#e41a1c"),
        ("Fixed_LHW", "FixLHW", "#ff7f00"),
        ("GatedC_G_LHW", "GatedC", "#1b7837"),
    ]
    x = np.arange(3)
    w = 0.15
    scen_keys = ["none", "outlier", "laplace"]
    scen_lbls = ["Prior conflict", "Outlier", "Laplace"]
    for idx_m, (mkey, mlabel, col) in enumerate(methods_ab):
        vals = [reg_scens[s][mkey]["nll_mean"] for s in scen_keys]
        errs = [reg_scens[s][mkey]["nll_se"] for s in scen_keys]
        offset = (idx_m - 2.0) * w
        ax3.bar(x + offset, vals, w, yerr=errs, capsize=2.0, label=mlabel, color=col, edgecolor="#222222", linewidth=0.5)

    ax3.set_yscale("log")
    ax3.set_ylim(1.0, 120.0)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scen_lbls)
    ax3.set_ylabel("Predictive NLL (log scale, lower is better)")
    ax3.set_title("Linear Regression Ablations", fontweight="bold")
    ax3.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    ax3.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="upper left", fontsize=7.0, ncol=3)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    bench = json.loads(Path("results/benchmark_results.json").read_text(encoding="utf-8"))
    audit = json.loads(Path("results/audit_results.json").read_text(encoding="utf-8"))

    make_fig1_forest(bench, fig_dir / "fig1_benchmark_forest.png")
    make_fig2_conflict_floor(audit, fig_dir / "fig2_conflict_floor_nll.png")
    make_fig3_pk_and_ablations(audit, bench, fig_dir / "fig3_pk_and_ablations.png")
    print("Generated 3 figures in paper/figures/")


if __name__ == "__main__":
    main()
