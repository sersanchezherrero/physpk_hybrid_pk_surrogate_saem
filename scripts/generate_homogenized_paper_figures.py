from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "paper_project_homogenized"
FIGURES_DIR = PROJECT_DIR / "figures"
ASSETS_DIR = ROOT / "outputs" / "paper_q1_assets_homogenized"

PARAM_LABELS = {
    "theta1": r"$\theta_1$",
    "theta2": r"$\theta_2$",
    "theta3": r"$\theta_3$",
    "theta4": r"$\theta_4$",
    "sigma": r"$\sigma$",
}

PARAM_TITLES = {
    "theta1": r"$\theta_1$ (CL)",
    "theta2": r"$\theta_2$ ($V_c$)",
    "theta3": r"$\theta_3$ ($V_p$)",
    "theta4": r"$\theta_4$ (Q)",
    "sigma": r"$\sigma$",
}

PARAM_UNITS = {
    "theta1": "L/h",
    "theta2": "L",
    "theta3": "L",
    "theta4": "L/h",
    "sigma": "unitless",
}

PARAM_COLORS = {
    "theta1": "#2563A6",
    "theta2": "#E76F00",
    "theta3": "#2E8B57",
    "theta4": "#C7353A",
    "sigma": "#6B7280",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#B8C0CC",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (FIGURES_DIR, ASSETS_DIR):
        fig.savefig(directory / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def generate_sensitivity_curves() -> None:
    df = pd.read_csv(ROOT / "outputs" / "sensitivity_real_design" / "sensitivity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    metrics = [
        ("mean_profile_rel_change", "A) Mean profile-level relative change"),
        ("mean_auc_rel_change", "B) Mean AUC relative change"),
    ]
    markers = {"theta1": "o", "theta2": "s", "theta3": "^", "theta4": "D"}
    for ax, (metric, title) in zip(axes, metrics):
        for theta in ["theta1", "theta2", "theta3", "theta4"]:
            subset = df[df["theta_name"] == theta].sort_values("multiplier")
            ax.plot(
                subset["multiplier"],
                subset[metric] * 100.0,
                marker=markers[theta],
                markersize=5.2,
                linewidth=2.0,
                color=PARAM_COLORS[theta],
                label=PARAM_LABELS[theta],
            )
        ax.axvline(1.0, color="#374151", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel(r"Parameter multiplier (unitless)")
        ax.set_ylabel("Relative change (%)")
        ax.set_xticks([0.70, 0.85, 1.00, 1.15, 1.30])
        ax.set_xlim(0.67, 1.33)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        title="Perturbed parameter",
        title_fontsize=10.5,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.0)
    save_figure(fig, "figure_sensitivity_curves.png")


def generate_sensitivity_ranking() -> None:
    df = pd.read_csv(ROOT / "outputs" / "paper_q1_assets" / "table_sensitivity_ranking.csv")
    ordered = ["theta1", "theta2", "theta4", "theta3"]
    metric_specs = [
        ("max_abs_profile_rel_change", "A) Max mean profile change", "#2E8B57"),
        ("max_abs_auc_rel_change", "B) Max mean AUC change", "#6B66B3"),
        ("max_abs_cmax_rel_change", r"C) Max mean $C_{\max}$ change", "#D55E00"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=False)
    for ax, (metric, title, color) in zip(axes, metric_specs):
        values = []
        for theta in ordered:
            row = df[df["theta_name"] == theta].iloc[0]
            values.append(float(row[metric]) * 100.0)
        bars = ax.bar([PARAM_LABELS[t] for t in ordered], values, color=color, alpha=0.92, width=0.65)
        ax.set_title(title)
        ax.set_ylabel("Relative change (%)")
        ax.set_ylim(0, max(values) * 1.18 + 0.05)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025 + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
    fig.suptitle("Local sensitivity ranking under the real study design", y=1.03)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "figure_sensitivity_ranking.png")


def generate_balanced_heatmap() -> None:
    df = pd.read_csv(ROOT / "outputs" / "paper_q1_assets" / "table_balanced_replicates.csv")
    truth = df[df["replicate"] == "truth"].iloc[0]
    reps = df[df["replicate"] != "truth"].copy()
    rep_map = {"r101_bal": "Rep-1", "r202_bal": "Rep-2", "r303_bal": "Rep-3"}
    params = ["theta1", "theta2", "theta3", "theta4", "sigma"]
    values = []
    for _, row in reps.iterrows():
        values.append([(float(row[p]) - float(truth[p])) / float(truth[p]) * 100.0 for p in params])
    matrix = np.asarray(values, dtype=float)
    row_labels = [rep_map.get(name, str(name)) for name in reps["replicate"]]
    max_abs = float(np.ceil(np.nanmax(np.abs(matrix)) / 10.0) * 10.0)

    fig, ax = plt.subplots(figsize=(10.0, 3.9))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_title("Balanced noisy replicates: relative error vs generating truth")
    ax.set_xticks(np.arange(len(params)))
    ax.set_xticklabels([PARAM_LABELS[p] for p in params])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Estimated parameter")
    ax.set_ylabel("Balanced bootstrap replicate")
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.55 * max_abs else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.045)
    cbar.set_label("Relative error (%)")
    fig.tight_layout()
    save_figure(fig, "figure_balanced_replicate_heatmap.png")


def generate_gof_external() -> None:
    df = pd.read_csv(ROOT / "outputs" / "bench_saem_xgb_residual_local_40_peak_noempcov" / "IWRES.csv")
    obs = df["OBS_TRANS"].to_numpy(dtype=float)
    pred = df["PRED_TRANS"].to_numpy(dtype=float)
    iwres = df["IWRES"].to_numpy(dtype=float)
    time = df["TIME"].to_numpy(dtype=float)
    ids = df["ID"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12.7, 3.9))
    cmap = plt.get_cmap("viridis", len(np.unique(ids)))
    id_to_color = {subject_id: cmap(idx) for idx, subject_id in enumerate(sorted(np.unique(ids)))}
    colors = [id_to_color[subject_id] for subject_id in ids]

    ax = axes[0]
    ax.scatter(pred, obs, c=colors, s=42, alpha=0.86, edgecolor="white", linewidth=0.45)
    lower = float(min(obs.min(), pred.min()) - 0.15)
    upper = float(max(obs.max(), pred.max()) + 0.15)
    ax.plot([lower, upper], [lower, upper], color="#1F2937", linestyle="--", linewidth=1.4)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted log-concentration")
    ax.set_ylabel("Observed log-concentration")
    ax.set_title("A) Observed vs predicted")

    ax = axes[1]
    ax.scatter(time, iwres, c=colors, s=42, alpha=0.86, edgecolor="white", linewidth=0.45)
    ax.axhline(0.0, color="#1F2937", linestyle="--", linewidth=1.2)
    ax.axhline(2.0, color="#B42318", linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color="#B42318", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("IWRES")
    ax.set_title("B) IWRES vs time")

    ax = axes[2]
    clean = np.sort(iwres[np.isfinite(iwres)])
    n = clean.size
    theoretical = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.scatter(theoretical, clean, s=42, color="#2A7F9E", alpha=0.86, edgecolor="white", linewidth=0.45)
    lim = float(max(np.max(np.abs(theoretical)), np.max(np.abs(clean))) + 0.2)
    ax.plot([-lim, lim], [-lim, lim], color="#1F2937", linestyle="--", linewidth=1.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Theoretical N(0,1) quantiles")
    ax.set_ylabel("Ordered IWRES")
    ax.set_title("C) QQ-plot of IWRES")

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "figure_gof_external.png")


def generate_main_validation_dumbbell() -> None:
    df = pd.read_csv(ROOT / "outputs" / "paper_q1_assets" / "table_main_results.csv")
    scenarios = [
        "External validation",
        "Internal small noisy (n=12)",
        "Internal extended clean (n=30)",
    ]
    scenario_labels = {
        "External validation": "External validation",
        "Internal small noisy (n=12)": "Small noisy internal (n=12)",
        "Internal extended clean (n=30)": "Clean internal (n=30)",
    }
    params = ["theta1", "theta2", "theta3", "theta4", "sigma"]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4), sharey=True)
    axes = axes.ravel()
    y_positions = np.arange(len(scenarios))

    for ax, param in zip(axes, params):
        ref_values = []
        emu_values = []
        for scenario in scenarios:
            subset = df[df["scenario"] == scenario]
            ref = subset[subset["method"] != "SAEM + XGB residual"].iloc[0]
            emu = subset[subset["method"] == "SAEM + XGB residual"].iloc[0]
            ref_values.append(float(ref[param]))
            emu_values.append(float(emu[param]))
        for y, ref_value, emu_value in zip(y_positions, ref_values, emu_values):
            ax.plot([ref_value, emu_value], [y, y], color="#9CA3AF", linewidth=1.6, zorder=1)
        ax.scatter(ref_values, y_positions, color="#2563A6", s=36, label="Reference", zorder=2)
        ax.scatter(emu_values, y_positions, color="#D95F02", marker="s", s=36, label="Surrogate SAEM", zorder=3)
        ax.set_title(PARAM_TITLES[param])
        unit = PARAM_UNITS[param]
        ax.set_xlabel(f"Estimate ({unit})" if unit != "unitless" else "Estimate (unitless)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([scenario_labels[s] for s in scenarios])
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, title="Method")
    fig.suptitle("Main validation scenarios: reference vs surrogate-assisted SAEM", y=1.02)
    fig.tight_layout(w_pad=1.6, h_pad=2.0)
    save_figure(fig, "figure_main_validation_dumbbell.png")


def generate_emulator_scatter() -> None:
    df = pd.read_csv(ROOT / "outputs" / "paper_q1_assets" / "emulator_vs_deck_pointwise.csv")
    with (ROOT / "outputs" / "paper_q1_assets" / "emulator_vs_deck_metrics.json").open() as fh:
        metrics = json.load(fh)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    subject_ids = sorted(df["ID"].unique())
    cmap = plt.get_cmap("viridis", len(subject_ids))
    for idx, subject_id in enumerate(subject_ids):
        subset = df[df["ID"] == subject_id]
        ax.scatter(
            subset["exact_pk"],
            subset["emulator_pk"],
            s=30,
            color=cmap(idx),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.35,
            label=f"ID {int(subject_id)}",
        )
    lower = float(min(df["exact_pk"].min(), df["emulator_pk"].min()) * 0.85)
    upper = float(max(df["exact_pk"].max(), df["emulator_pk"].max()) * 1.15)
    ax.plot([lower, upper], [lower, upper], color="#1F2937", linestyle="--", linewidth=1.4, label="Identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Exact PhysPK simulator prediction (ug/L, log scale)")
    ax.set_ylabel("Residual XGBoost emulator prediction (ug/L, log scale)")
    ax.set_title("Emulator versus exact simulator parity")
    info = (
        f"n = {int(metrics['n_points'])} points\n"
        f"Mean relative error = {metrics['mean_relative_error'] * 100:.3f}%\n"
        f"Median relative error = {metrics['median_relative_error'] * 100:.3f}%\n"
        f"90th percentile = {metrics['p90_relative_error'] * 100:.3f}%\n"
        f"Maximum relative error = {metrics['max_relative_error'] * 100:.3f}%"
    )
    ax.text(
        0.04,
        0.96,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, title="Subject")
    fig.tight_layout()
    save_figure(fig, "figure_emulator_scatter.png")


def copy_allparams_overlay() -> None:
    source = ROOT / "paper_project_review_extract" / "figures" / "figure_allparams_profile_overlay.png"
    if not source.exists():
        source = ROOT / "outputs" / "paper_q1_assets" / "figure_allparams_profile_overlay.png"
    for directory in (FIGURES_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, directory / "figure_allparams_profile_overlay.png")


def main() -> None:
    configure_style()
    generate_sensitivity_curves()
    generate_sensitivity_ranking()
    generate_balanced_heatmap()
    generate_gof_external()
    generate_main_validation_dumbbell()
    generate_emulator_scatter()
    copy_allparams_overlay()
    print(f"Homogenized figures saved to: {FIGURES_DIR}")
    print(f"Copies saved to: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
