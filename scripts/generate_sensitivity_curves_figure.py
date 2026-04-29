from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SENSITIVITY_CSV = ROOT / "outputs" / "sensitivity_real_design" / "sensitivity_summary.csv"
PAPER_FIGURES_DIR = ROOT / "paper_project_review_extract" / "figures"
ASSETS_DIR = ROOT / "outputs" / "paper_q1_assets"

PARAMETER_LABELS = {
    "theta1": r"$\theta_1$",
    "theta2": r"$\theta_2$",
    "theta3": r"$\theta_3$",
    "theta4": r"$\theta_4$",
}


def plot_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    colors = {
        "theta1": "#1F77B4",
        "theta2": "#FF7F0E",
        "theta3": "#2CA02C",
        "theta4": "#D62728",
    }
    markers = {
        "theta1": "o",
        "theta2": "s",
        "theta3": "^",
        "theta4": "D",
    }

    for theta_name in ["theta1", "theta2", "theta3", "theta4"]:
        subset = df[df["theta_name"] == theta_name].sort_values("multiplier")
        ax.plot(
            subset["multiplier"],
            subset[metric] * 100.0,
            marker=markers[theta_name],
            markersize=4.8,
            linewidth=1.8,
            color=colors[theta_name],
            label=PARAMETER_LABELS[theta_name],
        )

    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"$\theta$ multiplier")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.70, 0.85, 1.00, 1.15, 1.30])
    ax.set_xlim(0.67, 1.33)
    ax.grid(alpha=0.25)


def create_sensitivity_curves_figure() -> Path:
    if not SENSITIVITY_CSV.exists():
        raise FileNotFoundError(f"Sensitivity summary not found: {SENSITIVITY_CSV}")

    df = pd.read_csv(SENSITIVITY_CSV)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.fontsize": 8.8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    plot_metric(
        axes[0],
        df,
        "mean_profile_rel_change",
        "A) Mean profile-level relative change",
        "Relative change (%)",
    )
    plot_metric(
        axes[1],
        df,
        "mean_auc_rel_change",
        "B) Mean AUC relative change",
        "Relative change (%)",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        title="Perturbed parameter",
        title_fontsize=9.0,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=2.0)

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    paper_path = PAPER_FIGURES_DIR / "figure_sensitivity_curves.png"
    assets_path = ASSETS_DIR / "figure_sensitivity_curves.png"
    fig.savefig(paper_path, dpi=300, bbox_inches="tight")
    fig.savefig(assets_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paper_path


def main() -> None:
    out = create_sensitivity_curves_figure()
    print(f"Saved sensitivity curves figure: {out}")
    print(f"Copied sensitivity curves figure: {ASSETS_DIR / 'figure_sensitivity_curves.png'}")


if __name__ == "__main__":
    main()
