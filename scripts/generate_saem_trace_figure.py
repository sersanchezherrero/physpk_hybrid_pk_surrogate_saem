from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent

PARAMETERS = [
    ("theta1", r"$\theta_1$ (CL)", "L/h"),
    ("theta2", r"$\theta_2$ ($V_c$)", "L"),
    ("theta3", r"$\theta_3$ ($V_p$)", "L"),
    ("theta4", r"$\theta_4$ (Q)", "L/h"),
    ("sigma1", r"$\sigma$", "unitless"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#B8C0CC",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
        }
    )


def load_burn_in(config_path: Path) -> int | None:
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    burn = config.get("saem_cfg", {}).get("n_burn")
    return int(burn) if burn is not None else None


def generate_trace_figure(
    trace_path: Path,
    config_path: Path,
    output_png: Path,
    copy_png: Path | None,
) -> None:
    if not trace_path.exists():
        raise FileNotFoundError(f"SAEM trace not found: {trace_path}")
    trace = pd.read_csv(trace_path)
    burn_in = load_burn_in(config_path)

    configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True)
    axes = axes.ravel()
    panel_labels = ["A)", "B)", "C)", "D)", "E)"]

    for ax, (column, title, unit), label in zip(axes, PARAMETERS, panel_labels):
        ax.plot(
            trace["iter"],
            trace[column],
            color="#2563A6",
            linewidth=1.9,
            marker="o",
            markersize=3.8,
            label="SAEM trace",
        )
        ax.scatter(
            trace["iter"].iloc[-1],
            trace[column].iloc[-1],
            color="#D95F02",
            marker="s",
            s=40,
            zorder=3,
            label="Final estimate",
        )
        if burn_in is not None:
            ax.axvline(
                burn_in,
                color="#4B5563",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"Burn-in end ({burn_in})",
            )
        ax.set_title(f"{label} {title}")
        ax.set_xlabel("SAEM iteration")
        ax.set_ylabel(f"Estimate ({unit})" if unit != "unitless" else "Estimate (unitless)")
        ax.set_xlim(trace["iter"].min(), trace["iter"].max())

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False)
    axes[-1].text(
        0.5,
        0.28,
        "External proof-of-concept\nsurrogate-assisted SAEM",
        transform=axes[-1].transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
    )

    fig.suptitle("SAEM convergence trace for the external proof-of-concept scenario", y=1.01)
    fig.tight_layout(w_pad=1.8, h_pad=1.8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if copy_png is not None:
        copy_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_png, copy_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAEM convergence trace figure.")
    parser.add_argument(
        "--run_dir",
        default="outputs/bench_saem_xgb_residual_local_40_peak_noempcov",
        help="Run directory containing SAEM_trace.csv and config.json.",
    )
    parser.add_argument(
        "--output_png",
        default="paper_project_homogenized/figures/figure_saem_trace.png",
    )
    parser.add_argument(
        "--copy_png",
        default="outputs/paper_q1_assets_homogenized/figure_saem_trace.png",
    )
    args = parser.parse_args()

    run_dir = ROOT / args.run_dir
    output_png = ROOT / args.output_png
    copy_png = ROOT / args.copy_png if args.copy_png else None
    generate_trace_figure(run_dir / "SAEM_trace.csv", run_dir / "config.json", output_png, copy_png)
    print(f"Saved {output_png}")
    if copy_png is not None:
        print(f"Copied {copy_png}")


if __name__ == "__main__":
    main()
