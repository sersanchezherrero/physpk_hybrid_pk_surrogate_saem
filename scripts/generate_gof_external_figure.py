from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
EXTERNAL_DIR = ROOT / "outputs" / "bench_saem_xgb_residual_local_40_peak_noempcov"
PAPER_FIGURES_DIR = ROOT / "paper_project_review_extract" / "figures"
ASSETS_DIR = ROOT / "outputs" / "paper_q1_assets"


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def create_gof_external_figure() -> Path:
    iwres_path = EXTERNAL_DIR / "IWRES.csv"
    if not iwres_path.exists():
        raise FileNotFoundError(f"IWRES file not found: {iwres_path}")

    df = pd.read_csv(iwres_path)
    obs = df["OBS_TRANS"].to_numpy(dtype=float)
    pred = df["PRED_TRANS"].to_numpy(dtype=float)
    iwres = df["IWRES"].to_numpy(dtype=float)
    time = df["TIME"].to_numpy(dtype=float)
    ids = df["ID"].to_numpy()

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
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    cmap = plt.get_cmap("viridis", len(np.unique(ids)))
    id_to_color = {subject_id: cmap(idx) for idx, subject_id in enumerate(sorted(np.unique(ids)))}
    colors = [id_to_color[subject_id] for subject_id in ids]

    # A: observed vs predicted
    ax = axes[0]
    ax.scatter(pred, obs, c=colors, s=42, alpha=0.84, edgecolor="white", linewidth=0.45)
    lower = float(min(obs.min(), pred.min()) - 0.15)
    upper = float(max(obs.max(), pred.max()) + 0.15)
    ax.plot([lower, upper], [lower, upper], color="#222222", linestyle="--", linewidth=1.4)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.set_xlabel("Predicted log-concentration")
    ax.set_ylabel("Observed log-concentration")
    ax.set_title("Observed vs predicted")
    add_panel_label(ax, "A")

    # B: IWRES vs time
    ax = axes[1]
    ax.scatter(time, iwres, c=colors, s=42, alpha=0.84, edgecolor="white", linewidth=0.45)
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.axhline(2.0, color="#B42318", linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color="#B42318", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("IWRES")
    ax.set_title("IWRES vs time")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "B")

    # C: QQ plot
    ax = axes[2]
    clean_iwres = np.sort(iwres[np.isfinite(iwres)])
    n = clean_iwres.size
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = norm.ppf(p)
    ax.scatter(theoretical, clean_iwres, s=42, color="#2A7F9E", alpha=0.84, edgecolor="white", linewidth=0.45)
    lim = float(max(np.max(np.abs(theoretical)), np.max(np.abs(clean_iwres))) + 0.2)
    ax.plot([-lim, lim], [-lim, lim], color="#222222", linestyle="--", linewidth=1.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.set_xlabel("Theoretical N(0,1) quantiles")
    ax.set_ylabel("Ordered IWRES")
    ax.set_title("QQ-plot of IWRES")
    add_panel_label(ax, "C")

    fig.tight_layout(w_pad=2.2)

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    paper_path = PAPER_FIGURES_DIR / "figure_gof_external.png"
    assets_path = ASSETS_DIR / "figure_gof_external.png"
    fig.savefig(paper_path, dpi=300, bbox_inches="tight")
    fig.savefig(assets_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return paper_path


def main() -> None:
    out = create_gof_external_figure()
    print(f"Saved GOF figure: {out}")
    print(f"Copied GOF figure: {ASSETS_DIR / 'figure_gof_external.png'}")


if __name__ == "__main__":
    main()
