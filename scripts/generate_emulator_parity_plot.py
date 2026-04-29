from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def ensure_local_deck_python() -> None:
    """Force the local PhysPK Python wrapper to be used instead of any stale module on sys.path."""
    local_pyc = ROOT / "deck_python_3_11.pyc"
    if not local_pyc.exists():
        return

    loader = importlib.machinery.SourcelessFileLoader("deck_python_3_11", str(local_pyc))
    spec = importlib.util.spec_from_loader("deck_python_3_11", loader)
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules["deck_python_3_11"] = module


ensure_local_deck_python()

from deck_interface import load_emulator, original_run_deck, run_deck_emulated


def load_analysis_data(path: Path, sep: str = "\t") -> pd.DataFrame:
    data = pd.read_csv(path, sep=sep)
    unnamed_cols = [col for col in data.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        data = data.drop(columns=unnamed_cols)
    if "OUTL" in data.columns:
        data = data[data["OUTL"] == 0].copy()
    return data


def load_theta(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as fh:
        estimates = json.load(fh)
    return np.asarray(estimates["theta_hat"], dtype=float)


def to_pk_scale(values: np.ndarray, obs_column: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if "LN" in obs_column.upper() or "LOG" in obs_column.upper():
        return np.exp(np.clip(values, -60.0, 60.0))
    return np.clip(values, 0.0, None)


def load_exact_predictions_from_results_csv(
    path: Path,
    theta: np.ndarray,
    data: pd.DataFrame,
    time_tolerance: float = 1e-4,
) -> dict[tuple[float, float], float]:
    """Load exact DECK predictions already exported in results.csv.

    Keys are (dose, time) and values are simulator outputs on the DECK scale
    stored in WLnCCentral.so.signal[1].
    """
    usecols = [
        "#Status",
        "TIME",
        "Th_CL[Cmp1](ml)",
        "Th_CLd[Cmp1](ml)",
        "Th_Vc",
        "Th_Vp",
        "WLnCCentral.so.signal[1]",
        "mIVDrug[Cmp1,1](mg)",
    ]
    value_col = "WLnCCentral.so.signal[1]"
    dose_col = "mIVDrug[Cmp1,1](mg)"
    param_cols = ["Th_CL[Cmp1](ml)", "Th_CLd[Cmp1](ml)", "Th_Vc", "Th_Vp"]

    measured = data[data.get("Meas", 1) != 0].copy() if "Meas" in data.columns else data.copy()
    target_times = np.asarray(sorted(measured["TIME"].astype(float).unique()), dtype=float)
    target_doses = np.asarray(sorted(data.loc[data["Dose"] > 0, "Dose"].astype(float).unique()), dtype=float)
    required = {(float(dose), float(time)) for dose in target_doses for time in target_times}
    values: dict[tuple[float, float], list[float]] = {key: [] for key in required}

    # results.csv reports the same structural parameters with CLd/Vc order.
    target_params = np.asarray([theta[0], theta[2], theta[1], theta[3]], dtype=float)

    for chunk in pd.read_csv(path, sep=";", usecols=usecols, chunksize=200_000):
        for col in ["TIME", value_col, dose_col, *param_cols]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        param_mask = np.ones(len(chunk), dtype=bool)
        for col, target in zip(param_cols, target_params):
            param_mask &= np.isclose(chunk[col].to_numpy(dtype=float), target, rtol=0.0, atol=1e-8)

        status_mask = chunk["#Status"].isin(["INTEG_END", "IS_CINT"])
        dose_mask = np.isin(chunk[dose_col].to_numpy(dtype=float), target_doses)
        time_values = chunk["TIME"].to_numpy(dtype=float)
        time_mask = np.zeros(len(chunk), dtype=bool)
        nearest_targets = np.full(len(chunk), np.nan, dtype=float)
        for target_time in target_times:
            is_target = np.abs(time_values - target_time) <= time_tolerance
            time_mask |= is_target
            nearest_targets[is_target] = target_time

        selected = chunk.loc[param_mask & status_mask & dose_mask & time_mask].copy()
        if selected.empty:
            continue

        selected["_target_time"] = nearest_targets[param_mask & status_mask & dose_mask & time_mask]
        for _, row in selected.iterrows():
            key = (float(row[dose_col]), float(row["_target_time"]))
            if key in values:
                values[key].append(float(row[value_col]))

        if all(values[key] for key in required):
            break

    missing = [key for key in required if not values[key]]
    if missing:
        missing_text = ", ".join(f"dose={dose:g}, time={time:g}" for dose, time in missing[:8])
        raise RuntimeError(f"No se encontraron predicciones exactas en results.csv para: {missing_text}")

    return {key: float(np.mean(val)) for key, val in values.items()}


def collect_pointwise_predictions(
    data: pd.DataFrame,
    theta: np.ndarray,
    emulator,
    max_len: int,
    scaler_y,
    obs_column: str,
    exact_lookup: dict[tuple[float, float], float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for subject_id in sorted(data["ID"].unique()):
        data_ind = data[data["ID"] == subject_id].copy()
        pred_emulator = run_deck_emulated(theta, data_ind, emulator, max_len, scaler_y)

        if "Meas" in data_ind.columns:
            mask = data_ind["Meas"].to_numpy() != 0
        else:
            mask = np.ones(len(data_ind), dtype=bool)

        if exact_lookup is None:
            pred_exact = original_run_deck(theta, data_ind)
            exact_masked = np.asarray(pred_exact[mask], dtype=float)
        else:
            dose = float(data_ind.loc[data_ind["Dose"] > 0, "Dose"].iloc[0])
            exact_masked = np.asarray(
                [exact_lookup[(dose, float(time))] for time in data_ind.loc[mask, "TIME"].to_numpy(dtype=float)],
                dtype=float,
            )

        exact_pk = to_pk_scale(exact_masked, obs_column)
        emulator_pk = to_pk_scale(pred_emulator[mask], obs_column)
        times = data_ind.loc[mask, "TIME"].to_numpy(dtype=float)

        for time, exact, emu in zip(times, exact_pk, emulator_pk):
            rows.append(
                {
                    "ID": int(subject_id),
                    "TIME": float(time),
                    "exact_pk": float(exact),
                    "emulator_pk": float(emu),
                    "absolute_error": float(abs(emu - exact)),
                    "relative_error": float(abs(emu - exact) / (abs(exact) + 1e-8)),
                }
            )

    return pd.DataFrame(rows)


def summarize(points: pd.DataFrame) -> dict[str, float]:
    exact = points["exact_pk"].to_numpy(dtype=float)
    emu = points["emulator_pk"].to_numpy(dtype=float)
    rel = points["relative_error"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((emu - exact) ** 2)))
    corr = float(np.corrcoef(exact, emu)[0, 1])
    return {
        "n_points": float(len(points)),
        "mean_relative_error": float(np.mean(rel)),
        "median_relative_error": float(np.median(rel)),
        "p90_relative_error": float(np.quantile(rel, 0.90)),
        "max_relative_error": float(np.max(rel)),
        "rmse": rmse,
        "correlation": corr,
    }


def plot_parity(points: pd.DataFrame, metrics: dict[str, float], output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

    exact = points["exact_pk"].to_numpy(dtype=float)
    emu = points["emulator_pk"].to_numpy(dtype=float)
    positive_min = min(exact[exact > 0].min(), emu[emu > 0].min())
    lower = float(positive_min * 0.85)
    upper = float(max(exact.max(), emu.max()) * 1.08)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = plt.get_cmap("viridis", points["ID"].nunique())

    for idx, (subject_id, group) in enumerate(points.groupby("ID")):
        ax.scatter(
            group["exact_pk"],
            group["emulator_pk"],
            s=46,
            alpha=0.82,
            color=cmap(idx),
            edgecolor="white",
            linewidth=0.55,
            label=f"ID {subject_id}",
        )

    ax.plot([lower, upper], [lower, upper], color="#222222", linewidth=1.5, linestyle="--", label="Identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22, linewidth=0.8, which="both")
    ax.set_xlabel("Exact PhysPK simulator prediction (log scale)")
    ax.set_ylabel("Residual XGBoost emulator prediction (log scale)")
    ax.set_title("Emulator versus exact simulator parity")

    text = (
        f"n = {int(metrics['n_points'])} points\n"
        f"Mean relative error = {metrics['mean_relative_error']:.3%}\n"
        f"Median relative error = {metrics['median_relative_error']:.3%}\n"
        f"90th percentile = {metrics['p90_relative_error']:.3%}\n"
        f"Maximum relative error = {metrics['max_relative_error']:.3%}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D0D7DE", "alpha": 0.92},
    )

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D0D7DE",
        fontsize=8.8,
        title="Subject",
        title_fontsize=9.2,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a parity plot comparing the residual XGBoost emulator against the exact PhysPK simulator.")
    parser.add_argument("--data_file", default="Monoclonal_Antibody.txt")
    parser.add_argument("--data_sep", default="\t")
    parser.add_argument("--estimates", default="outputs/bench_real_prev_run_80/final_estimates.json")
    parser.add_argument("--emulator_dir", default="outputs/emu_xgb_residual_local_5k_peak")
    parser.add_argument("--exact_source", choices=["results_csv", "deck"], default="results_csv")
    parser.add_argument("--results_csv", default="results.csv")
    parser.add_argument("--obs_column", default="LNCONC")
    parser.add_argument("--output_png", default="paper_project_review_extract/figures/figure_emulator_scatter.png")
    parser.add_argument("--output_csv", default="outputs/paper_q1_assets/emulator_vs_deck_pointwise.csv")
    parser.add_argument("--output_metrics", default="outputs/paper_q1_assets/emulator_vs_deck_metrics.json")
    args = parser.parse_args()

    data = load_analysis_data(ROOT / args.data_file, sep=args.data_sep)
    theta = load_theta(ROOT / args.estimates)

    emulator_model_path = ROOT / args.emulator_dir / "emulator.h5"
    emulator, max_len = load_emulator(str(emulator_model_path), len(theta))

    scaler_path = ROOT / args.emulator_dir / "emulator_scaler_y.pkl"
    scaler_y = joblib.load(scaler_path) if scaler_path.exists() else None

    exact_lookup = None
    if args.exact_source == "results_csv":
        exact_lookup = load_exact_predictions_from_results_csv(ROOT / args.results_csv, theta, data)

    points = collect_pointwise_predictions(
        data,
        theta,
        emulator,
        max_len,
        scaler_y,
        args.obs_column,
        exact_lookup=exact_lookup,
    )
    metrics = summarize(points)

    output_csv = ROOT / args.output_csv
    output_metrics = ROOT / args.output_metrics
    output_png = ROOT / args.output_png
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)

    points.to_csv(output_csv, index=False)
    with output_metrics.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    plot_parity(points, metrics, output_png)

    print(f"Saved parity plot: {output_png}")
    print(f"Saved pointwise predictions: {output_csv}")
    print(f"Saved metrics: {output_metrics}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
