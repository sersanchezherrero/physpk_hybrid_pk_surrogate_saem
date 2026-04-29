import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ModelConfig
from deck_interface import build_schedule_signature, original_run_deck
from init_utils import load_previous_run_initialization


def load_analysis_data(data_file, sep="\t"):
    data = pd.read_csv(data_file, sep=sep)
    unnamed_cols = [col for col in data.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        data = data.drop(columns=unnamed_cols)
    if "OUTL" in data.columns:
        data = data[data["OUTL"] == 0].copy()
    else:
        data = data.copy()
    return data


def measurement_mask(data_ind):
    if "Meas" in data_ind.columns:
        return data_ind["Meas"].to_numpy(dtype=int) != 0
    return np.ones(data_ind.shape[0], dtype=bool)


def safe_auc_trapz(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return 0.0
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    return float(np.trapezoid(y, t))


def compute_terminal_slope(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 3:
        return 0.0
    order = np.argsort(t)
    tt = t[order][-3:]
    yy = y[order][-3:]
    denom = float(np.sum((tt - tt.mean()) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum((tt - tt.mean()) * (yy - yy.mean())) / denom)


def compute_profile_metrics(data_ind, pred_all):
    mask = measurement_mask(data_ind)
    t = data_ind.loc[mask, "TIME"].to_numpy(dtype=float)
    y = np.asarray(pred_all, dtype=float)[mask]
    if t.size == 0:
        return {
            "auc": 0.0,
            "cmax": 0.0,
            "tmax": 0.0,
            "terminal_slope": 0.0,
        }

    idx_max = int(np.argmax(y))
    return {
        "auc": safe_auc_trapz(t, y),
        "cmax": float(np.max(y)),
        "tmax": float(t[idx_max]),
        "terminal_slope": compute_terminal_slope(t, y),
    }


def mean_abs_relative_change(base, other):
    base = np.asarray(base, dtype=float)
    other = np.asarray(other, dtype=float)
    denom = np.maximum(np.abs(base), 1e-8)
    return float(np.mean(np.abs(other - base) / denom))


def representative_ids_by_schedule(data):
    reps = []
    seen = set()
    for i_id in data["ID"].dropna().unique().tolist():
        data_ind = data[data["ID"] == i_id].copy()
        signature = build_schedule_signature(data_ind)
        if signature in seen:
            continue
        seen.add(signature)
        reps.append((int(i_id), signature))
    return reps


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if str(x).strip()]


def ensure_output_dir(exp_name=None):
    if exp_name:
        output_dir = os.path.join("outputs", exp_name)
    else:
        output_dir = os.path.join(
            "outputs",
            f"sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_summary(detail_df):
    grouped = detail_df.groupby(["theta_index", "theta_name", "multiplier", "delta_pct"], as_index=False)
    summary = grouped.agg(
        mean_profile_rel_change=("profile_rel_change", "mean"),
        max_profile_rel_change=("profile_rel_change", "max"),
        mean_auc_rel_change=("auc_rel_change", "mean"),
        mean_cmax_rel_change=("cmax_rel_change", "mean"),
        mean_tmax_abs_change=("tmax_abs_change", "mean"),
        mean_terminal_slope_abs_change=("terminal_slope_abs_change", "mean"),
    )
    return summary


def plot_summary_lines(summary_df, output_dir):
    metric_specs = [
        ("mean_profile_rel_change", "Mean profile relative change", "profile_rel_change.png"),
        ("mean_auc_rel_change", "Mean AUC relative change", "auc_rel_change.png"),
        ("mean_cmax_rel_change", "Mean Cmax relative change", "cmax_rel_change.png"),
        ("mean_tmax_abs_change", "Mean Tmax absolute change", "tmax_abs_change.png"),
        ("mean_terminal_slope_abs_change", "Mean terminal slope absolute change", "terminal_slope_abs_change.png"),
    ]

    for metric, title, filename in metric_specs:
        plt.figure(figsize=(7, 4.5))
        for theta_name, sub in summary_df.groupby("theta_name"):
            sub = sub.sort_values("multiplier")
            plt.plot(sub["multiplier"], sub[metric], marker="o", label=theta_name)
        plt.xlabel("Theta multiplier")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def plot_profile_previews(data, theta_ref, detail_df, multipliers, output_dir):
    reps = representative_ids_by_schedule(data)
    if not reps:
        return

    rep_id, _ = reps[0]
    data_ind = data[data["ID"] == rep_id].copy()
    mask = measurement_mask(data_ind)
    time = data_ind.loc[mask, "TIME"].to_numpy(dtype=float)
    baseline_pred_log = np.asarray(original_run_deck(theta_ref, data_ind), dtype=float)[mask]
    valid_mask = np.isfinite(baseline_pred_log) & (baseline_pred_log > -100.0)
    time = time[valid_mask]
    baseline_pred = np.exp(baseline_pred_log[valid_mask])
    min_mult = min(multipliers)
    max_mult = max(multipliers)

    for theta_index in range(len(theta_ref)):
        theta_name = f"theta{theta_index + 1}"
        plt.figure(figsize=(7, 4.5))
        plt.plot(time, baseline_pred, marker="o", label="baseline")
        for multiplier, label in ((min_mult, f"{min_mult:.2f}x"), (max_mult, f"{max_mult:.2f}x")):
            theta_var = np.asarray(theta_ref, dtype=float).copy()
            theta_var[theta_index] *= multiplier
            pred_log = np.asarray(original_run_deck(theta_var, data_ind), dtype=float)[mask][valid_mask]
            pred = np.exp(pred_log)
            plt.plot(time, pred, marker="o", label=label)
        plt.xlabel("Time")
        plt.ylabel("Predicted concentration")
        plt.yscale("log")
        plt.title(f"Profile sensitivity for {theta_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"profile_{theta_name}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="Monoclonal_Antibody.txt")
    parser.add_argument("--data_sep", type=str, default="\t")
    parser.add_argument("--prev_run_dir", type=str, required=True)
    parser.add_argument("--prev_run_source", type=str, choices=["auto", "final", "init"], default="final")
    parser.add_argument("--theta_multipliers", type=str, default="0.7,0.85,1.0,1.15,1.3")
    parser.add_argument("--theta_names", type=str, default="theta1,theta2,theta3,theta4")
    parser.add_argument("--use_unique_schedules", action="store_true", default=True)
    parser.add_argument("--max_ids", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    data = load_analysis_data(args.data_file, sep=args.data_sep)
    cfg = ModelConfig()
    theta_ref, _, _, _ = load_previous_run_initialization(args.prev_run_dir, cfg, source=args.prev_run_source)
    theta_ref = np.asarray(theta_ref, dtype=float)
    theta_names = [x.strip() for x in str(args.theta_names).split(",") if x.strip()]
    if len(theta_names) != theta_ref.size:
        theta_names = [f"theta{i + 1}" for i in range(theta_ref.size)]
    multipliers = parse_float_list(args.theta_multipliers)
    if 1.0 not in multipliers:
        multipliers.append(1.0)
    multipliers = sorted(set(multipliers))

    if args.use_unique_schedules:
        rep_info = representative_ids_by_schedule(data)
        ids = [i_id for i_id, _ in rep_info]
        signatures = {i_id: signature for i_id, signature in rep_info}
    else:
        ids = [int(i_id) for i_id in data["ID"].dropna().unique().tolist()]
        signatures = {i_id: build_schedule_signature(data[data["ID"] == i_id].copy()) for i_id in ids}

    if args.max_ids is not None:
        ids = ids[: max(1, int(args.max_ids))]

    output_dir = ensure_output_dir(args.exp_name)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_file": args.data_file,
                "prev_run_dir": args.prev_run_dir,
                "prev_run_source": args.prev_run_source,
                "theta_ref": theta_ref.tolist(),
                "theta_names": theta_names,
                "theta_multipliers": multipliers,
                "ids": ids,
                "use_unique_schedules": bool(args.use_unique_schedules),
            },
            handle,
            indent=2,
        )

    baseline_predictions = {}
    baseline_metrics = {}
    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        pred = np.asarray(original_run_deck(theta_ref, data_ind), dtype=float)
        baseline_predictions[i_id] = pred
        baseline_metrics[i_id] = compute_profile_metrics(data_ind, pred)

    rows = []
    for theta_index, theta_name in enumerate(theta_names):
        for multiplier in multipliers:
            theta_var = theta_ref.copy()
            theta_var[theta_index] *= multiplier
            for i_id in ids:
                data_ind = data[data["ID"] == i_id].copy()
                pred = np.asarray(original_run_deck(theta_var, data_ind), dtype=float)
                metrics = compute_profile_metrics(data_ind, pred)
                base_pred = baseline_predictions[i_id]
                base_metrics = baseline_metrics[i_id]
                rows.append(
                    {
                        "theta_index": theta_index + 1,
                        "theta_name": theta_name,
                        "multiplier": float(multiplier),
                        "delta_pct": float((multiplier - 1.0) * 100.0),
                        "id": int(i_id),
                        "schedule_signature": signatures[i_id],
                        "baseline_theta_value": float(theta_ref[theta_index]),
                        "perturbed_theta_value": float(theta_var[theta_index]),
                        "profile_rel_change": mean_abs_relative_change(base_pred[measurement_mask(data_ind)], pred[measurement_mask(data_ind)]),
                        "baseline_auc": float(base_metrics["auc"]),
                        "auc": float(metrics["auc"]),
                        "auc_rel_change": float(abs(metrics["auc"] - base_metrics["auc"]) / (abs(base_metrics["auc"]) + 1e-8)),
                        "baseline_cmax": float(base_metrics["cmax"]),
                        "cmax": float(metrics["cmax"]),
                        "cmax_rel_change": float(abs(metrics["cmax"] - base_metrics["cmax"]) / (abs(base_metrics["cmax"]) + 1e-8)),
                        "baseline_tmax": float(base_metrics["tmax"]),
                        "tmax": float(metrics["tmax"]),
                        "tmax_abs_change": float(abs(metrics["tmax"] - base_metrics["tmax"])),
                        "baseline_terminal_slope": float(base_metrics["terminal_slope"]),
                        "terminal_slope": float(metrics["terminal_slope"]),
                        "terminal_slope_abs_change": float(abs(metrics["terminal_slope"] - base_metrics["terminal_slope"])),
                    }
                )

    detail_df = pd.DataFrame(rows)
    summary_df = build_summary(detail_df)
    detail_df.to_csv(os.path.join(output_dir, "sensitivity_detail.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "sensitivity_summary.csv"), index=False)

    if not args.no_plots:
        plot_summary_lines(summary_df, output_dir)
        plot_profile_previews(data, theta_ref, detail_df, multipliers, output_dir)

    print(f"Sensitivity analysis completed. Results in {output_dir}")


if __name__ == "__main__":
    main()
