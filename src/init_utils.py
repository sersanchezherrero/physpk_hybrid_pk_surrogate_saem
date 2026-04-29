import ast
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from deck_interface import original_run_deck

logger = logging.getLogger(__name__)


def _safe_positive(value, floor=1e-6):
    return float(max(floor, value))


def _parse_array_cell(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.array([], dtype=float)
        try:
            parsed = ast.literal_eval(text)
            return np.asarray(parsed, dtype=float)
        except Exception:
            text = text.strip("[]")
            if not text:
                return np.array([], dtype=float)
            return np.fromstring(text, sep=" ", dtype=float)
    return np.asarray([value], dtype=float)


def _measurement_mask(data_ind):
    if "Meas" in data_ind.columns:
        return data_ind["Meas"].to_numpy() != 0
    return np.ones(data_ind.shape[0], dtype=bool)


def _transformed_residuals(pred, y_obs, cfg):
    pred = np.asarray(pred, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    if cfg.transform == "log":
        pred_t = np.log(np.clip(pred, 1e-12, None))
        y_t = np.log(np.clip(y_obs, 1e-12, None))
    else:
        pred_t = pred
        y_t = y_obs
    return y_t - pred_t, pred_t


def _estimate_sigma_from_residuals(residuals, preds_transformed, cfg):
    residuals = np.asarray(residuals, dtype=float)
    preds_transformed = np.asarray(preds_transformed, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    preds_transformed = preds_transformed[np.isfinite(preds_transformed)]
    if residuals.size == 0:
        return np.array([0.1], dtype=float)

    if cfg.err_model == "add":
        return np.array([max(1e-4, float(np.std(residuals, ddof=1) if residuals.size > 1 else np.std(residuals)))], dtype=float)

    if cfg.err_model == "prop":
        if cfg.transform == "identity":
            valid = np.abs(preds_transformed) > 1e-8
            weighted = residuals[valid] / np.abs(preds_transformed[valid]) if np.any(valid) else residuals
            sigma = float(np.std(weighted, ddof=1) if weighted.size > 1 else np.std(weighted))
        else:
            sigma = float(np.std(residuals, ddof=1) if residuals.size > 1 else np.std(residuals))
        return np.array([max(1e-4, sigma)], dtype=float)

    if cfg.err_model == "add+prop":
        sigma_add = float(np.std(residuals, ddof=1) if residuals.size > 1 else np.std(residuals))
        if cfg.transform == "identity":
            valid = np.abs(preds_transformed) > 1e-8
            weighted = residuals[valid] / np.abs(preds_transformed[valid]) if np.any(valid) else residuals
            sigma_prop = float(np.std(weighted, ddof=1) if weighted.size > 1 else np.std(weighted))
        else:
            sigma_prop = max(1e-4, sigma_add * 0.25)
        return np.array([max(1e-4, sigma_add), max(1e-4, sigma_prop)], dtype=float)

    return np.array([0.1], dtype=float)


def _theta_from_individual_params(phi_matrix, cfg):
    phi_matrix = np.asarray(phi_matrix, dtype=float)
    if cfg.re_type in ("log-additive", "log-normal", "log-proportional"):
        return np.exp(np.mean(np.log(np.clip(phi_matrix, 1e-8, None)), axis=0))
    return np.mean(phi_matrix, axis=0)


def _etas_from_individual_params(phi_matrix, theta, cfg):
    phi_matrix = np.asarray(phi_matrix, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if cfg.re_type in ("log-additive", "log-normal", "log-proportional"):
        return np.log(np.clip(phi_matrix, 1e-8, None) / np.clip(theta, 1e-8, None))
    if cfg.re_type == "additive":
        return phi_matrix - theta
    if cfg.re_type == "proportional":
        return phi_matrix / np.clip(theta, 1e-8, None) - 1.0
    raise ValueError(f"Unknown RE type: {cfg.re_type}")


def _regularize_omega_from_etas(eta_matrix):
    eta_matrix = np.asarray(eta_matrix, dtype=float)
    n_par = eta_matrix.shape[1]
    if eta_matrix.shape[0] <= 1:
        return np.diag(np.full(n_par, 0.05, dtype=float))

    omega_emp = np.cov(eta_matrix, rowvar=False, ddof=1)
    omega_emp = np.atleast_2d(np.asarray(omega_emp, dtype=float))
    diag = np.clip(np.diag(omega_emp), 1e-4, None)
    omega_diag = np.diag(diag)
    omega = 0.5 * omega_emp + 0.5 * omega_diag
    omega = 0.5 * (omega + omega.T)
    omega[np.diag_indices_from(omega)] = np.clip(np.diag(omega), 1e-4, None)
    omega += 1e-6 * np.eye(n_par)
    return omega


def estimate_two_stage_initial_values(data, ids, cfg, theta_seed, method="L-BFGS-B", maxiter=60):
    theta_seed = np.asarray(theta_seed, dtype=float)
    n_par = int(cfg.n_par)

    lower = np.clip(theta_seed * 0.05, 1e-3, None)
    upper = np.maximum(theta_seed * 20.0, lower * 10.0)
    log_bounds = list(zip(np.log(lower), np.log(upper)))

    phi_rows = []
    fitted_ids = []
    residuals_all = []
    preds_all = []

    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        mask = _measurement_mask(data_ind)
        if not np.any(mask):
            continue
        y_obs = data_ind.loc[mask, cfg.obs_column].to_numpy(dtype=float)

        def objective(log_phi):
            phi = np.exp(np.asarray(log_phi, dtype=float))
            pred_all = original_run_deck(phi, data_ind)
            pred = np.asarray(pred_all[mask], dtype=float)
            resid, _ = _transformed_residuals(pred, y_obs, cfg)
            if not np.all(np.isfinite(resid)):
                return 1e12
            return float(np.sum(resid ** 2))

        res = minimize(
            objective,
            x0=np.log(np.clip(theta_seed, 1e-6, None)),
            method=method,
            bounds=log_bounds if method.upper() in {"L-BFGS-B", "TNC", "SLSQP"} else None,
            options={"maxiter": int(maxiter)},
        )

        if not res.success:
            logger.warning("Two-stage no convergio limpiamente para ID %s: %s", i_id, res.message)

        phi_hat = np.exp(np.asarray(res.x, dtype=float))
        pred_all = original_run_deck(phi_hat, data_ind)
        pred = np.asarray(pred_all[mask], dtype=float)
        resid, pred_t = _transformed_residuals(pred, y_obs, cfg)
        phi_rows.append(phi_hat)
        fitted_ids.append(int(i_id))
        residuals_all.extend(resid.tolist())
        preds_all.extend(np.asarray(pred_t, dtype=float).tolist())

    if not phi_rows:
        raise RuntimeError("Two-stage no produjo estimaciones individuales validas.")

    phi_matrix = np.asarray(phi_rows, dtype=float)
    theta0 = _theta_from_individual_params(phi_matrix, cfg)
    eta_matrix = _etas_from_individual_params(phi_matrix, theta0, cfg)
    Omega0 = _regularize_omega_from_etas(eta_matrix)
    sigma0 = _estimate_sigma_from_residuals(residuals_all, preds_all, cfg)

    eta_init_dict = {}
    for idx, i_id in enumerate(fitted_ids):
        eta_init_dict[int(i_id)] = np.asarray(eta_matrix[idx], dtype=float)

    return theta0, Omega0, sigma0, eta_init_dict


def load_previous_run_initialization(run_dir, cfg, source="auto"):
    if run_dir is None:
        raise ValueError("Se requiere prev_run_dir para cargar una corrida previa.")

    final_estimates_path = os.path.join(run_dir, "final_estimates.json")
    trace_path = os.path.join(run_dir, "SAEM_trace.csv")
    ebes_path = os.path.join(run_dir, "EBEs.csv")
    config_path = os.path.join(run_dir, "config.json")

    source = str(source or "auto").lower()
    if source not in {"auto", "final", "init"}:
        raise ValueError(f"Fuente prev_run no soportada: {source}")

    theta = None
    sigma = None
    Omega = None
    eta_init_dict = {}

    if source in {"auto", "final"}:
        if os.path.exists(final_estimates_path):
            with open(final_estimates_path, "r") as f:
                payload = json.load(f)
            theta = np.asarray(payload["theta_hat"], dtype=float)
            sigma = np.asarray(payload["sigma_hat"], dtype=float)
            Omega = np.asarray(payload["Omega_hat"], dtype=float)
        elif os.path.exists(trace_path):
            trace_df = pd.read_csv(trace_path)
            if trace_df.empty:
                raise ValueError(f"SAEM_trace.csv vacio en {run_dir}")
            last = trace_df.iloc[-1]
            theta = _parse_array_cell(last["theta"])
            sigma = _parse_array_cell(last["sigma"])
        elif source == "final":
            raise FileNotFoundError(
                f"No se encontro ni final_estimates.json ni SAEM_trace.csv en {run_dir}"
            )

    if theta is None or sigma is None or (Omega is None and source == "init"):
        if not os.path.exists(config_path):
            if source == "init":
                raise FileNotFoundError(f"No se encontro config.json en {run_dir}")
        else:
            with open(config_path, "r") as f:
                payload = json.load(f)
            theta = np.asarray(payload["theta0"], dtype=float)
            sigma = np.asarray(payload["sigma0"], dtype=float)
            Omega = np.asarray(payload["Omega0"], dtype=float)

    if theta is None or sigma is None:
        raise FileNotFoundError(
            f"No se encontro una inicializacion reutilizable en {run_dir}"
        )

    if os.path.exists(ebes_path):
        ebes_df = pd.read_csv(ebes_path)
        eta_cols = [c for c in ebes_df.columns if c.startswith("ETA")]
        for _, row in ebes_df.iterrows():
            eta_init_dict[int(row["ID"])] = row[eta_cols].to_numpy(dtype=float)
        if Omega is None and eta_cols:
            eta_matrix = ebes_df[eta_cols].to_numpy(dtype=float)
            Omega = _regularize_omega_from_etas(eta_matrix)

    if Omega is None:
        Omega = np.diag(np.full(int(cfg.n_par), 0.05, dtype=float))

    return np.asarray(theta, dtype=float), np.asarray(Omega, dtype=float), np.asarray(sigma, dtype=float), eta_init_dict
