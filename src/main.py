# main.py
import argparse
import logging
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Asegurar que podemos importar los módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, SAEMConfig
from deck_interface import use_emulator, load_emulator, disable_emulator
from init_utils import estimate_two_stage_initial_values, load_previous_run_initialization
from ml_utils import build_virtual_dataset, train_eta_warmstart_model, predict_eta_init_for_real_data
from bootstrap_utils import (
    apply_sparse_sampling_windows,
    generate_parametric_bootstrap_population,
    parse_sparse_windows,
    patient_bootstrap_population,
)
from saem import SAEM
from diagnostics import compute_ebes, compute_shrinkage, generate_diagnostic_plots

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'run.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_config(config_dict, filepath):
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)


def merge_eta_initializations(ids, n_par, primary=None, fallback=None):
    merged = {}
    primary = primary or {}
    fallback = fallback or {}
    for i_id in ids:
        if int(i_id) in primary:
            merged[int(i_id)] = np.asarray(primary[int(i_id)], dtype=float)
        elif int(i_id) in fallback:
            merged[int(i_id)] = np.asarray(fallback[int(i_id)], dtype=float)
        else:
            merged[int(i_id)] = np.zeros(n_par, dtype=float)
    return merged


def resolve_method_choice(explicit_method, use_nca_flag):
    if explicit_method:
        return explicit_method
    if use_nca_flag:
        return "nca"
    return "default"


def select_initialization(method, available, init_fallback=None):
    if method == "init":
        if init_fallback is None:
            raise ValueError("No se ha proporcionado una inicializacion base para el metodo 'init'.")
        return init_fallback, "init"

    if method == "hybrid":
        for key in ("prev_run", "two_stage", "nca"):
            if available.get(key) is not None:
                return available[key], key
        if init_fallback is not None:
            return init_fallback, "init"
        return available["default"], "default"

    if method not in available:
        raise ValueError(f"Metodo de inicializacion no soportado: {method}")

    chosen = available.get(method)
    if chosen is None:
        raise ValueError(f"No hay datos disponibles para el metodo de inicializacion '{method}'.")
    return chosen, method


def resolve_n_burn(n_iter, requested_n_burn, default_n_burn):
    n_iter = int(n_iter)
    n_burn = int(default_n_burn if requested_n_burn is None else requested_n_burn)
    n_burn = max(0, n_burn)

    if n_iter <= 1:
        return 0

    if n_burn >= n_iter:
        return max(1, min(n_iter - 1, n_iter // 2))

    return n_burn


def resolve_emulator_artifact_paths(output_dir, reuse_emulator_from=None, train_emulator=False):
    reuse_active = bool(reuse_emulator_from) and not bool(train_emulator)
    load_dir = reuse_emulator_from if reuse_active else output_dir
    load_path = os.path.join(load_dir, "emulator.h5")
    save_path = os.path.join(output_dir, "emulator.h5")
    return load_path, save_path, reuse_active


def resolve_output_dir(exp_name=None, resume_from_checkpoint=None):
    if exp_name:
        return os.path.join("outputs", exp_name)
    if resume_from_checkpoint:
        return os.path.normpath(resume_from_checkpoint)
    return os.path.join("outputs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


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


def count_measured_rows(data, meas_col="Meas"):
    if meas_col in data.columns:
        return int((data[meas_col].to_numpy() != 0).sum())
    return int(data.shape[0])


def resolve_bootstrap_output_path(output_dir, bootstrap_kind, bootstrap_output_file=None):
    if bootstrap_output_file:
        return bootstrap_output_file
    return os.path.join(output_dir, f"bootstrap_{bootstrap_kind}_virtual.txt")


def export_bootstrap_dataset(
    data,
    ids,
    bootstrap_kind,
    output_path,
    data_sep,
    cfg,
    theta=None,
    Omega=None,
    sigma=None,
    n_virtual=None,
    seed=123,
    include_residual_noise=True,
    balanced=False,
    sparse_windows=None,
    sparse_points_per_window=1,
    sparse_seed=None,
):
    if bootstrap_kind == "patient":
        boot_df = patient_bootstrap_population(
            data=data,
            ids=ids,
            n_virtual=n_virtual,
            seed=seed,
            balanced=balanced,
        )
    elif bootstrap_kind == "parametric":
        if theta is None or Omega is None or sigma is None:
            raise ValueError("El bootstrap parametrico requiere theta, Omega y sigma.")
        boot_df = generate_parametric_bootstrap_population(
            data=data,
            ids=ids,
            theta_pop=theta,
            Omega=Omega,
            sigma=sigma,
            cfg=cfg,
            n_virtual=(len(ids) if n_virtual is None else n_virtual),
            seed=seed,
            include_residual_noise=include_residual_noise,
            balanced=balanced,
        )
    else:
        raise ValueError(f"Tipo de bootstrap no soportado: {bootstrap_kind}")

    if sparse_windows:
        boot_df = apply_sparse_sampling_windows(
            boot_df,
            windows=sparse_windows,
            points_per_window=sparse_points_per_window,
            seed=(seed if sparse_seed is None else sparse_seed),
        )

    output_dirname = os.path.dirname(os.path.abspath(output_path))
    if output_dirname:
        os.makedirs(output_dirname, exist_ok=True)
    boot_df.to_csv(output_path, sep=data_sep, index=False)
    return output_path, boot_df


def _trace_from_partial_dataframe(trace_df, n_par, n_sigma):
    rows = []
    if trace_df is None or trace_df.empty:
        return rows
    trace_df = trace_df.sort_values("iter")
    for _, row in trace_df.iterrows():
        trace_row = {
            "iter": int(row["iter"]),
            "gamma": float(row.get("gamma", np.nan)),
            "theta": np.array(
                [row.get(f"theta{idx}", np.nan) for idx in range(1, n_par + 1)],
                dtype=float,
            ),
            "omega_diag": np.array(
                [row.get(f"omega_diag{idx}", np.nan) for idx in range(1, n_par + 1)],
                dtype=float,
            ),
            "sigma": np.array(
                [row.get(f"sigma{idx}", np.nan) for idx in range(1, n_sigma + 1)],
                dtype=float,
            ),
            "acc_rate": float(row.get("acc_rate", np.nan)),
            "step_size": float(row.get("step_size", np.nan)),
        }
        rows.append(trace_row)
    return rows


def load_legacy_saem_resume_state(checkpoint_dir, n_par, n_sigma):
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    run_log_path = os.path.join(checkpoint_dir, "run.log")
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(run_log_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No se encontraron artefactos suficientes para reconstruir el checkpoint en {checkpoint_dir}"
        )

    iter_pattern = re.compile(
        r"Iter\s+(?P<iter>\d+):\s+theta=\[(?P<theta>[^\]]+)\],\s+sigma=(?P<sigma>[-+0-9.eE]+),\s+acc=(?P<acc>[-+0-9.eE]+)"
    )
    last_match = None
    with open(run_log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = iter_pattern.search(line)
            if match is not None:
                last_match = match

    if last_match is None:
        raise FileNotFoundError(
            f"No se pudo extraer ninguna iteracion SAEM de {run_log_path}"
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        config_payload = json.load(handle)

    theta_hat = np.fromstring(last_match.group("theta"), sep=" ", dtype=float)
    sigma_hat = np.array([float(last_match.group("sigma"))], dtype=float)
    if theta_hat.shape != (n_par,):
        raise ValueError(f"No se pudo reconstruir theta con dimension {n_par} desde {run_log_path}")
    if sigma_hat.shape != (n_sigma,):
        raise ValueError(f"No se pudo reconstruir sigma con dimension {n_sigma} desde {run_log_path}")

    omega_source = None
    final_estimates_path = os.path.join(checkpoint_dir, "final_estimates.json")
    if os.path.exists(final_estimates_path):
        with open(final_estimates_path, "r", encoding="utf-8") as handle:
            final_payload = json.load(handle)
        omega_source = final_payload.get("Omega_hat")
    if omega_source is None:
        omega_source = config_payload.get("Omega0")
    Omega_hat = np.asarray(omega_source, dtype=float)
    if Omega_hat.shape != (n_par, n_par):
        raise ValueError("No se pudo reconstruir Omega con la dimension esperada desde el checkpoint legado.")

    state = {
        "checkpoint_dir": checkpoint_dir,
        "status": "legacy_bootstrap",
        "iter": int(last_match.group("iter")),
        "theta_hat": theta_hat,
        "Omega_hat": Omega_hat,
        "sigma_hat": sigma_hat,
        "trace": [{
            "iter": int(last_match.group("iter")),
            "gamma": np.nan,
            "theta": theta_hat.copy(),
            "omega_diag": np.diag(Omega_hat).copy(),
            "sigma": sigma_hat.copy(),
            "acc_rate": float(last_match.group("acc")),
            "step_size": np.nan,
        }],
        "etas": {},
    }
    logging.getLogger(__name__).warning(
        "Reanudacion aproximada: se reconstruye el estado SAEM desde run.log y config.json en %s. "
        "No es una continuacion exacta porque faltaban checkpoints parciales.",
        checkpoint_dir,
    )
    return state


def load_saem_checkpoint(checkpoint_dir, n_par, n_sigma):
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "final_estimates.partial.json")
    trace_path = os.path.join(checkpoint_dir, "SAEM_trace.partial.csv")
    etas_path = os.path.join(checkpoint_dir, "ETAs_partial.csv")

    if not os.path.exists(checkpoint_path):
        return load_legacy_saem_resume_state(checkpoint_dir, n_par, n_sigma)

    with open(checkpoint_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    theta_hat = np.asarray(payload.get("theta_hat", []), dtype=float)
    Omega_hat = np.asarray(payload.get("Omega_hat", []), dtype=float)
    sigma_hat = np.asarray(payload.get("sigma_hat", []), dtype=float)
    if theta_hat.shape != (n_par,):
        raise ValueError("El checkpoint no contiene theta_hat con la dimension esperada.")
    if Omega_hat.shape != (n_par, n_par):
        raise ValueError("El checkpoint no contiene Omega_hat con la dimension esperada.")
    if sigma_hat.shape != (n_sigma,):
        raise ValueError("El checkpoint no contiene sigma_hat con la dimension esperada.")

    trace_rows = []
    if os.path.exists(trace_path):
        trace_rows = _trace_from_partial_dataframe(pd.read_csv(trace_path), n_par, n_sigma)
    if not trace_rows:
        trace_rows = [{
            "iter": int(payload.get("iter", 0)),
            "gamma": float(payload.get("gamma", np.nan)),
            "theta": theta_hat.copy(),
            "omega_diag": np.diag(Omega_hat).copy(),
            "sigma": sigma_hat.copy(),
            "acc_rate": float(payload.get("acc_rate", np.nan)),
            "step_size": float(payload.get("step_size", np.nan)),
        }]

    etas = {}
    if os.path.exists(etas_path):
        etas_df = pd.read_csv(etas_path)
        eta_cols = [f"ETA{idx}" for idx in range(1, n_par + 1)]
        missing_eta_cols = [col for col in eta_cols if col not in etas_df.columns]
        if missing_eta_cols:
            raise ValueError(f"Faltan columnas de ETA en el checkpoint: {missing_eta_cols}")
        for _, row in etas_df.iterrows():
            etas[int(row["ID"])] = row[eta_cols].to_numpy(dtype=float)

    return {
        "checkpoint_dir": checkpoint_dir,
        "status": payload.get("status", "unknown"),
        "iter": int(payload.get("iter", 0)),
        "theta_hat": theta_hat,
        "Omega_hat": Omega_hat,
        "sigma_hat": sigma_hat,
        "trace": trace_rows,
        "etas": etas,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_nca", action="store_true", help="Usar inicialización con NCA")
    parser.add_argument("--init_method", type=str,
                        choices=["default", "nca", "two_stage", "prev_run", "hybrid"],
                        default=None,
                        help="Estrategia de inicializacion para theta/Omega/sigma")
    parser.add_argument("--prev_run_dir", type=str, default=None,
                        help="Carpeta de una corrida previa para reutilizar estimaciones")
    parser.add_argument("--prev_run_source", type=str, choices=["auto", "final", "init"], default="auto",
                        help="Fuente a reutilizar de prev_run_dir: estimaciones finales o theta0/Omega0/sigma0 de config.json")
    parser.add_argument("--emulator_center_method", type=str,
                        choices=["init", "nca", "two_stage", "prev_run", "hybrid"],
                        default="init",
                        help="Centro paramétrico usado para generar el dataset del emulador")
    parser.add_argument("--two_stage_method", type=str, choices=["L-BFGS-B", "Powell"], default="L-BFGS-B",
                        help="Metodo numerico del ajuste two-stage")
    parser.add_argument("--two_stage_maxiter", type=int, default=60,
                        help="Numero maximo de iteraciones por individuo en two-stage")
    parser.add_argument("--n_burn", type=int, default=None,
                        help="Numero de iteraciones de burn-in para SAEM")
    parser.add_argument("--saem_checkpoint_every", type=int, default=1,
                        help="Guardar checkpoints parciales de SAEM cada N iteraciones (<=0 desactiva)")
    parser.add_argument("--stop_after_iter", type=int, default=None,
                        help="Detener SAEM de forma controlada tras completar esa iteracion y conservar checkpoints parciales")
    parser.add_argument("--virtual_data", type=str, default=None,help="Ruta al archivo .npz con dataset virtual pre-generado para el emulador")
    parser.add_argument("--compute_se", action="store_true", help="Calcular errores estándar poblacionales")
    parser.add_argument("--use_emulator", action="store_true")
    parser.add_argument("--train_emulator", action="store_true")
    parser.add_argument("--reuse_emulator_from", type=str, default=None,
                        help="Carpeta desde la que cargar un emulador ya entrenado sin mezclar artefactos con la corrida actual")
    parser.add_argument("--emulator_data_mode", type=str, choices=["lhs", "local"], default="local",
                        help="Modo de generacion del dataset del emulador")
    parser.add_argument("--emulator_n_virtual", type=int, default=1000,
                        help="Numero objetivo de perfiles virtuales para el emulador en modo local")
    parser.add_argument("--emulator_n_pop", type=int, default=200,
                        help="Numero de poblaciones LHS para generar el dataset del emulador")
    parser.add_argument("--emulator_n_ind_per_pop", type=int, default=5,
                        help="Numero de individuos virtuales por poblacion LHS")
    parser.add_argument("--emulator_epochs", type=int, default=30,
                        help="Numero maximo de epocas para entrenar el emulador")
    parser.add_argument("--emulator_batch_size", type=int, default=64,
                        help="Batch size para entrenar el emulador")
    parser.add_argument("--emulator_early_stopping_patience", type=int, default=8,
                        help="Paciencia del early stopping del emulador")
    parser.add_argument("--emulator_model_type", type=str, choices=["transformer", "lstm", "xgb_pointwise", "xgb_residual_pointwise"], default="lstm",
                        help="Arquitectura del emulador")
    parser.add_argument("--emulator_d_model", type=int, default=64,
                        help="Dimension latente del transformer del emulador")
    parser.add_argument("--emulator_n_heads", type=int, default=4,
                        help="Numero de cabezas del transformer del emulador")
    parser.add_argument("--emulator_n_layers", type=int, default=2,
                        help="Numero de capas del emulador")
    parser.add_argument("--emulator_dropout", type=float, default=0.1,
                        help="Dropout del emulador")
    parser.add_argument("--xgb_peak_emphasis", type=float, default=4.0,
                        help="Peso adicional para los puntos cercanos al pico al entrenar emuladores XGBoost")
    parser.add_argument("--xgb_peak_window", type=int, default=1,
                        help="Numero de vecinos a cada lado del pico que reciben ponderacion adicional en XGBoost")
    parser.add_argument("--xgb_dose_emphasis", type=float, default=1.5,
                        help="Peso adicional para filas con dosis > 0 al entrenar emuladores XGBoost")
    parser.add_argument("--eta_warmstart_shap", action="store_true",
                        help="Calcular SHAP para el warm start de ETAs")
    parser.add_argument("--mstep_method", type=str, choices=["powell", "lbfgsb", "hybrid"], default="powell",
                        help="Metodo del M-step para theta en SAEM")
    parser.add_argument("--use_empirical_eta_cov", action="store_true",
                        help="Usar covarianza empirica adaptativa para la propuesta MCMC de eta")
    parser.add_argument("--emp_cov_update_freq", type=int, default=10,
                        help="Frecuencia de actualizacion de la covarianza empirica de eta")
    parser.add_argument("--max_emp_cov_buffer", type=int, default=2000,
                        help="Tamano maximo del buffer de ETAs para la covarianza empirica")
    parser.add_argument("--validate_emulator_etas", action="store_true",
                        help="Validar el emulador final con ETAs muestreadas frente al DECK real")
    parser.add_argument("--emulator_eta_validation_n_val", type=int, default=20,
                        help="Numero de individuos usados en la validacion avanzada del emulador")
    parser.add_argument("--emulator_eta_validation_n_etas", type=int, default=5,
                        help="Numero de ETAs muestreadas por individuo en la validacion avanzada del emulador")
    parser.add_argument("--emulator_max_mean_rel_error", type=float, default=0.10,
                        help="Umbral maximo aceptable del error relativo medio para activar el emulador en SAEM")
    parser.add_argument("--force_use_emulator", action="store_true",
                        help="Usar el emulador aunque su validacion no alcance el umbral configurado")
    parser.add_argument("--emulator_theta_trust_radius", type=float, default=0.5,
                        help="Radio fijo en escala log para theta alrededor del centro del emulador durante el M-step; <=0 desactiva")
    parser.add_argument("--emulator_only", action="store_true",
                        help="Entrenar y validar solo el emulador, sin ejecutar SAEM")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Carpeta con final_estimates.partial.json y parciales de SAEM para reanudar una corrida")
    parser.add_argument("--data_file", type=str, default="Monoclonal_Antibody.txt",
                        help="Archivo tabulado con los datos observacionales para ajuste/validacion")
    parser.add_argument("--data_sep", type=str, default="\t",
                        help="Separador del archivo de datos observacionales")
    parser.add_argument("--export_bootstrap", type=str, choices=["patient", "parametric"], default=None,
                        help="Exportar un dataset virtual bootstrap y salir sin lanzar SAEM")
    parser.add_argument("--bootstrap_n_virtual", type=int, default=None,
                        help="Numero de pacientes virtuales a exportar en bootstrap; por defecto usa el numero de IDs reales")
    parser.add_argument("--bootstrap_output_file", type=str, default=None,
                        help="Ruta del archivo tabulado de salida para el dataset bootstrap")
    parser.add_argument("--bootstrap_seed", type=int, default=123,
                        help="Semilla del bootstrap")
    parser.add_argument("--bootstrap_balanced", action="store_true",
                        help="Repartir los pacientes virtuales de forma equilibrada entre los IDs origen")
    parser.add_argument("--bootstrap_without_residual_noise", action="store_true",
                        help="En bootstrap parametrico, exportar solo la prediccion mecanistica sin anadir error residual")
    parser.add_argument("--sparse_windows", type=str, default=None,
                        help="Ventanas temporales para un dise\u00f1o sparse, por ejemplo 0-1,1-24,24-120,120-336,336-840")
    parser.add_argument("--sparse_points_per_window", type=int, default=1,
                        help="Numero de observaciones a seleccionar por ventana temporal y por individuo")
    parser.add_argument("--sparse_seed", type=int, default=123,
                        help="Semilla para la seleccion aleatoria del dise\u00f1o sparse")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--exp_name", type=str, default=None, help="Nombre del experimento (carpeta de salida)")
    args = parser.parse_args()

    # Crear carpeta de salida
    output_dir = resolve_output_dir(args.exp_name, args.resume_from_checkpoint)
    exp_name = args.exp_name if args.exp_name else os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Configurar logging
    logger = setup_logging(output_dir)
    logger.info("Inicio del experimento: %s", exp_name)

    sparse_windows = None
    if args.sparse_windows:
        sparse_windows = parse_sparse_windows(args.sparse_windows)
        logger.info(
            "Dise\u00f1o sparse solicitado: %s (%d ventana(s), %d punto(s) por ventana, seed=%d)",
            args.sparse_windows,
            len(sparse_windows),
            args.sparse_points_per_window,
            args.sparse_seed,
        )

    if args.reuse_emulator_from and args.train_emulator:
        logger.warning(
            "Se ignora reuse_emulator_from=%s porque train_emulator esta activo; se entrenara un nuevo emulador en la carpeta actual.",
            args.reuse_emulator_from,
        )

    emulator_load_path, emulator_save_path, reuse_emulator_active = resolve_emulator_artifact_paths(
        output_dir,
        reuse_emulator_from=args.reuse_emulator_from,
        train_emulator=args.train_emulator,
    )
    if reuse_emulator_active:
        logger.info(
            "Reutilizando emulador desde %s; se desactiva su reentrenamiento dentro de SAEM para preservar el artefacto origen.",
            args.reuse_emulator_from,
        )

    default_saem_cfg = SAEMConfig()
    resolved_n_burn = resolve_n_burn(args.n_iter, args.n_burn, default_saem_cfg.n_burn)
    requested_n_burn = default_saem_cfg.n_burn if args.n_burn is None else int(args.n_burn)
    if resolved_n_burn != requested_n_burn:
        logger.warning(
            "n_burn=%d no es compatible con n_iter=%d. Se ajusta automaticamente a %d para asegurar fase SA.",
            requested_n_burn,
            args.n_iter,
            resolved_n_burn,
        )

    # Cargar datos
    data = load_analysis_data(args.data_file, sep=args.data_sep)
    ids = np.sort(data["ID"].unique())
    logger.info("Datos cargados desde %s: %d individuos", args.data_file, len(ids))
      
    # Configuración
    model_cfg = ModelConfig()
    resume_state = None
    if args.resume_from_checkpoint:
        resume_state = load_saem_checkpoint(
            args.resume_from_checkpoint,
            n_par=model_cfg.n_par,
            n_sigma=model_cfg.n_sigma,
        )
        logger.info(
            "Checkpoint SAEM cargado desde %s (iter=%d, status=%s)",
            resume_state["checkpoint_dir"],
            resume_state["iter"],
            resume_state["status"],
        )
    saem_cfg = SAEMConfig(
        n_iter=args.n_iter,
        n_burn=resolved_n_burn,
        stop_after_iter=args.stop_after_iter,
        retrain_emulator_every=50,
        mstep_method=args.mstep_method,
        use_empirical_eta_cov=args.use_empirical_eta_cov,
        emp_cov_update_freq=args.emp_cov_update_freq,
        max_emp_cov_buffer=args.max_emp_cov_buffer
    )
    if reuse_emulator_active and args.use_emulator:
        saem_cfg.retrain_emulator_every = 0

    init_method = resolve_method_choice(args.init_method, args.use_nca)

    default_init = (
        np.array([2.0, 40.0, 20.0, 20.0], dtype=float),
        np.diag([0.212] * 4).astype(float),
        np.array([0.1], dtype=float),
        {}
    )
    available_initializations = {"default": default_init, "nca": None, "two_stage": None, "prev_run": None}

    needs_nca = init_method in {"nca", "hybrid"} or args.emulator_center_method in {"nca", "hybrid"}
    needs_prev_run = init_method in {"prev_run", "hybrid"} or args.emulator_center_method in {"prev_run", "hybrid"}
    needs_two_stage = init_method in {"two_stage", "hybrid"} or args.emulator_center_method in {"two_stage", "hybrid"}

    if needs_nca:
        try:
            from nca_utils import load_nca_initial_values

            nca_file = "SolNCA.txt"
            nca_init = load_nca_initial_values(nca_file, data, ids, model_cfg.__dict__)
            available_initializations["nca"] = nca_init
            logger.info("Inicialización con NCA cargada:")
            logger.info("  theta0 = %s", nca_init[0])
            logger.info("  Omega0 diag = %s", np.diag(nca_init[1]))
            logger.info("  sigma0 = %s", nca_init[2])
        except Exception as e:
            logger.warning("No se pudo cargar NCA: %s", e)
            if init_method == "nca":
                logger.warning("Se usara la inicializacion por defecto.")

    if needs_prev_run:
        try:
            prev_init = load_previous_run_initialization(
                args.prev_run_dir,
                model_cfg,
                source=args.prev_run_source,
            )
            available_initializations["prev_run"] = prev_init
            logger.info("Fuente prev_run seleccionada: %s", args.prev_run_source)
            logger.info("Inicialización cargada desde corrida previa: %s", args.prev_run_dir)
            logger.info("  theta_prev = %s", prev_init[0])
            logger.info("  Omega_prev diag = %s", np.diag(prev_init[1]))
            logger.info("  sigma_prev = %s", prev_init[2])
        except Exception as e:
            logger.warning("No se pudo cargar prev_run: %s", e)
            if init_method == "prev_run":
                logger.warning("Se usara la inicializacion por defecto.")

    if needs_two_stage:
        try:
            theta_seed = (
                available_initializations["prev_run"][0] if available_initializations["prev_run"] is not None
                else (
                    available_initializations["nca"][0] if available_initializations["nca"] is not None
                    else default_init[0]
                )
            )
            two_stage_init = estimate_two_stage_initial_values(
                data=data,
                ids=ids,
                cfg=model_cfg,
                theta_seed=theta_seed,
                method=args.two_stage_method,
                maxiter=args.two_stage_maxiter,
            )
            available_initializations["two_stage"] = two_stage_init
            logger.info("Inicialización two-stage calculada:")
            logger.info("  theta_ts = %s", two_stage_init[0])
            logger.info("  Omega_ts diag = %s", np.diag(two_stage_init[1]))
            logger.info("  sigma_ts = %s", two_stage_init[2])
        except Exception as e:
            logger.warning("No se pudo calcular two-stage: %s", e)
            if init_method == "two_stage":
                logger.warning("Se usara la inicializacion por defecto.")

    try:
        selected_init, resolved_init_method = select_initialization(init_method, available_initializations)
    except Exception as exc:
        logger.warning("No se pudo resolver init_method='%s': %s. Se usara 'default'.", init_method, exc)
        selected_init, resolved_init_method = available_initializations["default"], "default"
    theta0, Omega0, sigma0, eta_init_dict_init = selected_init
    try:
        emulator_center, resolved_emulator_center_method = select_initialization(
            args.emulator_center_method,
            available_initializations,
            init_fallback=selected_init
        )
    except Exception as exc:
        logger.warning(
            "No se pudo resolver emulator_center_method='%s': %s. Se usara 'init'.",
            args.emulator_center_method,
            exc
        )
        emulator_center, resolved_emulator_center_method = selected_init, "init"
    if resume_state is not None:
        theta0 = resume_state["theta_hat"].copy()
        Omega0 = resume_state["Omega_hat"].copy()
        sigma0 = resume_state["sigma_hat"].copy()
        eta_init_dict_init = {
            int(i_id): np.asarray(eta, dtype=float)
            for i_id, eta in resume_state["etas"].items()
        }
    emulator_theta0, emulator_Omega0, emulator_sigma0, _ = emulator_center
    logger.info(
        "Inicialización seleccionada (%s): theta0=%s, Omega0 diag=%s, sigma0=%s",
        resolved_init_method, theta0, np.diag(Omega0), sigma0
    )
    if resume_state is not None:
        logger.info(
            "Reanudando SAEM desde iter %d con estado parcial cargado del checkpoint.",
            resume_state["iter"],
        )
    logger.info(
        "Centro del emulador (%s): theta=%s, Omega diag=%s, sigma=%s",
        resolved_emulator_center_method, emulator_theta0, np.diag(emulator_Omega0), emulator_sigma0
    )

    # Guardar configuración
    config_dict = {
        "model_cfg": {
            "n_par": model_cfg.n_par,
            "re_type": model_cfg.re_type,
            "err_model": model_cfg.err_model,
            "transform": model_cfg.transform,
            "obs_column": model_cfg.obs_column,
            "n_sigma": model_cfg.n_sigma
        },
        "saem_cfg": {
            "n_iter": saem_cfg.n_iter,
            "n_burn": saem_cfg.n_burn,
            "n_mcmc": saem_cfg.n_mcmc,
            "checkpoint_every": args.saem_checkpoint_every,
            "stop_after_iter": saem_cfg.stop_after_iter,
            "step_size_eta": saem_cfg.step_size_eta,
            "adapt_step": saem_cfg.adapt_step,
            "target_acc": saem_cfg.target_acc,
            "use_neural_sstep": saem_cfg.use_neural_sstep,
            "neural_start_after": saem_cfg.neural_start_after,
            "neural_retrain_every": saem_cfg.neural_retrain_every,
            "neural_clip_eta": saem_cfg.neural_clip_eta,
            "mstep_method": saem_cfg.mstep_method,
            "use_empirical_eta_cov": saem_cfg.use_empirical_eta_cov,
            "emp_cov_update_freq": saem_cfg.emp_cov_update_freq,
            "max_emp_cov_buffer": saem_cfg.max_emp_cov_buffer
        },
        "args": vars(args),
        "resolved_init_method": resolved_init_method,
        "resolved_emulator_center_method": resolved_emulator_center_method,
        "theta0": theta0.tolist(),
        "Omega0": Omega0.tolist(),
        "sigma0": sigma0.tolist(),
        "xgb_training_cfg": {
            "peak_emphasis": args.xgb_peak_emphasis,
            "peak_window": args.xgb_peak_window,
            "dose_emphasis": args.xgb_dose_emphasis,
        },
    }
    save_config(config_dict, os.path.join(output_dir, "config.json"))
    logger.info("Configuración guardada en %s", os.path.join(output_dir, "config.json"))
    logger.info(
        "Perfil del emulador: modo=%s, n_virtual=%d, n_pop=%d, n_ind_per_pop=%d, epochs=%d, modelo=%s",
        args.emulator_data_mode,
        args.emulator_n_virtual,
        args.emulator_n_pop,
        args.emulator_n_ind_per_pop,
        args.emulator_epochs,
        args.emulator_model_type,
    )

    if args.export_bootstrap:
        bootstrap_output_path = resolve_bootstrap_output_path(
            output_dir,
            args.export_bootstrap,
            bootstrap_output_file=args.bootstrap_output_file,
        )
        bootstrap_output_path, bootstrap_df = export_bootstrap_dataset(
            data=data,
            ids=ids,
            bootstrap_kind=args.export_bootstrap,
            output_path=bootstrap_output_path,
            data_sep=args.data_sep,
            cfg=model_cfg,
            theta=theta0,
            Omega=Omega0,
            sigma=sigma0,
            n_virtual=args.bootstrap_n_virtual,
            seed=args.bootstrap_seed,
            balanced=args.bootstrap_balanced,
            include_residual_noise=not args.bootstrap_without_residual_noise,
            sparse_windows=sparse_windows,
            sparse_points_per_window=args.sparse_points_per_window,
            sparse_seed=args.sparse_seed,
        )
        logger.info(
            "Dataset bootstrap (%s) exportado en %s con %d filas y %d individuos virtuales.",
            args.export_bootstrap,
            bootstrap_output_path,
            bootstrap_df.shape[0],
            bootstrap_df["ID"].nunique() if "ID" in bootstrap_df.columns else -1,
        )
        if sparse_windows:
            logger.info(
                "Dise\u00f1o sparse aplicado al bootstrap exportado: %d observaciones activas en total.",
                count_measured_rows(bootstrap_df),
            )
        logger.info("Ejecucion finalizada tras exportar el dataset bootstrap.")
        return

    if sparse_windows:
        n_meas_before = count_measured_rows(data)
        data = apply_sparse_sampling_windows(
            data,
            windows=sparse_windows,
            points_per_window=args.sparse_points_per_window,
            seed=args.sparse_seed,
        )
        n_meas_after = count_measured_rows(data)
        sparse_data_path = os.path.join(output_dir, "sparse_input_data.txt")
        data.to_csv(sparse_data_path, sep=args.data_sep, index=False)
        logger.info(
            "Dise\u00f1o sparse aplicado al dataset de ajuste: %d -> %d observaciones activas. Dataset guardado en %s",
            n_meas_before,
            n_meas_after,
            sparse_data_path,
        )

    # --- Inicializar emulador ---
    from ml_utils import train_emulator
    emulator_path = emulator_save_path
    emulator_load_keras_path = emulator_load_path.replace('.h5', '.keras')
    emulator_load_joblib_path = emulator_load_path.replace('.h5', '.joblib')
    emulator = None
    max_len = None
    scaler_y = None
    emulator_validation = None
    emulator_active_for_fit = False
    trained_emulator = None
    trained_max_len = None
    trained_scaler_y = None

    if args.use_emulator:
        emulator_exists = (
            os.path.exists(emulator_load_path)
            or os.path.exists(emulator_load_keras_path)
            or os.path.exists(emulator_load_joblib_path)
        )

        def train_or_reload_emulator(force_retrain):
            return train_emulator(
                data, ids, emulator_theta0, emulator_Omega0, emulator_sigma0, model_cfg,
                n_virtual=args.emulator_n_virtual, model_path=emulator_path,
                force_retrain=force_retrain,
                epochs=args.emulator_epochs,
                batch_size=args.emulator_batch_size,
                early_stopping_patience=args.emulator_early_stopping_patience,
                n_pop=args.emulator_n_pop,
                n_ind_per_pop=args.emulator_n_ind_per_pop,
                virtual_data_path=args.virtual_data,
                emulator_data_mode=args.emulator_data_mode,
                emulator_model_type=args.emulator_model_type,
                emulator_d_model=args.emulator_d_model,
                emulator_n_heads=args.emulator_n_heads,
                emulator_n_layers=args.emulator_n_layers,
                emulator_dropout=args.emulator_dropout,
                xgb_peak_emphasis=args.xgb_peak_emphasis,
                xgb_peak_window=args.xgb_peak_window,
                xgb_dose_emphasis=args.xgb_dose_emphasis,
            )

        if emulator_exists and not args.train_emulator:
            emulator_file = (
                emulator_load_keras_path
                if os.path.exists(emulator_load_keras_path)
                else (emulator_load_joblib_path if os.path.exists(emulator_load_joblib_path) else emulator_load_path)
            )
            try:
                logger.info("Cargando emulador desde %s", emulator_file)
                emulator, max_len = load_emulator(emulator_load_path, model_cfg.n_par)
                scaler_path = emulator_load_path.replace('.h5', '_scaler_y.pkl')
                if os.path.exists(scaler_path):
                    scaler_y = joblib.load(scaler_path)
                    logger.info("Scaler cargado desde %s", scaler_path)
                else:
                    logger.warning("No se encontró el scaler. El emulador puede no funcionar correctamente.")
            except Exception as e:
                logger.error("Error al cargar el emulador: %s. Se entrenará uno nuevo.", e)
                emulator, max_len, scaler_y = train_or_reload_emulator(force_retrain=False)
        else:
            if args.train_emulator:
                logger.info("Reentrenando emulador por petición del usuario...")
            else:
                logger.info("No se encontró emulador, entrenando...")
            emulator, max_len, scaler_y = train_or_reload_emulator(force_retrain=args.train_emulator)
        trained_emulator = emulator
        trained_max_len = max_len
        trained_scaler_y = scaler_y
        logger.info(
            "Backend efectivo del emulador listo para validacion/SAEM: %s",
            getattr(emulator, "backend_name", args.emulator_model_type),
        )
    else:
        disable_emulator()
        saem_cfg.retrain_emulator_every = 0
        emulator_path = None
        logger.info("Modo DECK real (sin emulador).")

    # --- Validar emulador ---
    if args.use_emulator and (scaler_y is not None or hasattr(emulator, "predict_sequence")):
        from ml_utils import validate_emulator
        emulator_validation = validate_emulator(
            emulator, max_len, scaler_y, data, ids, theta0, sigma0, model_cfg, n_val=20
        )
        logger.info("Validación del emulador:")
        for k, v in emulator_validation.items():
            if isinstance(v, float):
                logger.info("  %s: %.4f", k, v)
            else:
                logger.info("  %s: %s", k, v)
    elif args.use_emulator:
        logger.warning("No se pudo validar el emulador porque falta scaler_y.")

    if args.use_emulator:
        mean_rel_error = np.nan
        if emulator_validation is not None:
            mean_rel_error = float(emulator_validation.get("mean_rel_error", np.nan))
        emulator_is_acceptable = np.isfinite(mean_rel_error) and mean_rel_error <= args.emulator_max_mean_rel_error

        if emulator_is_acceptable or args.force_use_emulator:
            use_emulator(emulator, max_len, scaler_y)
            emulator_active_for_fit = True
            if emulator_is_acceptable:
                logger.info(
                    "Modo emulador activado para SAEM (mean_rel_error=%.4f <= %.4f).",
                    mean_rel_error,
                    args.emulator_max_mean_rel_error,
                )
            else:
                logger.warning(
                    "Modo emulador activado por --force_use_emulator aunque mean_rel_error=%.4f supere %.4f.",
                    mean_rel_error,
                    args.emulator_max_mean_rel_error,
                )
        else:
            disable_emulator()
            saem_cfg.retrain_emulator_every = 0
            emulator_path = None
            logger.warning(
                "El emulador no se activarÃ¡ para SAEM porque mean_rel_error=%.4f supera %.4f. "
                "Se usarÃ¡ el DECK real para inferencia, warm-start y diagnÃ³sticos finales.",
                mean_rel_error,
                args.emulator_max_mean_rel_error,
            )

    if emulator_validation is not None:
        save_config(emulator_validation, os.path.join(output_dir, "emulator_validation.json"))

    if args.emulator_only:
        if args.use_emulator and args.validate_emulator_etas:
            if (
                trained_emulator is None
                or (
                    trained_scaler_y is None
                    and not hasattr(trained_emulator, "predict_sequence")
                )
            ):
                logger.warning(
                    "No se puede ejecutar la validacion avanzada del emulador porque faltan el emulador activo o el scaler."
                )
            else:
                from ml_utils import validate_emulator_with_etas

                logger.info("Validando emulador con ETAs muestreadas en modo emulator_only...")
                val_etas_results = validate_emulator_with_etas(
                    trained_emulator,
                    trained_max_len,
                    trained_scaler_y,
                    data,
                    ids,
                    emulator_theta0,
                    emulator_Omega0,
                    emulator_sigma0,
                    model_cfg,
                    n_etas_per_id=args.emulator_eta_validation_n_etas,
                    n_val=args.emulator_eta_validation_n_val,
                )
                for key, value in val_etas_results.items():
                    if isinstance(value, float):
                        logger.info("  %s: %.4f", key, value)
                    else:
                        logger.info("  %s: %s", key, value)
                save_config(
                    val_etas_results,
                    os.path.join(output_dir, "emulator_validation_etas.json")
                )
        logger.info("Modo emulator_only: se omite SAEM tras entrenar/validar el emulador.")
        logger.info("Experimento finalizado. Resultados en %s", output_dir)
        return

    # --- ML warm‑start poblacional (opcional) ---
    if saem_cfg.use_ml_pop_warmstart:
        logger.info("--- ML warm‑start para parámetros poblacionales ---")
        if scaler_y is None:
            logger.warning("El emulador no tiene scaler. No se puede generar dataset virtual para ML poblacional.")
        else:
            from ml_utils import (
                generate_population_virtual_dataset,
                train_population_warmstart_model,
                predict_pop_params_for_real_data
            )
            # Definir rangos para theta (ajusta según tu conocimiento)
            param_ranges_theta = [
                (1.0, 200.0),   # THETA1 (CL)
                (5.0, 300.0),   # THETA2 (V1)
                (1.0, 200.0),   # THETA3 (V2)
                (5.0, 300.0)    # THETA4 (Q)
            ]
            
            if saem_cfg.ml_pop_train or not os.path.exists(saem_cfg.ml_pop_path):
                logger.info("Generando dataset virtual poblacional...")
                X_pop, Y_pop = generate_population_virtual_dataset(
                    data=data,
                    ids=ids,
                    cfg=model_cfg,
                    n_pop_sets=saem_cfg.n_virtual_pop_sets,
                    n_ind_per_set=saem_cfg.n_ind_per_pop_set,
                    param_ranges_theta=param_ranges_theta,
                    omega_scale_range=(0.01, 0.5),
                    sigma_range=(0.01, 0.5),
                    seed=123,
                    use_emulator=True,
                    emulator=emulator,
                    max_len=max_len,
                    scaler_y=scaler_y
                )
                logger.info(f"Dataset generado: X={X_pop.shape}, Y={Y_pop.shape}")
                pop_models = train_population_warmstart_model(X_pop, Y_pop, seed=123)
                joblib.dump(pop_models, saem_cfg.ml_pop_path)
                logger.info(f"Modelo guardado en {saem_cfg.ml_pop_path}")
            else:
                pop_models = joblib.load(saem_cfg.ml_pop_path)
                logger.info(f"Modelo cargado desde {saem_cfg.ml_pop_path}")
            
            # Predecir nuevos valores iniciales
            theta0_ml, Omega0_ml, sigma0_log_ml = predict_pop_params_for_real_data(
                pop_models, data, ids, model_cfg
            )
            logger.info("Predicción ML para parámetros poblacionales:")
            logger.info(f"  theta0 = {theta0_ml}")
            logger.info(f"  Omega0 diag = {np.diag(Omega0_ml)}")
            logger.info(f"  sigma0 = {np.exp(sigma0_log_ml)}")
            
            # Sobrescribir valores iniciales
            theta0 = theta0_ml
            Omega0 = Omega0_ml
            sigma0 = np.exp(sigma0_log_ml)
            
            # Actualizar configuración guardada (opcional)
            config_dict["theta0"] = theta0.tolist()
            config_dict["Omega0"] = Omega0.tolist()
            config_dict["sigma0"] = sigma0.tolist()
            # Volver a guardar la configuración actualizada
            save_config(config_dict, os.path.join(output_dir, "config.json"))

    # --- ML warm‑start de ETAs ---
    if resume_state is not None:
        eta_init_dict_ml = {}
        eta_init_dict = merge_eta_initializations(
            ids, model_cfg.n_par, primary=resume_state["etas"], fallback=eta_init_dict_init
        )
        n_eta_resume = sum(int(i_id) in resume_state["etas"] for i_id in ids)
        logger.info(
            "ETAs reanudadas para %d individuos desde checkpoint (%d checkpoint, %d respaldo metodo base)",
            len(eta_init_dict), n_eta_resume, len(eta_init_dict) - n_eta_resume
        )
    else:
        eta_models_path = os.path.join(output_dir, "eta_models.pkl")
        if os.path.exists(eta_models_path):
            logger.info("Cargando modelos de ETAs desde %s", eta_models_path)
            eta_models = joblib.load(eta_models_path)
        else:
            logger.info("Generando dataset virtual para warm start de ETAs...")
            Xv, Yv = build_virtual_dataset(data, ids, theta0, Omega0, sigma0, model_cfg, n_per_id=50)
            logger.info("Dataset generado: X=%s, Y=%s", Xv.shape, Yv.shape)
            eta_models = train_eta_warmstart_model(Xv, Yv, compute_shap=args.eta_warmstart_shap)
            joblib.dump(eta_models, eta_models_path)
            logger.info(f"Modelos de ETAs guardados en {eta_models_path}")
        eta_init_dict_ml = predict_eta_init_for_real_data(eta_models, data, ids, model_cfg)
        eta_init_dict = merge_eta_initializations(
            ids, model_cfg.n_par, primary=eta_init_dict_ml, fallback=eta_init_dict_init
        )
        n_eta_fallback = sum(
            int(i_id) not in eta_init_dict_ml and int(i_id) in eta_init_dict_init for i_id in ids
        )
        logger.info(
            "ETAs iniciales preparadas para %d individuos (%d ML, %d respaldo metodo base)",
            len(eta_init_dict), len(eta_init_dict_ml), n_eta_fallback
        )

    # --- Crear instancia de SAEM y ejecutar ---
    saem = SAEM(
        data, ids, model_cfg, saem_cfg, theta0, Omega0, sigma0,
        retrain_emulator_every=saem_cfg.retrain_emulator_every,
        emulator_path=(
            None if reuse_emulator_active else emulator_path
        ) if emulator_active_for_fit else None,
        n_virtual_retrain=saem_cfg.n_virtual_retrain,
        emulator_train_kwargs={
            "emulator_data_mode": args.emulator_data_mode,
            "emulator_model_type": args.emulator_model_type,
            "epochs": args.emulator_epochs,
            "batch_size": args.emulator_batch_size,
            "early_stopping_patience": args.emulator_early_stopping_patience,
            "n_pop": args.emulator_n_pop,
            "n_ind_per_pop": args.emulator_n_ind_per_pop,
            "emulator_d_model": args.emulator_d_model,
            "emulator_n_heads": args.emulator_n_heads,
            "emulator_n_layers": args.emulator_n_layers,
            "emulator_dropout": args.emulator_dropout,
        },
        theta_trust_center=(
            emulator_theta0 if emulator_active_for_fit and args.emulator_theta_trust_radius > 0 else None
        ),
        theta_trust_log_radius=(
            args.emulator_theta_trust_radius if emulator_active_for_fit and args.emulator_theta_trust_radius > 0 else None
        ),
        checkpoint_dir=output_dir,
        checkpoint_every=args.saem_checkpoint_every,
        stop_after_iter=args.stop_after_iter,
    )
    saem.etas = {int(k): v for k, v in eta_init_dict.items()}  # warm‑start
    if resume_state is not None:
        saem.trace = list(resume_state["trace"])
    logger.info("Iniciando SAEM...")
    try:
        theta_hat, Omega_hat, sigma_hat, etas_final = saem.fit()
    except KeyboardInterrupt:
        saem.save_checkpoint(force=True, status="interrupted")
        logger.warning("SAEM interrumpido. Se conservan checkpoints parciales en %s", output_dir)
        return
    except Exception:
        saem.save_checkpoint(force=True, status="failed")
        logger.exception("SAEM abortado por error. Se conservan checkpoints parciales en %s", output_dir)
        raise
    if saem.stop_reason == "stop_after_iter":
        logger.info(
            "SAEM detenido de forma controlada en iter %d por --stop_after_iter=%d. "
            "Se conservan checkpoints parciales y se omiten diagnosticos finales.",
            saem.trace[-1]["iter"] if saem.trace else 0,
            args.stop_after_iter,
        )
        return
    logger.info("SAEM finalizado. theta_hat=%s, sigma_hat=%s", theta_hat, sigma_hat)
    if emulator_active_for_fit:
        disable_emulator()
        logger.info("Modo emulador desactivado antes de errores estandar y diagnosticos finales; se usarÃ¡ DECK real.")
    save_config(
        {
            "theta_hat": np.asarray(theta_hat, dtype=float).tolist(),
            "Omega_hat": np.asarray(Omega_hat, dtype=float).tolist(),
            "sigma_hat": np.asarray(sigma_hat, dtype=float).tolist(),
        },
        os.path.join(output_dir, "final_estimates.json")
    )

    # --- Cálculo de errores estándar (opcional) ---
    if args.compute_se:
        from diagnostics import compute_population_se
        logger.info("Calculando errores estándar poblacionales (esto puede llevar tiempo)...")
        # Usamos el deck_executor de la instancia SAEM
        se_df = compute_population_se(
            theta_hat, Omega_hat, sigma_hat, data, ids,
            model_cfg.__dict__, deck_executor=saem.deck_executor
        )
        if se_df is not None:
            se_path = os.path.join(output_dir, "population_SE.csv")
            se_df.to_csv(se_path, index=False)
            logger.info(f"Errores estándar guardados en {se_path}")

    # Guardar traza
    trace_df = saem.trace_as_dataframe()
    trace_path = os.path.join(output_dir, "SAEM_trace.csv")
    trace_df.to_csv(trace_path, index=False)
    logger.info("Traza guardada en %s", trace_path)

    # Diagnósticos
    logger.info("Calculando EBEs y shrinkage...")
    ebes_df = compute_ebes(
        theta_hat, Omega_hat, sigma_hat, data, ids, model_cfg,
        etas_final, deck_executor=saem.deck_executor
    )
    ebes_path = os.path.join(output_dir, "EBEs.csv")
    ebes_df.to_csv(ebes_path, index=False)
    logger.info("EBEs guardadas en %s", ebes_path)

    shrinkage_df = compute_shrinkage(ebes_df, Omega_hat)
    shrinkage_path = os.path.join(output_dir, "shrinkage.csv")
    shrinkage_df.to_csv(shrinkage_path, index=False)
    logger.info("Shrinkage guardado en %s", shrinkage_path)

    logger.info("Generando gráficos de diagnóstico...")
    iwres_df, eps_shrink = generate_diagnostic_plots(
        ebes_df, data, theta_hat, Omega_hat, sigma_hat, model_cfg,
        output_dir, deck_executor=saem.deck_executor, ids=ids
    )
    iwres_path = os.path.join(output_dir, "IWRES.csv")
    iwres_df.to_csv(iwres_path, index=False)
    logger.info("IWRES guardado en %s", iwres_path)

    if args.use_emulator and args.validate_emulator_etas:
        if (
            trained_emulator is None
            or (
                trained_scaler_y is None
                and not hasattr(trained_emulator, "predict_sequence")
            )
        ):
            logger.warning(
                "No se puede ejecutar la validacion avanzada del emulador porque faltan el emulador activo o el scaler."
            )
        else:
            from ml_utils import validate_emulator_with_etas

            logger.info("Validando emulador con ETAs muestreadas...")
            val_etas_results = validate_emulator_with_etas(
                trained_emulator,
                trained_max_len,
                trained_scaler_y,
                data,
                ids,
                theta_hat,
                Omega_hat,
                sigma_hat,
                model_cfg,
                n_etas_per_id=args.emulator_eta_validation_n_etas,
                n_val=args.emulator_eta_validation_n_val,
                deck_executor=saem.deck_executor
            )
            for key, value in val_etas_results.items():
                if isinstance(value, float):
                    logger.info("  %s: %.4f", key, value)
                else:
                    logger.info("  %s: %s", key, value)
            val_etas_path = os.path.join(output_dir, "emulator_validation_etas_final.csv")
            pd.DataFrame([val_etas_results]).to_csv(val_etas_path, index=False)
            logger.info("Validacion avanzada del emulador guardada en %s", val_etas_path)

    # Restaurar deck real si se usó emulador
    logger.info("Experimento finalizado. Resultados en %s", output_dir)

if __name__ == "__main__":
    main()
