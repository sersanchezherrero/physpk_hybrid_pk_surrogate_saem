# ml_utils.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
import hashlib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from models import individual_params, residual_sd as residual_model
from deck_interface import (
    XGBPointwiseEmulator,
    build_xgb_pointwise_features,
    build_schedule_signature,
    build_schedule_signature_from_arrays,
    original_run_deck,
    run_deck,
    run_deck_emulated,
)
import logging

EMULATOR_DATA_VERSION = 2


# ----------------------------------------------------------------------
# Funciones auxiliares para el emulador (necesarias en train_emulator)
# ----------------------------------------------------------------------
def _hash_params(theta, Omega, sigma):
    theta = np.asarray(theta)
    Omega = np.asarray(Omega)
    sigma = np.asarray(sigma)
    param_bytes = theta.tobytes() + Omega.tobytes() + sigma.tobytes()
    return hashlib.md5(param_bytes).hexdigest()


def _emulator_artifact_stem(model_path):
    for extension in (".keras", ".h5", ".joblib"):
        if model_path.endswith(extension):
            return model_path[: -len(extension)]
    return model_path


def _emulator_data_cache_path(model_path, emulator_data_mode):
    stem = _emulator_artifact_stem(model_path)
    if emulator_data_mode == "lhs":
        return f"{stem}_data.npz"
    return f"{stem}_{emulator_data_mode}_data.npz"


def _emulator_scaler_path(model_path):
    return f"{_emulator_artifact_stem(model_path)}_scaler_y.pkl"


def _emulator_meta_path(model_path):
    return f"{_emulator_artifact_stem(model_path)}_meta.json"


def _emulator_keras_path(model_path):
    return f"{_emulator_artifact_stem(model_path)}.keras"


def _emulator_joblib_path(model_path):
    return f"{_emulator_artifact_stem(model_path)}.joblib"


def _emulator_baseline_cache_path(model_path):
    return f"{_emulator_artifact_stem(model_path)}_baseline_cache.joblib"


def save_emulator_data(filepath, Xp, Xs, y, lens, max_len, theta, Omega, sigma):
    """Guarda el dataset y los parámetros usados."""
    param_hash = _hash_params(theta, Omega, sigma)
    np.savez_compressed(
        filepath,
        Xp=Xp, Xs=Xs, y=y, lens=lens, max_len=max_len,
        param_hash=param_hash,
        data_version=np.array([EMULATOR_DATA_VERSION], dtype=int)
    )
    with open(filepath.replace('.npz', '_meta.json'), 'w') as f:
        json.dump(
            {
                'max_len': int(max_len),
                'n_params': len(theta),
                'data_version': EMULATOR_DATA_VERSION
            },
            f
        )

def load_emulator_data(filepath, theta, Omega, sigma):
    """Carga el dataset si los parámetros coinciden."""
    data = np.load(filepath, allow_pickle=True)
    param_hash = _hash_params(theta, Omega, sigma)
    data_version = int(np.asarray(data.get('data_version', np.array([1]))).ravel()[0])
    if data_version != EMULATOR_DATA_VERSION:
        return None
    if data['param_hash'] != param_hash:
        return None
    return data['Xp'], data['Xs'], data['y'], data['lens'], data['max_len']


def build_sequence_mask(lengths, max_len, dtype=np.float32):
    lengths = np.asarray(lengths, dtype=int).reshape(-1)
    positions = np.arange(int(max_len))[None, :]
    mask = positions < lengths[:, None]
    return mask.astype(dtype)


def fit_target_scaler(y, lengths):
    from sklearn.preprocessing import StandardScaler

    mask = build_sequence_mask(lengths, y.shape[1], dtype=bool)
    scaler = StandardScaler()
    scaler.fit(y[mask].reshape(-1, 1))
    return scaler


def scale_target_sequences(y, lengths, scaler):
    mask = build_sequence_mask(lengths, y.shape[1], dtype=bool)
    y_scaled = np.zeros_like(y, dtype=float)
    y_scaled[mask] = scaler.transform(y[mask].reshape(-1, 1))
    return y_scaled


def inverse_scale_target_sequences(y_scaled, lengths, scaler):
    mask = build_sequence_mask(lengths, y_scaled.shape[1], dtype=bool)
    y = np.zeros_like(y_scaled, dtype=float)
    y[mask] = scaler.inverse_transform(y_scaled[mask].reshape(-1, 1))
    return y


def masked_mae(y_true, y_pred, lengths):
    mask = build_sequence_mask(lengths, y_true.shape[1], dtype=bool)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _transform_emulator_targets(values, target_transform):
    target_transform = str(target_transform or "identity").lower()
    values = np.asarray(values, dtype=float)

    if target_transform == "identity":
        return values
    if target_transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))

    raise ValueError(f"Transformacion de target no soportada: {target_transform}")


def build_xgb_baseline_cache(data, ids, theta_center, Xs=None, lens=None):
    theta_center = np.asarray(theta_center, dtype=float)
    cache = {}

    def _store_schedule(time, dose, duration):
        signature = build_schedule_signature_from_arrays(time, dose, duration)
        if signature in cache:
            return
        schedule_df = pd.DataFrame({
            "TIME": np.asarray(time, dtype=float),
            "Dose": np.asarray(dose, dtype=float),
            "Duration": np.asarray(duration, dtype=float),
        })
        cache[signature] = np.asarray(original_run_deck(theta_center, schedule_df), dtype=float)

    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        if data_ind.empty:
            continue
        _store_schedule(
            data_ind["TIME"].to_numpy(dtype=float),
            data_ind["Dose"].to_numpy(dtype=float),
            data_ind["Duration"].to_numpy(dtype=float),
        )

    if Xs is not None and lens is not None:
        for idx in range(int(Xs.shape[0])):
            n_rows = int(lens[idx])
            if n_rows <= 0:
                continue
            seq = np.asarray(Xs[idx, :n_rows], dtype=float)
            _store_schedule(seq[:, 0], seq[:, 1], seq[:, 2])

    return cache


def flatten_xgb_pointwise_dataset(
    Xp,
    Xs,
    y,
    lens,
    baseline_cache=None,
    residual_mode=False,
    baseline_feature=False,
):
    X_rows = []
    y_rows = []
    baseline_cache = baseline_cache or {}

    for idx in range(Xp.shape[0]):
        n_rows = int(lens[idx])
        if n_rows <= 0:
            continue
        seq = Xs[idx, :n_rows]
        baseline = None
        if residual_mode or baseline_feature:
            signature = build_schedule_signature_from_arrays(seq[:, 0], seq[:, 1], seq[:, 2])
            baseline = np.asarray(
                baseline_cache.get(signature, np.zeros(n_rows, dtype=float)),
                dtype=float,
            ).reshape(-1)
            if baseline.shape[0] != n_rows:
                baseline = np.zeros(n_rows, dtype=float)
        features = build_xgb_pointwise_features(
            Xp[idx],
            seq[:, 0],
            seq[:, 1],
            seq[:, 2],
            baseline=baseline if baseline_feature else None,
        )
        X_rows.append(features)
        target = np.asarray(y[idx, :n_rows, 0], dtype=float)
        if residual_mode:
            target = target - baseline
        y_rows.append(target)

    if not X_rows:
        n_features = int(2 * Xp.shape[1] + 7 + (1 if baseline_feature else 0))
        return np.zeros((0, n_features), dtype=np.float32), np.zeros(0, dtype=float)

    return np.vstack(X_rows).astype(np.float32), np.concatenate(y_rows).astype(float)


def build_xgb_pointwise_sample_weights(
    Xs,
    y,
    lens,
    peak_emphasis=4.0,
    peak_window=1,
    dose_emphasis=1.5,
):
    weights = []
    peak_emphasis = max(float(peak_emphasis), 0.0)
    peak_window = max(int(peak_window), 0)
    dose_emphasis = max(float(dose_emphasis), 0.0)

    for idx in range(int(len(lens))):
        n_rows = int(lens[idx])
        if n_rows <= 0:
            continue
        seq = np.asarray(Xs[idx, :n_rows], dtype=float)
        target = np.asarray(y[idx, :n_rows, 0], dtype=float)
        row_weights = np.ones(n_rows, dtype=float)

        if target.size > 0 and peak_emphasis > 0.0:
            peak_idx = int(np.argmax(target))
            lo = max(0, peak_idx - peak_window)
            hi = min(n_rows, peak_idx + peak_window + 1)
            row_weights[lo:hi] += peak_emphasis

        if dose_emphasis > 0.0:
            row_weights[seq[:, 1] > 0.0] += dose_emphasis

        weights.append(row_weights)

    if not weights:
        return np.zeros(0, dtype=float)

    return np.concatenate(weights).astype(float)

def generate_emulator_data(
    data, ids, cfg,
    n_pop=2000,               # número de combinaciones poblacionales
    n_ind_per_pop=10,          # individuos por población
    theta_center=None,         # centro para theta (si None, se usan valores por defecto)
    theta_sigma=0.5,           # desviación en escala log para theta
    omega_range=(0.01, 0.5),   # rango para diagonales de Omega (en escala natural)
    sigma_range=(0.01, 0.5),   # rango para sigma
    dose_range=(50, 200),      # rango de dosis (min, max)
    seed=123
):
    """
    Genera dataset para emulador usando Latin Hypercube Sampling,
    incluyendo variabilidad en dosis.
    """
    from scipy.stats import qmc, norm
    rng = np.random.default_rng(seed)
    max_len = max(data.groupby("ID").size())
    n_par = cfg.n_par
    n_sigma = cfg.n_sigma

    # Centro de theta (si no se da, usar valores típicos)
    if theta_center is None:
        theta_center = np.array([7.0, 25.0, 50.0, 10.0])  # Ajusta según tu fármaco

    # Número total de dimensiones poblacionales: theta (n_par) + omega diag (n_par) + sigma (n_sigma)
    n_dim = n_par + n_par + n_sigma

    # Generar LHS
    lhs = qmc.LatinHypercube(d=n_dim, seed=seed)
    samples = lhs.random(n=n_pop)   # forma (n_pop, n_dim)

    # Transformar las columnas
    u_theta = samples[:, :n_par]
    z_theta = norm.ppf(u_theta)
    log_theta = np.log(theta_center) + theta_sigma * z_theta
    theta_pop = np.exp(log_theta)   # (n_pop, n_par)

    u_omega = samples[:, n_par:2*n_par]
    omega_diag = omega_range[0] + u_omega * (omega_range[1] - omega_range[0])
    Omega_list = [np.diag(omega_diag[i]) for i in range(n_pop)]

    u_sigma = samples[:, -n_sigma:]
    sigma_pop = sigma_range[0] + u_sigma * (sigma_range[1] - sigma_range[0])
    if n_sigma == 1:
        sigma_pop = sigma_pop.reshape(-1, 1)

    X_params_list = []
    X_seq_list = []
    y_list = []
    len_list = []

    for i_pop in range(n_pop):
        theta = theta_pop[i_pop]
        Omega = Omega_list[i_pop]
        sigma = sigma_pop[i_pop]

        if (i_pop + 1) % max(1, n_pop // 10) == 0 or i_pop == 0:
            print(f"  Progreso dataset LHS: poblacion {i_pop + 1}/{n_pop}")

        for j in range(n_ind_per_pop):
            i_id = rng.choice(ids)
            template = data[data["ID"] == i_id].copy()
            n_rows = template.shape[0]
            if n_rows == 0:
                continue

            # --- Variabilidad en dosis ---
            # Muestrear una dosis aleatoria
            dose = rng.uniform(dose_range[0], dose_range[1])
            # Asumimos que la dosis se administra en el primer tiempo (t=0) y luego es 0.
            # Guardamos la dosis original para posible escalado (si hubiera múltiples dosis, habría que escalar)
            # Por simplicidad, reemplazamos toda la columna Dose con la nueva dosis en la primera fila y cero en las demás.
            template["Dose"] = 0.0
            template.loc[template.index[0], "Dose"] = dose
            # Nota: si hubiera múltiples dosis, podríamos escalar proporcionalmente:
            # original_dose = template["Dose"].max()  # o la primera dosis
            # if original_dose > 0:
            #     template["Dose"] = template["Dose"] * (dose / original_dose)

            # Muestrear eta
            eta = rng.multivariate_normal(mean=np.zeros(n_par), cov=Omega)
            pars_ind = individual_params(theta, eta, cfg.re_type)

            # Simular con deck real
            from deck_interface import original_run_deck
            pred_all = original_run_deck(pars_ind, template)

            # Guardar
            X_params_list.append(pars_ind)
            seq_pad = np.zeros((max_len, 3))
            seq_pad[:n_rows] = template[["TIME", "Dose", "Duration"]].values
            X_seq_list.append(seq_pad)
            y_pad = np.zeros((max_len, 1))
            # The emulator must reproduce the raw DECK output scale as-is.
            y_pad[:n_rows, 0] = np.asarray(pred_all, dtype=float)
            y_list.append(y_pad)
            len_list.append(n_rows)

    X_params = np.array(X_params_list)
    X_seq = np.array(X_seq_list)
    y = np.array(y_list)
    lengths = np.array(len_list)
    return X_params, X_seq, y, lengths, max_len


def generate_emulator_data_local(data, ids, theta_pop, Omega, sigma, cfg, n_virtual, seed=123):
    """
    Genera un dataset local alrededor de una sola población.
    Reutiliza los calendarios reales de los individuos y solo introduce
    variabilidad interindividual a través de eta.
    """
    rng = np.random.default_rng(seed)
    max_len = max(data.groupby("ID").size())
    n_par = cfg.n_par
    from deck_interface import original_run_deck as real_run_deck

    templates = []
    for i_id in ids:
        template = data[data["ID"] == i_id].copy()
        if template.shape[0] > 0:
            templates.append(template)

    if not templates:
        raise ValueError("No hay individuos validos para generar el dataset local del emulador.")

    n_virtual = int(max(1, n_virtual))
    if n_virtual <= len(templates):
        selected_idx = rng.choice(len(templates), size=n_virtual, replace=False).tolist()
    else:
        selected_idx = list(range(len(templates)))
        selected_idx.extend(rng.choice(len(templates), size=n_virtual - len(templates), replace=True).tolist())
        rng.shuffle(selected_idx)

    X_params_list = []
    X_seq_list = []
    y_list = []
    len_list = []

    for template_idx in selected_idx:
        template = templates[template_idx].copy()
        n_rows = template.shape[0]

        eta = rng.multivariate_normal(mean=np.zeros(n_par), cov=Omega)
        pars_ind = individual_params(theta_pop, eta, cfg.re_type)
        pred_all = np.asarray(real_run_deck(pars_ind, template), dtype=float)

        X_params_list.append(np.asarray(pars_ind, dtype=float))
        seq_pad = np.zeros((max_len, 3))
        seq_pad[:n_rows] = template[["TIME", "Dose", "Duration"]].values
        X_seq_list.append(seq_pad)

        y_pad = np.zeros((max_len, 1))
        y_pad[:n_rows, 0] = pred_all
        y_list.append(y_pad)
        len_list.append(n_rows)

    X_params = np.array(X_params_list, dtype=float)
    X_seq = np.array(X_seq_list, dtype=float)
    y = np.array(y_list, dtype=float)
    lengths = np.array(len_list, dtype=int)
    return X_params, X_seq, y, lengths, max_len

# ----------------------------------------------------------------------
# Funciones de features
# ----------------------------------------------------------------------
def _safe_auc_trapz(t, y):
    t = np.asarray(t)
    y = np.asarray(y)
    if t.size < 2:
        return 0.0
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    return float(np.trapezoid(y, t))

def build_pk_features_from_individual(data_ind, obs_col, meas_col="Meas"):
    """
    Extrae features de un individuo a partir de sus observaciones.

    Args:
        data_ind (pd.DataFrame): Datos de un solo individuo (debe contener columnas TIME, Dose, Duration, y la columna de observación).
        obs_col (str): Nombre de la columna que contiene las observaciones (por ejemplo, "LNCONC").
        meas_col (str): Nombre de la columna que indica si la observación es válida (por defecto "Meas").

    Returns:
        np.array: Vector de 12 características (n, t_min, t_max, cmax, tmax, cmin, auc,
                  slope_early, slope_late, dose_sum, dose_max, dur).
    """
    df = data_ind.copy()
    if meas_col in df.columns:
        df = df[df[meas_col].to_numpy() != 0].copy()
    if df.shape[0] == 0:
        return np.zeros(12, dtype=float)

    t = df["TIME"].to_numpy(dtype=float)
    y = df[obs_col].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size == 0:
        return np.zeros(12, dtype=float)

    n = float(t.size)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    idx_max = int(np.argmax(y))
    cmax = float(y[idx_max])
    tmax = float(t[idx_max])
    cmin = float(np.min(y))
    auc = _safe_auc_trapz(t, y)

    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]

    slope_early = 0.0
    if t_sorted.size >= 2:
        dt = t_sorted[1] - t_sorted[0]
        if abs(dt) > 1e-12:
            slope_early = float((y_sorted[1] - y_sorted[0]) / dt)

    slope_late = 0.0
    if t_sorted.size >= 3:
        tt = t_sorted[-3:]
        yy = y_sorted[-3:]
        denom = float(np.sum((tt - tt.mean()) ** 2))
        if denom > 1e-12:
            slope_late = float(np.sum((tt - tt.mean()) * (yy - yy.mean())) / denom)

    dose_sum = float(df["Dose"].sum()) if "Dose" in df.columns else 0.0
    dose_max = float(df["Dose"].max()) if "Dose" in df.columns else 0.0
    dur_sum = float(df["Duration"].sum()) if "Duration" in df.columns else 0.0
    dur_max = float(df["Duration"].max()) if "Duration" in df.columns else 0.0
    dur = dur_max if dur_max > 0 else dur_sum

    feats = np.array([
        n, t_min, t_max, cmax, tmax, cmin, auc,
        slope_early, slope_late,
        dose_sum, dose_max,
        dur
    ], dtype=float)
    feats[~np.isfinite(feats)] = 0.0
    return feats

# ----------------------------------------------------------------------
# Generación de individuos virtuales y datasets
# ----------------------------------------------------------------------
def generate_virtual_individual(theta_pop, Omega, sigma, data_ind_template, cfg, rng, simulate_fn=None):
    """Crea un individuo virtual y devuelve (features, eta)."""
    n_par = cfg.n_par
    eta = rng.multivariate_normal(mean=np.zeros(n_par), cov=Omega).astype(float)
    pars_ind = individual_params(theta_pop, eta, cfg.re_type)

    if simulate_fn is None:
        simulate_fn = original_run_deck

    pred_all = simulate_fn(pars_ind, data_ind_template)
    if "Meas" in data_ind_template.columns:
        mask = data_ind_template["Meas"].to_numpy() != 0
    else:
        mask = np.ones_like(pred_all, dtype=bool)

    pred = pred_all[mask].astype(float)

    if cfg.transform == "log":
        pred = np.clip(pred, 1e-12, np.inf)
        pred_t = np.log(pred)
    else:
        pred_t = pred

    sd = residual_model(pred, sigma=sigma, err_model=cfg.err_model, transform=cfg.transform)
    eps = rng.normal(loc=0.0, scale=sd, size=pred_t.shape)
    y_t = pred_t + eps

    virt = data_ind_template.copy()
    obs_col = cfg.obs_column
    if obs_col not in virt.columns:
        virt[obs_col] = np.nan
    virt.loc[virt.index[mask], obs_col] = y_t

    feats = build_pk_features_from_individual(virt, obs_col=obs_col, meas_col="Meas")
    return feats, eta

def build_virtual_dataset(data, ids, theta, Omega, sigma, cfg, n_per_id, seed=123, simulate_fn=None):
    """Genera dataset virtual para warm‑start de ETAs."""
    X, Y = [], []
    rng = np.random.default_rng(seed)
    for i_id in ids:
        template = data[data["ID"] == i_id].copy()
        for _ in range(n_per_id):
            feats, eta = generate_virtual_individual(
                theta, Omega, sigma, template, cfg, rng, simulate_fn=simulate_fn
            )
            X.append(feats)
            Y.append(eta)
    return np.array(X), np.array(Y)

# ----------------------------------------------------------------------
# Modelos de ML para ETAs
# ----------------------------------------------------------------------
def train_eta_warmstart_model(X, Y, seed=123, compute_shap=False):
    """
    Train one XGBoost per ETA with early stopping.
    X: (n_samples, n_features)
    Y: (n_samples, n_par)
    Returns: list of trained XGBoost models (one per output)
    """
    from sklearn.model_selection import train_test_split
    logger = logging.getLogger(__name__)
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=seed)
    
    n_outputs = Y.shape[1]
    models = []
    
    for i in range(n_outputs):
        logger.info(f"Training model for ETA{i+1}...")
        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            early_stopping_rounds=50,
            eval_metric='mae',
            verbosity=0
        )
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False
        )
        models.append(model)
        logger.info(f"  Best iteration: {model.best_iteration}, best MAE: {model.best_score:.4f}")
    
    if compute_shap:
        logger.info("Computing SHAP values for ETA1 (this may take a few seconds)...")
        explainer = shap.TreeExplainer(models[0])
        X_sample = shap.sample(X_train, 100)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=[f"feat_{j}" for j in range(X.shape[1])],
            show=False
        )
        plt.title("SHAP summary for ETA1")
        plt.tight_layout()
        plt.savefig("SHAP_ETA1.png", dpi=150)
        plt.close()

        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type="bar",
            feature_names=[f"feat_{j}" for j in range(X.shape[1])],
            show=False
        )
        plt.title("Feature importance for ETA1")
        plt.tight_layout()
        plt.savefig("SHAP_importance_ETA1.png", dpi=150)
        plt.close()
    
    return models

# ----------------------------------------------------------------------
# Red neuronal para posterior de ETAs
# ----------------------------------------------------------------------
# ======================================================================
# Red neuronal para posterior de ETAs (VERSIÓN CORREGIDA)
# ======================================================================

def nll_loss(y_true, y_pred):
    """Pérdida de log-verosimilitud negativa para una distribución normal con media y logvar."""
    n_par = y_pred.shape[-1] // 2
    mu = y_pred[..., :n_par]
    logvar = y_pred[..., n_par:]
    variance = tf.exp(logvar)
    return 0.5 * tf.reduce_mean(tf.math.log(variance) + (y_true - mu)**2 / variance)

def build_eta_posterior_net(input_dim, n_par):
    inp = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(2 * n_par)(x)  # (mu, logvar)
    return tf.keras.Model(inp, out)

def train_eta_posterior_net(model, X, mu, scaler=None, epochs=30, batch_size=64):
    """
    Entrena la red para predecir mu y logvar.
    Si se proporciona scaler, se aplica a X.
    """
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=nll_loss)
    model.fit(X_scaled, mu, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def sample_from_eta_net(model, feats, rng, scaler=None):
    if scaler is not None:
        feats_scaled = scaler.transform(feats.reshape(1, -1))
    else:
        feats_scaled = feats.reshape(1, -1)
    out = model(feats_scaled, training=False).numpy().ravel()
    n_par = out.shape[0] // 2
    mu = out[:n_par]
    logvar = out[n_par:]
    sd = np.exp(0.5 * np.clip(logvar, -10, 4))
    return mu + sd * rng.normal(size=mu.shape)

# ----------------------------------------------------------------------
# ML poblacional (warm‑start de theta, Omega, sigma)
# ----------------------------------------------------------------------
def predict_eta_init_for_real_data(models, data, ids, cfg):
    """
    For each real ID, compute features and predict ETA init using list of XGBoost models.
    Returns dict: {ID: eta_init}
    """
    eta_init = {}
    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        feats = build_pk_features_from_individual(
            data_ind, obs_col=cfg.obs_column, meas_col="Meas"
        )
        # Predict with each model and concatenate
        eta = np.array([m.predict(feats.reshape(1, -1))[0] for m in models])
        eta_init[int(i_id)] = eta
    return eta_init

# ----------------------------------------------------------------------
# Entrenamiento del emulador LSTM
# ----------------------------------------------------------------------
def extract_population_features(data, ids, obs_col, meas_col="Meas"):
    """
    Extract population-level features from all individuals.
    Returns a 1D numpy array.
    """
    all_feats = []
    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        feats_ind = build_pk_features_from_individual(data_ind, obs_col, meas_col)
        all_feats.append(feats_ind)
    all_feats = np.array(all_feats)  # (n_ids, n_feat)
    
    pop_feats = []
    for j in range(all_feats.shape[1]):
        col = all_feats[:, j]
        pop_feats.extend([
            np.mean(col),
            np.median(col),
            np.std(col),
            np.percentile(col, 25),
            np.percentile(col, 75),
            np.min(col),
            np.max(col)
        ])
    return np.array(pop_feats, dtype=float)


def generate_population_virtual_dataset(
    data, ids, cfg,
    n_pop_sets, n_ind_per_set,
    param_ranges_theta,  # list of (min, max) tuples for each theta
    omega_scale_range=(0.01, 0.5),
    sigma_range=(0.01, 0.5),
    seed=123,
    use_emulator=False,
    emulator=None,
    max_len=None,
    scaler_y=None
):
    """
    Generate a virtual dataset for training the population warm-start model.
    Returns:
        X: (n_pop_sets, n_pop_feat)
        Y: (n_pop_sets, n_par + n_par*(n_par+1)//2 + n_sigma)  (theta + L_flat + log_sigma)
    """
    rng = np.random.default_rng(seed)
    n_par = cfg.n_par
    n_sigma = cfg.n_sigma
    n_L = n_par * (n_par + 1) // 2

    X_list = []
    Y_list = []

    for _ in range(n_pop_sets):
        # Sample theta within given ranges
        theta = np.array([rng.uniform(low, high) for low, high in param_ranges_theta])

        # Sample Omega (diagonal for simplicity, could add correlations)
        omega_diag_log = rng.uniform(np.log(omega_scale_range[0]**2), np.log(omega_scale_range[1]**2), size=n_par)
        Omega = np.diag(np.exp(omega_diag_log))
        Omega = 0.5 * (Omega + Omega.T)
        L = np.linalg.cholesky(Omega)
        L_flat = L[np.tril_indices(n_par)]

        # Sample sigma (log scale)
        sigma_log = rng.uniform(np.log(sigma_range[0]), np.log(sigma_range[1]), size=n_sigma)

        # Simulate individuals for this population set
        pop_feats_list = []
        for _ in range(n_ind_per_set):
            i_id = rng.choice(ids)
            data_ind = data[data["ID"] == i_id].copy()
            eta = rng.multivariate_normal(mean=np.zeros(n_par), cov=Omega)
            pars_ind = individual_params(theta, eta, cfg.re_type)

            # Simulate concentrations (use emulator if available)
            if use_emulator and emulator is not None:
                pred_all = run_deck_emulated(pars_ind, data_ind, emulator, max_len, scaler_y)
            else:
                pred_all = run_deck(pars_ind, data_ind)

            # Add residual noise
            if "Meas" in data_ind.columns:
                mask = data_ind["Meas"].to_numpy() != 0
            else:
                mask = np.ones_like(pred_all, dtype=bool)

            pred = pred_all[mask]
            if cfg.transform == "log":
                pred = np.clip(pred, 1e-12, np.inf)
                y_t = np.log(pred)
            else:
                y_t = pred

            sd = residual_model(pred, sigma=np.exp(sigma_log), err_model=cfg.err_model, transform=cfg.transform)
            eps = rng.normal(0, sd)
            y_obs_t = y_t + eps

            data_ind_virt = data_ind.copy()
            obs_col = cfg.obs_column
            data_ind_virt.loc[mask, obs_col] = y_obs_t
            feats_ind = build_pk_features_from_individual(data_ind_virt, obs_col, "Meas")
            pop_feats_list.append(feats_ind)

        pop_feats = np.array(pop_feats_list)
        pop_feats_agg = []
        for j in range(pop_feats.shape[1]):
            col = pop_feats[:, j]
            pop_feats_agg.extend([
                np.mean(col),
                np.median(col),
                np.std(col),
                np.percentile(col, 25),
                np.percentile(col, 75),
                np.min(col),
                np.max(col)
            ])
        pop_feats_agg = np.array(pop_feats_agg)

        y_vec = np.concatenate([theta, L_flat, sigma_log])
        X_list.append(pop_feats_agg)
        Y_list.append(y_vec)

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y


def train_population_warmstart_model(X, Y, seed=123):
    """
    Train one XGBoost per output component with early stopping.
    X: (n_pop_sets, n_pop_feat)
    Y: (n_pop_sets, n_par + n_L + n_sigma)
    Returns: list of XGBoost models (one per output)
    """
    from sklearn.model_selection import train_test_split
    logger = logging.getLogger(__name__)
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=seed)
    
    n_outputs = Y.shape[1]
    models = []
    
    for i in range(n_outputs):
        logger.info(f"Training model for output component {i+1}/{n_outputs}...")
        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            early_stopping_rounds=50,
            eval_metric='mae',
            verbosity=0
        )
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False
        )
        models.append(model)
        logger.info(f"  Best iteration: {model.best_iteration}, best MAE: {model.best_score:.4f}")
    
    # SHAP for first output (e.g., THETA1)
    logger.info("Computing SHAP values for first output (THETA1)...")
    explainer = shap.TreeExplainer(models[0])
    X_sample = shap.sample(X_train, 100)
    shap_values = explainer.shap_values(X_sample)
    
    shap.summary_plot(shap_values, X_sample, feature_names=[f"pop_feat_{j}" for j in range(X.shape[1])], show=False)
    plt.title("SHAP summary for THETA1")
    plt.tight_layout()
    plt.savefig("SHAP_THETA1.png", dpi=150)
    plt.show()
    
    shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=[f"pop_feat_{j}" for j in range(X.shape[1])], show=False)
    plt.title("Feature importance for THETA1")
    plt.tight_layout()
    plt.savefig("SHAP_importance_THETA1.png", dpi=150)
    plt.show()
    
    return models


def predict_pop_params_for_real_data(models, data, ids, cfg):
    """
    Apply the list of XGBoost models to real data to get theta0, L0_flat, sigma0_log.
    Returns: theta0, Omega0, sigma0_log
    """
    n_par = cfg.n_par
    n_sigma = cfg.n_sigma
    n_L = n_par * (n_par + 1) // 2
    
    pop_feats = extract_population_features(data, ids, cfg.obs_column, "Meas")
    y_pred = np.array([m.predict(pop_feats.reshape(1, -1))[0] for m in models])
    
    theta0 = y_pred[:n_par]
    L_flat = y_pred[n_par:n_par + n_L]
    sigma0_log = y_pred[n_par + n_L:]
    
    L = np.zeros((n_par, n_par))
    idx = 0
    for i in range(n_par):
        for j in range(i + 1):
            L[i, j] = L_flat[idx]
            idx += 1
    Omega0 = L @ L.T
    Omega0[np.diag_indices_from(Omega0)] += 1e-6  # regularize
    
    return theta0, Omega0, sigma0_log

def _legacy_train_emulator(
    data, ids, theta_pop, Omega, sigma, cfg, n_virtual,
    model_path, data_path=None, seed=123, force_retrain=False,
    validation_split=0.15, epochs=30, batch_size=64,
    early_stopping_patience=8,
    # Parámetros para la generación de datos (opcionales)
    n_pop=200, n_ind_per_pop=5, theta_sigma=0.5,
    omega_range=(0.01, 0.5), sigma_range=(0.01, 0.5),
    dose_range=(50, 200),          # <--- NUEVO
    virtual_data_path=None,
    emulator_data_mode="local",
    emulator_model_type="lstm",
    emulator_d_model=64,
    emulator_n_heads=4,
    emulator_n_layers=2,
    emulator_dropout=0.1
):
    """
    Entrena el emulador.
    Si virtual_data_path no es None, carga el dataset desde ese archivo .npz
    y omite la generación con el DECK real.
    """
    emulator_data_mode = str(emulator_data_mode).lower()
    if emulator_data_mode not in {"lhs", "local"}:
        raise ValueError(f"Modo de dataset de emulador no soportado: {emulator_data_mode}")
    emulator_model_type = str(emulator_model_type).lower()
    if emulator_model_type not in {"transformer", "lstm", "xgb_pointwise", "xgb_residual_pointwise"}:
        raise ValueError(f"Tipo de emulador no soportado: {emulator_model_type}")
    print(
        "Configuracion emulador: "
        f"modo={emulator_data_mode}, n_virtual={n_virtual}, n_pop={n_pop}, "
        f"n_ind_per_pop={n_ind_per_pop}, epochs={epochs}, batch_size={batch_size}, "
        f"modelo={emulator_model_type}"
    )

    if data_path is None:
        data_path = _emulator_data_cache_path(model_path, emulator_data_mode)

    # Si se proporciona un dataset virtual, cargarlo directamente
    if virtual_data_path is not None:
        if not os.path.exists(virtual_data_path):
            raise FileNotFoundError(f"Archivo de dataset virtual no encontrado: {virtual_data_path}")
        print(f"Cargando dataset virtual desde {virtual_data_path}")
        # Cargar el archivo .npz (debe tener las mismas claves que save_emulator_data)
        loaded = np.load(virtual_data_path, allow_pickle=True)
        data_version = int(np.asarray(loaded.get('data_version', np.array([1]))).ravel()[0])
        if data_version != EMULATOR_DATA_VERSION:
            raise ValueError(
                f"Dataset virtual incompatible (version {data_version}). Regenera el archivo con el código actual."
            )
        Xp = loaded['Xp']
        Xs = loaded['Xs']
        y = loaded['y']
        lens = loaded['lens']
        max_len = int(loaded['max_len'])
        # Opcional: podrías verificar que los parámetros coinciden, pero no es necesario
    else:
        # Comportamiento original: cargar desde data_path o generar
        if os.path.exists(data_path) and not force_retrain:
            print(f"Cargando dataset desde {data_path}")
            loaded = load_emulator_data(data_path, theta_pop, Omega, sigma)
            if loaded is not None:
                Xp, Xs, y, lens, max_len = loaded
            else:
                force_retrain = True

        if force_retrain or not os.path.exists(data_path):
            if emulator_data_mode == "local":
                print("Generando dataset local para emulador...")
                Xp, Xs, y, lens, max_len = generate_emulator_data_local(
                    data=data,
                    ids=ids,
                    theta_pop=theta_pop,
                    Omega=Omega,
                    sigma=sigma,
                    cfg=cfg,
                    n_virtual=n_virtual,
                    seed=seed
                )
            else:
                print("Generando dataset para emulador con LHS...")
                Xp, Xs, y, lens, max_len = generate_emulator_data(
                    data=data,
                    ids=ids,
                    cfg=cfg,
                    n_pop=n_pop,
                    n_ind_per_pop=n_ind_per_pop,
                    theta_center=theta_pop,
                    theta_sigma=theta_sigma,
                    omega_range=omega_range,
                    sigma_range=sigma_range,
                    dose_range=dose_range,
                    seed=seed
                )
            print(f"Dataset generado: {Xp.shape[0]} muestras, longitud máxima {max_len}")
            save_emulator_data(data_path, Xp, Xs, y, lens, max_len, theta_pop, Omega, sigma)
            print(f"Dataset guardado en {data_path}")

    # --- El resto del código (división, escalado, entrenamiento) permanece igual ---
    from sklearn.model_selection import train_test_split
    indices = np.arange(Xp.shape[0])
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, random_state=seed)

    Xp_train, Xp_val, Xp_test = Xp[train_idx], Xp[val_idx], Xp[test_idx]
    Xs_train, Xs_val, Xs_test = Xs[train_idx], Xs[val_idx], Xs[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    lens_train, lens_val, lens_test = lens[train_idx], lens[val_idx], lens[test_idx]

    # Escalar la salida
    scaler_y = fit_target_scaler(y_train, lens_train)
    y_train_scaled = scale_target_sequences(y_train, lens_train, scaler_y)
    y_val_scaled = scale_target_sequences(y_val, lens_val, scaler_y)
    y_test_scaled = scale_target_sequences(y_test, lens_test, scaler_y)
    sw_train = build_sequence_mask(lens_train, max_len)
    sw_val = build_sequence_mask(lens_val, max_len)
    sw_test = build_sequence_mask(lens_test, max_len)

    # Guardar el scaler
    scaler_path = _emulator_scaler_path(model_path)
    joblib.dump(scaler_y, scaler_path)
    print(f"Scaler guardado en {scaler_path}")

    # Construir modelo del emulador
    model = build_emulator(
        max_len,
        n_params=cfg.n_par,
        model_type=emulator_model_type,
        d_model=emulator_d_model,
        n_heads=emulator_n_heads,
        n_layers=emulator_n_layers,
        dropout=emulator_dropout
    )
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    print("Entrenando emulador...")
    history = model.fit(
        [Xs_train, Xp_train], y_train_scaled,
        sample_weight=sw_train,
        validation_data=([Xs_val, Xp_val], y_val_scaled, sw_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluación en test (opcional)
    y_pred_scaled = model.predict([Xs_test, Xp_test], verbose=0)
    y_pred = inverse_scale_target_sequences(y_pred_scaled, lens_test, scaler_y)
    y_true = y_test
    y_pred[sw_test == 0] = 0.0
    mae = masked_mae(y_true, y_pred, lens_test)
    print(f"MAE en test (escala original del DECK): {mae:.4f}")

    # Guardar modelo
    keras_path = _emulator_keras_path(model_path)
    model.save(keras_path)
    with open(_emulator_meta_path(model_path), 'w') as f:
        json.dump({'max_len': int(max_len), 'n_params': cfg.n_par}, f)
    print(f"Emulador guardado en {keras_path}")

    return model, max_len, scaler_y


def train_emulator(
    data, ids, theta_pop, Omega, sigma, cfg, n_virtual,
    model_path, data_path=None, seed=123, force_retrain=False,
    validation_split=0.15, epochs=30, batch_size=64,
    early_stopping_patience=8,
    n_pop=200, n_ind_per_pop=5, theta_sigma=0.5,
    omega_range=(0.01, 0.5), sigma_range=(0.01, 0.5),
    dose_range=(50, 200),
    virtual_data_path=None,
    emulator_data_mode="local",
    emulator_model_type="lstm",
    emulator_d_model=64,
    emulator_n_heads=4,
    emulator_n_layers=2,
    emulator_dropout=0.1,
    xgb_peak_emphasis=4.0,
    xgb_peak_window=1,
    xgb_dose_emphasis=1.5,
):
    """
    Entrena el emulador. Permite backends secuenciales (LSTM/transformer)
    y un backend pointwise con XGBoost para CPU.
    """
    emulator_data_mode = str(emulator_data_mode).lower()
    if emulator_data_mode not in {"lhs", "local"}:
        raise ValueError(f"Modo de dataset de emulador no soportado: {emulator_data_mode}")

    emulator_model_type = str(emulator_model_type).lower()
    if emulator_model_type not in {"transformer", "lstm", "xgb_pointwise", "xgb_residual_pointwise"}:
        raise ValueError(f"Tipo de emulador no soportado: {emulator_model_type}")

    print(
        "Configuracion emulador: "
        f"modo={emulator_data_mode}, n_virtual={n_virtual}, n_pop={n_pop}, "
        f"n_ind_per_pop={n_ind_per_pop}, epochs={epochs}, batch_size={batch_size}, "
        f"modelo={emulator_model_type}"
    )

    if data_path is None:
        data_path = _emulator_data_cache_path(model_path, emulator_data_mode)

    if virtual_data_path is not None:
        if not os.path.exists(virtual_data_path):
            raise FileNotFoundError(f"Archivo de dataset virtual no encontrado: {virtual_data_path}")
        print(f"Cargando dataset virtual desde {virtual_data_path}")
        loaded = np.load(virtual_data_path, allow_pickle=True)
        data_version = int(np.asarray(loaded.get("data_version", np.array([1]))).ravel()[0])
        if data_version != EMULATOR_DATA_VERSION:
            raise ValueError(
                f"Dataset virtual incompatible (version {data_version}). Regenera el archivo con el codigo actual."
            )
        Xp = loaded["Xp"]
        Xs = loaded["Xs"]
        y = loaded["y"]
        lens = loaded["lens"]
        max_len = int(loaded["max_len"])
    else:
        if os.path.exists(data_path) and not force_retrain:
            print(f"Cargando dataset desde {data_path}")
            loaded = load_emulator_data(data_path, theta_pop, Omega, sigma)
            if loaded is not None:
                Xp, Xs, y, lens, max_len = loaded
            else:
                force_retrain = True

        if force_retrain or not os.path.exists(data_path):
            if emulator_data_mode == "local":
                print("Generando dataset local para emulador...")
                Xp, Xs, y, lens, max_len = generate_emulator_data_local(
                    data=data,
                    ids=ids,
                    theta_pop=theta_pop,
                    Omega=Omega,
                    sigma=sigma,
                    cfg=cfg,
                    n_virtual=n_virtual,
                    seed=seed
                )
            else:
                print("Generando dataset para emulador con LHS...")
                Xp, Xs, y, lens, max_len = generate_emulator_data(
                    data=data,
                    ids=ids,
                    cfg=cfg,
                    n_pop=n_pop,
                    n_ind_per_pop=n_ind_per_pop,
                    theta_center=theta_pop,
                    theta_sigma=theta_sigma,
                    omega_range=omega_range,
                    sigma_range=sigma_range,
                    dose_range=dose_range,
                    seed=seed
                )
            print(f"Dataset generado: {Xp.shape[0]} muestras, longitud maxima {max_len}")
            save_emulator_data(data_path, Xp, Xs, y, lens, max_len, theta_pop, Omega, sigma)
            print(f"Dataset guardado en {data_path}")

    indices = np.arange(Xp.shape[0])
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, random_state=seed)

    Xp_train, Xp_val, Xp_test = Xp[train_idx], Xp[val_idx], Xp[test_idx]
    Xs_train, Xs_val, Xs_test = Xs[train_idx], Xs[val_idx], Xs[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    lens_train, lens_val, lens_test = lens[train_idx], lens[val_idx], lens[test_idx]

    meta = {
        "max_len": int(max_len),
        "n_params": int(cfg.n_par),
        "backend": emulator_model_type,
    }

    if emulator_model_type in {"xgb_pointwise", "xgb_residual_pointwise"}:
        residual_mode = emulator_model_type == "xgb_residual_pointwise"
        baseline_feature = residual_mode
        baseline_cache = {}
        if residual_mode:
            baseline_cache = build_xgb_baseline_cache(
                data,
                ids,
                theta_center=theta_pop,
                Xs=Xs,
                lens=lens,
            )
            print(f"Cache de baseline residual: {len(baseline_cache)} calendarios unicos")

        X_train_rows, y_train_rows = flatten_xgb_pointwise_dataset(
            Xp_train,
            Xs_train,
            y_train,
            lens_train,
            baseline_cache=baseline_cache,
            residual_mode=residual_mode,
            baseline_feature=baseline_feature,
        )
        X_val_rows, y_val_rows = flatten_xgb_pointwise_dataset(
            Xp_val,
            Xs_val,
            y_val,
            lens_val,
            baseline_cache=baseline_cache,
            residual_mode=residual_mode,
            baseline_feature=baseline_feature,
        )
        X_test_rows, y_test_rows = flatten_xgb_pointwise_dataset(
            Xp_test,
            Xs_test,
            y_test,
            lens_test,
            baseline_cache=baseline_cache,
            residual_mode=residual_mode,
            baseline_feature=baseline_feature,
        )
        sample_weight_train = build_xgb_pointwise_sample_weights(
            Xs_train,
            y_train,
            lens_train,
            peak_emphasis=xgb_peak_emphasis,
            peak_window=xgb_peak_window,
            dose_emphasis=xgb_dose_emphasis,
        )
        sample_weight_val = build_xgb_pointwise_sample_weights(
            Xs_val,
            y_val,
            lens_val,
            peak_emphasis=xgb_peak_emphasis,
            peak_window=xgb_peak_window,
            dose_emphasis=xgb_dose_emphasis,
        )
        target_transform = "signed_log1p"
        y_train_rows_t = _transform_emulator_targets(y_train_rows, target_transform)
        y_val_rows_t = _transform_emulator_targets(y_val_rows, target_transform)

        model = xgb.XGBRegressor(
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            early_stopping_rounds=max(20, early_stopping_patience * 5),
            eval_metric="mae",
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )

        print(
            "Entrenando emulador XGBoost pointwise..."
            f" peak_emphasis={xgb_peak_emphasis}, peak_window={xgb_peak_window},"
            f" dose_emphasis={xgb_dose_emphasis}"
        )
        model.fit(
            X_train_rows,
            y_train_rows_t,
            sample_weight=sample_weight_train,
            eval_set=[(X_val_rows, y_val_rows_t)],
            sample_weight_eval_set=[sample_weight_val],
            verbose=False,
        )

        y_pred_rows_t = model.predict(X_test_rows)
        y_pred_rows = np.sign(y_pred_rows_t) * np.expm1(np.abs(y_pred_rows_t))
        mae = mean_absolute_error(y_test_rows, y_pred_rows)
        print(f"MAE en test (escala original del DECK): {mae:.4f}")

        joblib_path = _emulator_joblib_path(model_path)
        scaler_path = _emulator_scaler_path(model_path)
        joblib.dump(model, joblib_path)
        joblib.dump(None, scaler_path)
        meta["target_transform"] = target_transform
        meta["residual_mode"] = residual_mode
        meta["baseline_feature"] = baseline_feature
        meta["unknown_schedule_policy"] = "fallback_real" if residual_mode else "zero_baseline"
        meta["xgb_peak_emphasis"] = float(xgb_peak_emphasis)
        meta["xgb_peak_window"] = int(xgb_peak_window)
        meta["xgb_dose_emphasis"] = float(xgb_dose_emphasis)
        baseline_cache_path = _emulator_baseline_cache_path(model_path)
        if residual_mode:
            joblib.dump(baseline_cache, baseline_cache_path)
        with open(_emulator_meta_path(model_path), "w") as f:
            json.dump(meta, f)
        print(f"Emulador guardado en {joblib_path}")
        return XGBPointwiseEmulator(
            model,
            target_transform=target_transform,
            residual_mode=residual_mode,
            baseline_feature=baseline_feature,
            baseline_cache=baseline_cache,
        ), max_len, None

    scaler_y = fit_target_scaler(y_train, lens_train)
    y_train_scaled = scale_target_sequences(y_train, lens_train, scaler_y)
    y_val_scaled = scale_target_sequences(y_val, lens_val, scaler_y)
    y_test_scaled = scale_target_sequences(y_test, lens_test, scaler_y)
    sw_train = build_sequence_mask(lens_train, max_len)
    sw_val = build_sequence_mask(lens_val, max_len)
    sw_test = build_sequence_mask(lens_test, max_len)

    scaler_path = _emulator_scaler_path(model_path)
    joblib.dump(scaler_y, scaler_path)
    print(f"Scaler guardado en {scaler_path}")

    model = build_emulator(
        max_len,
        n_params=cfg.n_par,
        model_type=emulator_model_type,
        d_model=emulator_d_model,
        n_heads=emulator_n_heads,
        n_layers=emulator_n_layers,
        dropout=emulator_dropout
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    print("Entrenando emulador...")
    model.fit(
        [Xs_train, Xp_train], y_train_scaled,
        sample_weight=sw_train,
        validation_data=([Xs_val, Xp_val], y_val_scaled, sw_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    y_pred_scaled = model.predict([Xs_test, Xp_test], verbose=0)
    y_pred = inverse_scale_target_sequences(y_pred_scaled, lens_test, scaler_y)
    y_pred[sw_test == 0] = 0.0
    mae = masked_mae(y_test, y_pred, lens_test)
    print(f"MAE en test (escala original del DECK): {mae:.4f}")

    keras_path = _emulator_keras_path(model_path)
    model.save(keras_path)
    with open(_emulator_meta_path(model_path), "w") as f:
        json.dump(meta, f)
    print(f"Emulador guardado en {keras_path}")

    return model, max_len, scaler_y


def validate_emulator(emulator, max_len, scaler_y, data, ids, theta, sigma, cfg, n_val=20):
    """
    Validate emulator against real deck for a sample of individuals.
    Returns mean relative error for concentrations, AUC and Cmax.
    """
    from deck_interface import original_run_deck
    rng = np.random.default_rng(123)
    ids_val = rng.choice(ids, size=min(n_val, len(ids)), replace=False)
    
    errors = []
    auc_errors = []
    cmax_errors = []
    
    for i_id in ids_val:
        data_ind = data[data["ID"] == i_id].copy()
        # Use population theta (or could use random ETAs, but here just theta)
        pars = theta
        # Real deck
        pred_real = original_run_deck(pars, data_ind)
        # Emulator
        pred_emu = run_deck_emulated(pars, data_ind, emulator, max_len, scaler_y)
        
        if "Meas" in data_ind.columns:
            mask = data_ind["Meas"].to_numpy() != 0
        else:
            mask = np.ones(len(data_ind), dtype=bool)
        
        if mask.sum() > 0:
            pred_real_pk = _to_pk_profile(pred_real[mask], cfg)
            pred_emu_pk = _to_pk_profile(pred_emu[mask], cfg)
            rel_error = np.mean(np.abs((pred_real_pk - pred_emu_pk) / (np.abs(pred_real_pk) + 1e-8)))
            errors.append(rel_error)
            
            # AUC (trapezoidal)
            t = data_ind.loc[mask, "TIME"].values
            auc_real = np.trapezoid(pred_real_pk, t)
            auc_emu = np.trapezoid(pred_emu_pk, t)
            auc_errors.append(abs(auc_real - auc_emu) / (abs(auc_real) + 1e-8))
            
            # Cmax
            cmax_real = np.max(pred_real_pk)
            cmax_emu = np.max(pred_emu_pk)
            cmax_errors.append(abs(cmax_real - cmax_emu) / (abs(cmax_real) + 1e-8))
    
    return {
        "mean_rel_error": np.mean(errors) if errors else np.nan,
        "mean_auc_error": np.mean(auc_errors) if auc_errors else np.nan,
        "mean_cmax_error": np.mean(cmax_errors) if cmax_errors else np.nan,
        "n_val": len(errors)
    }


def _output_is_log_scale(cfg, values):
    obs_column = str(getattr(cfg, "obs_column", "")).upper()
    if getattr(cfg, "transform", "identity") == "log":
        return False
    if "LN" in obs_column or "LOG" in obs_column:
        return True
    values = np.asarray(values, dtype=float)
    return np.any(values <= 0.0)


def _terminal_log_profile(values, cfg):
    values = np.asarray(values, dtype=float)
    if _output_is_log_scale(cfg, values):
        return values
    if np.any(values <= 0.0):
        return None
    return np.log(np.clip(values, 1e-12, None))


def _to_pk_profile(values, cfg):
    values = np.asarray(values, dtype=float)
    if _output_is_log_scale(cfg, values):
        return np.exp(np.clip(values, -60.0, 60.0))
    return np.clip(values, 0.0, None)


def _nan_summary(values, reducer):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan
    return float(reducer(clean))


def validate_emulator_with_etas(
    emulator,
    max_len,
    scaler_y,
    data,
    ids,
    theta,
    Omega,
    sigma,
    cfg,
    n_etas_per_id=5,
    n_val=20,
    deck_executor=None
):
    """
    Valida el emulador frente al DECK real usando varias ETAs muestreadas por individuo.
    Devuelve errores para perfil, AUC, Cmax, Tmax y pendiente terminal.
    """
    from deck_interface import deck_executor as global_deck_executor

    executor = deck_executor if deck_executor is not None else global_deck_executor
    rng = np.random.default_rng(123)
    ids = np.asarray(ids)
    ids_val = rng.choice(ids, size=min(int(n_val), len(ids)), replace=False)

    errors = []
    auc_errors = []
    cmax_errors = []
    tmax_errors = []
    slope_errors = []

    theta = np.asarray(theta, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    for i_id in ids_val:
        data_ind = data[data["ID"] == i_id].copy()
        for _ in range(int(n_etas_per_id)):
            eta = rng.multivariate_normal(np.zeros(len(theta)), Omega)
            pars = individual_params(theta, eta, cfg.re_type)
            pred_real = executor.run_real(pars, data_ind)
            pred_emu = run_deck_emulated(pars, data_ind, emulator, max_len, scaler_y)

            if "Meas" in data_ind.columns:
                mask = data_ind["Meas"].to_numpy() != 0
            else:
                mask = np.ones(len(data_ind), dtype=bool)

            if mask.sum() == 0:
                continue

            pred_real_masked = np.asarray(pred_real[mask], dtype=float)
            pred_emu_masked = np.asarray(pred_emu[mask], dtype=float)
            pred_real_pk = _to_pk_profile(pred_real_masked, cfg)
            pred_emu_pk = _to_pk_profile(pred_emu_masked, cfg)
            t = data_ind.loc[mask, "TIME"].to_numpy(dtype=float)

            rel_error = np.mean(
                np.abs((pred_real_pk - pred_emu_pk) / (np.abs(pred_real_pk) + 1e-8))
            )
            errors.append(float(rel_error))

            auc_real = np.trapezoid(pred_real_pk, t)
            auc_emu = np.trapezoid(pred_emu_pk, t)
            auc_errors.append(float(abs(auc_real - auc_emu) / (abs(auc_real) + 1e-8)))

            cmax_real = np.max(pred_real_pk)
            cmax_emu = np.max(pred_emu_pk)
            cmax_errors.append(float(abs(cmax_real - cmax_emu) / (abs(cmax_real) + 1e-8)))

            idx_max_real = int(np.argmax(pred_real_masked))
            idx_max_emu = int(np.argmax(pred_emu_masked))
            tmax_real = float(t[idx_max_real])
            tmax_emu = float(t[idx_max_emu])
            tmax_errors.append(abs(tmax_real - tmax_emu))

            mask_terminal = t > tmax_real
            if np.sum(mask_terminal) >= 2:
                t_term = t[mask_terminal]
                prof_real = _terminal_log_profile(pred_real_masked[mask_terminal], cfg)
                prof_emu = _terminal_log_profile(pred_emu_masked[mask_terminal], cfg)
                if prof_real is not None and prof_emu is not None:
                    coef_real = np.polyfit(t_term, prof_real, 1)
                    coef_emu = np.polyfit(t_term, prof_emu, 1)
                    lambda_z_real = -float(coef_real[0])
                    lambda_z_emu = -float(coef_emu[0])
                    slope_errors.append(abs(lambda_z_real - lambda_z_emu) / (abs(lambda_z_real) + 1e-8))

    return {
        "mean_rel_error": _nan_summary(errors, np.mean),
        "median_rel_error": _nan_summary(errors, np.median),
        "p90_rel_error": _nan_summary(errors, lambda x: np.percentile(x, 90)),
        "max_rel_error": _nan_summary(errors, np.max),
        "mean_auc_error": _nan_summary(auc_errors, np.mean),
        "mean_cmax_error": _nan_summary(cmax_errors, np.mean),
        "mean_tmax_abs_error": _nan_summary(tmax_errors, np.mean),
        "mean_terminal_slope_error": _nan_summary(slope_errors, np.mean),
        "n_profiles": int(len(errors)),
        "n_ids": int(len(ids_val))
    }

def build_emulator(seq_len, n_params=4, model_type='transformer', d_model=128, n_heads=8, n_layers=4, dropout=0.1):
    """Construye el modelo emulador (transformer por defecto)."""
    from tensorflow.keras import layers, models

    seq_input = layers.Input(shape=(seq_len, 3), name='sequence')
    param_input = layers.Input(shape=(n_params,), name='params')

    if model_type == 'transformer':
        x = layers.Dense(d_model)(seq_input)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=d_model)(positions)
        x = x + pos_embedding
        x = layers.Dropout(dropout)(x)
        key_dim = max(8, d_model // max(1, n_heads))

        for _ in range(n_layers):
            attn_out = layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim)(x, x)
            attn_out = layers.Dropout(dropout)(attn_out)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

            ff = layers.Dense(d_model * 4, activation='relu')(x)
            ff = layers.Dropout(dropout)(ff)
            ff = layers.Dense(d_model)(ff)
            ff = layers.Dropout(dropout)(ff)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

        params_repeated = layers.RepeatVector(seq_len)(param_input)
        concat = layers.Concatenate()([x, params_repeated])
        y = layers.TimeDistributed(layers.Dense(64, activation='relu'))(concat)
        y = layers.TimeDistributed(layers.Dense(1))(y)
    elif model_type == 'lstm':
        # Si quieres mantener la opción LSTM
        x = layers.Masking(mask_value=0.0)(seq_input)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(32, return_sequences=True)(x)
        params_repeated = layers.RepeatVector(seq_len)(param_input)
        concat = layers.Concatenate()([x, params_repeated])
        y = layers.TimeDistributed(layers.Dense(16, activation='relu'))(concat)
        y = layers.TimeDistributed(layers.Dense(1))(y)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = models.Model(inputs=[seq_input, param_input], outputs=y)
    model.compile(optimizer='adam', loss='mse')
    return model
