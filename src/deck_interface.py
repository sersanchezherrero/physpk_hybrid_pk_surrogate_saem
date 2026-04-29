# deck_interface.py
import importlib
import hashlib
import json
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

MAX_DECK_SAMPLES = 50
MAX_DECK_PARAMS = 10


def _emulator_artifact_stem(model_path):
    for extension in (".keras", ".h5", ".joblib"):
        if model_path.endswith(extension):
            return model_path[: -len(extension)]
    return model_path


def build_schedule_signature_from_arrays(time, dose, duration):
    schedule = np.column_stack([
        np.asarray(time, dtype=np.float64).reshape(-1),
        np.asarray(dose, dtype=np.float64).reshape(-1),
        np.asarray(duration, dtype=np.float64).reshape(-1),
    ])
    return hashlib.md5(schedule.tobytes()).hexdigest()


def build_schedule_signature(data_ind):
    return build_schedule_signature_from_arrays(
        data_ind["TIME"].to_numpy(dtype=np.float64),
        data_ind["Dose"].to_numpy(dtype=np.float64),
        data_ind["Duration"].to_numpy(dtype=np.float64),
    )


def build_xgb_pointwise_features(params, time, dose, duration, baseline=None):
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    log_params = np.log(np.clip(params, 1e-12, None))
    time = np.asarray(time, dtype=np.float32).reshape(-1)
    dose = np.asarray(dose, dtype=np.float32).reshape(-1)
    duration = np.asarray(duration, dtype=np.float32).reshape(-1)

    rate = np.divide(
        dose,
        np.maximum(duration, 1e-6),
        out=np.zeros_like(dose),
        where=duration > 0
    )
    is_dose = (dose > 0).astype(np.float32)
    log_time = np.log1p(np.clip(time, 0.0, None))
    log_dose = np.log1p(np.clip(dose, 0.0, None))
    params_block = np.repeat(params.reshape(1, -1), time.shape[0], axis=0)
    log_params_block = np.repeat(log_params.reshape(1, -1), time.shape[0], axis=0)
    feature_blocks = [
        params_block,
        log_params_block,
        time,
        log_time,
        dose,
        log_dose,
        duration,
        rate,
        is_dose,
    ]
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=np.float32).reshape(-1)
        if baseline.shape[0] != time.shape[0]:
            raise ValueError("baseline debe tener la misma longitud que time")
        feature_blocks.append(baseline)
    return np.column_stack(feature_blocks).astype(np.float32)


def _inverse_target_transform(values, target_transform):
    target_transform = str(target_transform or "identity").lower()
    values = np.asarray(values, dtype=float)

    if target_transform == "identity":
        return values
    if target_transform == "signed_log1p":
        return np.sign(values) * np.expm1(np.abs(values))

    raise ValueError(f"Transformacion de target no soportada: {target_transform}")


class XGBPointwiseEmulator:
    def __init__(
        self,
        model,
        target_transform="identity",
        residual_mode=False,
        baseline_feature=False,
        baseline_cache=None,
        unknown_schedule_policy=None,
        backend_name=None,
    ):
        self.model = model
        self.target_transform = str(target_transform or "identity").lower()
        self.residual_mode = bool(residual_mode)
        self.baseline_feature = bool(baseline_feature)
        self.baseline_cache = dict(baseline_cache or {})
        self.unknown_schedule_policy = str(
            unknown_schedule_policy
            or ("fallback_real" if (self.residual_mode or self.baseline_feature) else "zero_baseline")
        ).lower()
        self.backend_name = str(
            backend_name
            or ("xgb_residual_pointwise" if self.residual_mode else "xgb_pointwise")
        )
        self._warned_missing_signatures = set()

    def _missing_baseline(self, data_ind, reason):
        signature = build_schedule_signature(data_ind)
        if self.unknown_schedule_policy == "fallback_real":
            if signature not in self._warned_missing_signatures:
                logger.warning(
                    "Schedule fuera del dominio del emulador (%s, backend=%s, signature=%s); "
                    "se usa fallback al DECK real para esta prediccion.",
                    reason,
                    self.backend_name,
                    signature[:8],
                )
                self._warned_missing_signatures.add(signature)
            return None
        if signature not in self._warned_missing_signatures:
            logger.warning(
                "Schedule fuera del dominio del emulador (%s, backend=%s, signature=%s); "
                "se usa baseline nulo.",
                reason,
                self.backend_name,
                signature[:8],
            )
            self._warned_missing_signatures.add(signature)
        return np.zeros(data_ind.shape[0], dtype=float)

    def _lookup_baseline(self, data_ind):
        if not self.baseline_cache:
            return self._missing_baseline(data_ind, reason="cache_empty")
        signature = build_schedule_signature(data_ind)
        baseline = self.baseline_cache.get(signature)
        if baseline is None:
            return self._missing_baseline(data_ind, reason="schedule_missing")
        baseline = np.asarray(baseline, dtype=float).reshape(-1)
        if baseline.shape[0] != data_ind.shape[0]:
            return self._missing_baseline(data_ind, reason="baseline_length_mismatch")
        return baseline

    def predict_sequence(self, params, data_ind):
        baseline = None
        if self.residual_mode or self.baseline_feature:
            baseline = self._lookup_baseline(data_ind)
            if baseline is None and self.unknown_schedule_policy == "fallback_real":
                return np.asarray(original_run_deck(params, data_ind), dtype=float)
        features = build_xgb_pointwise_features(
            params,
            data_ind["TIME"].to_numpy(dtype=np.float32),
            data_ind["Dose"].to_numpy(dtype=np.float32),
            data_ind["Duration"].to_numpy(dtype=np.float32),
            baseline=baseline if self.baseline_feature else None,
        )
        pred = np.asarray(self.model.predict(features), dtype=float)
        pred = _inverse_target_transform(pred, self.target_transform)
        if self.residual_mode:
            pred = pred + np.asarray(baseline, dtype=float)
        return pred
# ----------------------------------------------------------------------
# Cargar el DECK real (EcosimPro)
# ----------------------------------------------------------------------
# Ajusta la ruta y el nombre de la biblioteca segun tu sistema.
# En la copia publica del repositorio se admite un backend local no versionado
# en vendor/physpk_backend para no mezclar artefactos propietarios con Git.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_MODULE_DIR, "vendor", "physpk_backend")
for _path in (_MODULE_DIR, _VENDOR_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)
try:
    from deck_python_3_11 import *
    deck = None
    if deck is not None and not deck.ok:
        raise RuntimeError("Error loading deck.")
except Exception as e:
    print(f"⚠️ No se pudo cargar el DECK real: {e}")
    deck = None


def _load_real_deck():
    """Carga el deck real solo cuando realmente se necesita."""
    global deck

    if deck is not None:
        return deck

    try:
        DeckPython
    except NameError as exc:
        raise RuntimeError(
            "DECK real no disponible. Coloca la exportacion local de PhysPK "
            "junto a deck_interface.py o en vendor/physpk_backend."
        ) from exc

    explicit_stem = os.environ.get("PHYSPK_DECK_STEM", "").strip()
    candidate_stems = []
    if explicit_stem:
        if os.path.isabs(explicit_stem):
            candidate_stems.append(explicit_stem)
        else:
            candidate_stems.append(os.path.join(_MODULE_DIR, explicit_stem))
    candidate_stems.extend([
        os.path.join(_MODULE_DIR, "dkd_win64_gcc64v11_2"),
        os.path.join(_VENDOR_DIR, "dkd_win64_gcc64v11_2"),
    ])

    last_error = None
    for stem in dict.fromkeys(candidate_stems):
        try:
            deck = DeckPython(stem, False)
        except Exception as exc:  # pragma: no cover - depende del backend local
            last_error = exc
            continue
        if deck.ok:
            logger.info("DECK real cargado correctamente desde %s", stem)
            return deck
        last_error = RuntimeError(f"Error loading deck from {stem}.")
        deck = None

    raise RuntimeError(
        "No se pudo cargar el backend DECK real. "
        "Revisa PHYSPK_DECK_STEM o coloca la exportacion local en "
        "vendor/physpk_backend."
    ) from last_error

# ----------------------------------------------------------------------
# Variables globales para el emulador
# ----------------------------------------------------------------------
_emulator = None
_max_len = None
_use_emulator_flag = False

# ----------------------------------------------------------------------
# Función original que ejecuta el DECK real
# ----------------------------------------------------------------------
def original_run_deck(params, data_ind):
    """
    Ejecuta el modelo PK real en EcosimPro para un individuo.
    params : array de parámetros (CL, V1, V2, Q)
    data_ind : DataFrame con columnas TIME, Dose, Duration (y opcionalmente Meas)
    Retorna predicciones para todas las filas (en escala original).
    """
    deck_obj = _load_real_deck()

    params = np.asarray(params, dtype=float)
    num_samples = data_ind.shape[0]
    n_params = params.size

    if n_params > MAX_DECK_PARAMS:
        raise ValueError(
            f"n_params={n_params} supera el maximo soportado por el DECK ({MAX_DECK_PARAMS}) "
            "para parameter[]. Aumenta el tamano del array en el DECK o reduce el numero de parametros."
        )
    if num_samples > MAX_DECK_SAMPLES:
        raise ValueError(
            f"num_samples={num_samples} supera el maximo soportado por el DECK ({MAX_DECK_SAMPLES}) "
            "para timeSamples/inputValues/inputDuration/concentrationSamples[]. "
            "Aumenta el tamano de los arrays en el DECK o reduce el numero de muestras."
        )

    # Fijar parámetros estructurales
    for k, val in enumerate(params, start=1):
        deck_obj.setD(f"parameter[{k:02d}]", float(val))

    # Fijar tiempos y entradas
    deck_obj.setI("timeSamplesCount", int(num_samples))
    for i in range(num_samples):
        deck_obj.setD(f"timeSamples[{i+1:02d}]", float(data_ind["TIME"].iloc[i]))
        deck_obj.setD(f"inputValues[{i+1:02d}]", float(data_ind["Dose"].iloc[i]))
        deck_obj.setD(f"inputDuration[{i+1:02d}]", float(data_ind["Duration"].iloc[i]))

    # Ejecutar
    deck_obj.run()

    # Leer concentraciones
    pred = np.zeros(num_samples, dtype=float)
    for i in range(num_samples):
        rc, value = deck_obj.getD(f"concentrationSamples[{i+1:02d}]")
        pred[i] = float(value)

    return pred

# ----------------------------------------------------------------------
# Función que usa el emulador (LSTM)
# ----------------------------------------------------------------------
def run_deck_emulated(params, data_ind, emulator, max_len, scaler_y):
    if hasattr(emulator, "predict_sequence"):
        params_flat = np.asarray(params, dtype=np.float32).reshape(-1)
        return np.asarray(emulator.predict_sequence(params_flat, data_ind), dtype=float)

    params = np.asarray(params, dtype=np.float32).reshape(1, -1)
    n_rows = data_ind.shape[0]

    seq = np.zeros((1, max_len, 3), dtype=np.float32)
    seq[0, :n_rows, 0] = data_ind["TIME"].to_numpy(dtype=np.float32)
    seq[0, :n_rows, 1] = data_ind["Dose"].to_numpy(dtype=np.float32)
    seq[0, :n_rows, 2] = data_ind["Duration"].to_numpy(dtype=np.float32)

    pred_padded = emulator([seq, params], training=False).numpy()
    pred = pred_padded[0, :n_rows, 0]
    if scaler_y is not None:
        pred = pred * float(scaler_y.scale_[0]) + float(scaler_y.mean_[0])
    return pred

# ----------------------------------------------------------------------
# Función activa (por defecto apunta al DECK real)
# ----------------------------------------------------------------------
def run_deck(params, data_ind):
    if _use_emulator_flag:
        return run_deck_emulated(params, data_ind, _emulator, _max_len, _scaler_y)
    else:
        return original_run_deck(params, data_ind)

# ----------------------------------------------------------------------
# Control del emulador
# ----------------------------------------------------------------------
_scaler_y = None

def use_emulator(emulator, max_len, scaler_y=None):
    global _emulator, _max_len, _use_emulator_flag, _scaler_y
    _emulator = emulator
    _max_len = max_len
    _scaler_y = scaler_y
    _use_emulator_flag = True
    logger.info("Modo emulador activado.")

def disable_emulator():
    global _use_emulator_flag
    _use_emulator_flag = False
    logger.info("Modo DECK real restaurado.")

# ----------------------------------------------------------------------
# Cargar emulador guardado (modelo + metadatos)
# ----------------------------------------------------------------------
def _legacy_load_emulator(model_path, n_params):
    import tensorflow as tf
    import os
    import json

    # Intentar con extensión .keras
    keras_path = model_path.replace('.h5', '.keras')
    if os.path.exists(keras_path):
        emulator = tf.keras.models.load_model(keras_path, compile=False)
        meta_path = keras_path.replace('.keras', '_meta.json')
    else:
        # Fallback a .h5
        emulator = tf.keras.models.load_model(model_path, compile=False)
        meta_path = model_path.replace('.h5', '_meta.json')

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        max_len = meta['max_len']
    else:
        max_len = emulator.input[0].shape[1]
    return emulator, max_len


def load_emulator(model_path, n_params):
    stem = _emulator_artifact_stem(model_path)
    meta_path = f"{stem}_meta.json"
    meta = {}

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    backend = str(meta.get("backend", "")).lower()
    joblib_path = f"{stem}.joblib"
    if backend in {"xgb_pointwise", "xgb_residual_pointwise"} or (
        not backend and os.path.exists(joblib_path) and not os.path.exists(f"{stem}.keras") and not os.path.exists(f"{stem}.h5")
    ):
        import joblib
        if not os.path.exists(joblib_path):
            raise FileNotFoundError(f"No se encontro el artefacto del emulador XGBoost: {joblib_path}")
        baseline_cache_path = f"{stem}_baseline_cache.joblib"
        baseline_cache = {}
        if os.path.exists(baseline_cache_path):
            baseline_cache = joblib.load(baseline_cache_path)
        emulator = XGBPointwiseEmulator(
            joblib.load(joblib_path),
            target_transform=meta.get("target_transform", "identity"),
            residual_mode=meta.get("residual_mode", backend == "xgb_residual_pointwise"),
            baseline_feature=meta.get("baseline_feature", backend == "xgb_residual_pointwise"),
            baseline_cache=baseline_cache,
            unknown_schedule_policy=meta.get("unknown_schedule_policy"),
            backend_name=backend or "xgb_pointwise",
        )
        max_len = int(meta.get("max_len", 0))
        logger.info(
            "Emulador cargado: backend=%s, residual=%s, baseline_feature=%s, schedule_policy=%s",
            emulator.backend_name,
            emulator.residual_mode,
            emulator.baseline_feature,
            emulator.unknown_schedule_policy,
        )
        return emulator, max_len

    import tensorflow as tf

    keras_path = f"{stem}.keras"
    h5_path = f"{stem}.h5"
    if os.path.exists(keras_path):
        emulator = tf.keras.models.load_model(keras_path, compile=False)
    elif os.path.exists(h5_path):
        emulator = tf.keras.models.load_model(h5_path, compile=False)
    else:
        raise FileNotFoundError(f"No se encontro ningun artefacto del emulador para {stem}")

    if meta:
        max_len = int(meta["max_len"])
    else:
        max_len = int(emulator.input[0].shape[1])
    try:
        setattr(emulator, "backend_name", backend or "keras")
    except Exception:
        pass
    logger.info("Emulador cargado: backend=%s", getattr(emulator, "backend_name", backend or "keras"))
    return emulator, max_len

# ----------------------------------------------------------------------
# Funciones auxiliares (para compatibilidad con scripts antiguos)
# ----------------------------------------------------------------------
def set_deck(*args, **kwargs):
    """Placeholder para compatibilidad."""
    pass


class DeckExecutor:
    """Ejecuta el modelo PK, ya sea real o emulado."""
    def __init__(self):
        self.use_real_deck = True   # por defecto usamos el real si está disponible
        self.emulator = None
        self.max_len = None
        self.scaler_y = None

    def use_emulator(self, emulator, max_len, scaler_y):
        self.emulator = emulator
        self.max_len = max_len
        self.scaler_y = scaler_y
        self.use_real_deck = False
        logger.info("Modo emulador activado.")

    def disable_emulator(self):
        self.use_real_deck = True
        self.emulator = None
        logger.info("Modo DECK real restaurado.")

    def run(self, params, data_ind):
        if self.use_real_deck:
            return original_run_deck(params, data_ind)
        return run_deck_emulated(params, data_ind, self.emulator, self.max_len, self.scaler_y)

    def run_real(self, params, data_ind):
        """Ejecuta siempre el DECK real, aunque el emulador esté activo."""
        return original_run_deck(params, data_ind)

# Instancia global para mantener compatibilidad con código existente (opcional)
deck_executor = DeckExecutor()

# Funciones auxiliares (opcionales, pero podemos mantenerlas por ahora)
def use_emulator(emulator, max_len, scaler_y):
    deck_executor.use_emulator(emulator, max_len, scaler_y)

def disable_emulator():
    deck_executor.disable_emulator()

def run_deck(params, data_ind):
    return deck_executor.run(params, data_ind)
