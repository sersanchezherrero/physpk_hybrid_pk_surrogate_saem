import numpy as np
import pandas as pd

from deck_interface import original_run_deck
from models import individual_params, residual_sd as residual_model


def _measurement_mask(data_ind, meas_col="Meas"):
    if meas_col in data_ind.columns:
        return data_ind[meas_col].to_numpy() != 0
    return np.ones(data_ind.shape[0], dtype=bool)


def parse_sparse_windows(spec):
    """
    Convierte una cadena tipo ``"0-1,1-24,24-120"`` en una lista de ventanas.

    Se espera una sintaxis ``inicio-fin`` separada por comas. Los limites se
    interpretan en horas.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None

    windows = []
    for chunk in spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" not in piece:
            raise ValueError(
                f"Ventana sparse invalida '{piece}'. Usa el formato inicio-fin, por ejemplo 0-24."
            )
        lower_txt, upper_txt = piece.split("-", 1)
        lower = float(lower_txt.strip())
        upper = float(upper_txt.strip())
        if upper <= lower:
            raise ValueError(
                f"Ventana sparse invalida '{piece}': el limite superior debe ser mayor que el inferior."
            )
        windows.append((lower, upper))

    if not windows:
        raise ValueError("No se ha podido interpretar ninguna ventana sparse.")
    return windows


def _window_mask(times, lower, upper, include_upper=False):
    if include_upper:
        return (times >= lower) & (times <= upper)
    return (times >= lower) & (times < upper)


def apply_sparse_sampling_windows(
    data,
    windows,
    points_per_window=1,
    seed=123,
    id_col="ID",
    time_col="TIME",
    meas_col="Meas",
):
    """
    Aplica un dise\u00f1o sparse seleccionando aleatoriamente puntos por ventanas.

    - Conserva todas las filas del schedule original.
    - Desactiva observaciones no seleccionadas poniendo ``Meas=0``.
    - La selecci\u00f3n se hace por individuo y por ventana temporal.
    """
    windows = list(windows or [])
    if not windows:
        return data.copy()

    points_per_window = int(max(1, points_per_window))
    rng = np.random.default_rng(seed)

    out = data.copy()
    original_mask = _measurement_mask(out, meas_col=meas_col)
    if meas_col not in out.columns:
        out[meas_col] = original_mask.astype(int)

    out["SPARSE_SELECTED"] = 0
    out["SPARSE_WINDOW"] = ""

    # Reinicia las observaciones medidas y vuelve a activarlas solo si se seleccionan.
    out.loc[original_mask, meas_col] = 0

    for i_id in out[id_col].dropna().unique():
        idx_id = out.index[out[id_col] == i_id]
        if len(idx_id) == 0:
            continue

        measured_idx = idx_id[original_mask[idx_id]]
        if len(measured_idx) == 0:
            continue

        times = out.loc[measured_idx, time_col].to_numpy(dtype=float)
        already_selected = set()

        for window_idx, (lower, upper) in enumerate(windows):
            include_upper = window_idx == (len(windows) - 1)
            in_window = _window_mask(times, lower, upper, include_upper=include_upper)
            candidate_idx = measured_idx[in_window]
            candidate_idx = [idx for idx in candidate_idx if idx not in already_selected]
            if not candidate_idx:
                continue

            n_select = min(points_per_window, len(candidate_idx))
            chosen = rng.choice(np.asarray(candidate_idx, dtype=int), size=n_select, replace=False)
            label = f"{lower:g}-{upper:g}"
            for idx in np.asarray(chosen, dtype=int).tolist():
                already_selected.add(int(idx))
                out.at[idx, meas_col] = 1
                out.at[idx, "SPARSE_SELECTED"] = 1
                out.at[idx, "SPARSE_WINDOW"] = label

    return out


def _select_source_ids(ids, n_virtual, rng, balanced=False):
    ids = list(ids)
    if not ids:
        raise ValueError("No hay IDs disponibles para bootstrap.")
    n_virtual = int(max(1, n_virtual))

    if not balanced:
        indices = rng.choice(len(ids), size=n_virtual, replace=True)
        return [ids[int(idx)] for idx in indices]

    full_repeats, remainder = divmod(n_virtual, len(ids))
    selected_ids = list(ids) * full_repeats
    if remainder:
        extra_indices = rng.choice(len(ids), size=remainder, replace=False)
        selected_ids.extend(ids[int(idx)] for idx in extra_indices)
    rng.shuffle(selected_ids)
    return selected_ids


def patient_bootstrap_population(data, ids=None, n_virtual=None, seed=123, balanced=False):
    """
    Re-muestrea pacientes completos con reemplazo y renumera los IDs.

    Este baseline preserva exactamente las observaciones originales y sirve
    como comparador estadistico simple frente a metodos model-informed.
    """
    if ids is None:
        ids = data["ID"].dropna().unique().tolist()
    if n_virtual is None:
        n_virtual = len(ids)

    rng = np.random.default_rng(seed)
    selected_ids = _select_source_ids(ids, n_virtual, rng, balanced=balanced)

    virtual_frames = []
    for new_id, source_id in enumerate(selected_ids, start=1):
        template = data[data["ID"] == source_id].copy()
        if template.empty:
            continue
        template["SOURCE_ID"] = source_id
        template["BOOTSTRAP_KIND"] = "patient"
        template["ID"] = new_id
        virtual_frames.append(template)

    if not virtual_frames:
        raise ValueError("No se pudieron generar pacientes bootstrap a partir de los IDs dados.")

    return pd.concat(virtual_frames, ignore_index=True)


def generate_parametric_bootstrap_individual(
    theta_pop,
    Omega,
    sigma,
    data_ind_template,
    cfg,
    rng,
    simulate_fn=None,
    include_residual_noise=True,
):
    """
    Simula un paciente virtual a partir del modelo NLME ajustado.

    - Reutiliza el calendario del paciente plantilla.
    - Muestrea eta ~ N(0, Omega).
    - Simula con el DECK real (o simulate_fn).
    - Opcionalmente añade error residual en la misma escala del pipeline.
    """
    n_par = cfg.n_par
    eta = rng.multivariate_normal(mean=np.zeros(n_par), cov=Omega).astype(float)
    pars_ind = individual_params(theta_pop, eta, cfg.re_type)

    if simulate_fn is None:
        simulate_fn = original_run_deck

    pred_all = np.asarray(simulate_fn(pars_ind, data_ind_template), dtype=float)
    mask = _measurement_mask(data_ind_template)
    pred = pred_all[mask].astype(float)

    if cfg.transform == "log":
        pred = np.clip(pred, 1e-12, np.inf)
        pred_t = np.log(pred)
    else:
        pred_t = pred

    if include_residual_noise:
        sd = residual_model(pred, sigma=sigma, err_model=cfg.err_model, transform=cfg.transform)
        eps = rng.normal(loc=0.0, scale=sd, size=pred_t.shape)
        y_t = pred_t + eps
    else:
        y_t = pred_t.copy()

    virt = data_ind_template.copy()
    obs_col = cfg.obs_column
    if obs_col not in virt.columns:
        virt[obs_col] = np.nan
    virt.loc[virt.index[mask], obs_col] = y_t
    virt["SIM_PRED"] = pred_all
    virt["BOOTSTRAP_KIND"] = "parametric"
    for idx, eta_value in enumerate(eta, start=1):
        virt[f"ETA_{idx:02d}"] = float(eta_value)

    return virt, eta, pars_ind


def generate_parametric_bootstrap_population(
    data,
    ids,
    theta_pop,
    Omega,
    sigma,
    cfg,
    n_virtual,
    seed=123,
    simulate_fn=None,
    include_residual_noise=True,
    balanced=False,
):
    """
    Genera una poblacion virtual por parametric bootstrap a nivel paciente.

    Cada individuo virtual:
    - re-muestrea un schedule real por paciente,
    - simula un nuevo perfil con eta muestreada,
    - y opcionalmente añade error residual.
    """
    rng = np.random.default_rng(seed)
    selected_ids = _select_source_ids(ids, n_virtual, rng, balanced=balanced)

    virtual_frames = []
    for new_id, source_id in enumerate(selected_ids, start=1):
        template = data[data["ID"] == source_id].copy()
        if template.empty:
            continue
        virt, _, pars_ind = generate_parametric_bootstrap_individual(
            theta_pop=theta_pop,
            Omega=Omega,
            sigma=sigma,
            data_ind_template=template,
            cfg=cfg,
            rng=rng,
            simulate_fn=simulate_fn,
            include_residual_noise=include_residual_noise,
        )
        virt["SOURCE_ID"] = source_id
        virt["ID"] = new_id
        for idx, par_value in enumerate(np.asarray(pars_ind, dtype=float), start=1):
            virt[f"PARAM_{idx:02d}"] = float(par_value)
        virtual_frames.append(virt)

    if not virtual_frames:
        raise ValueError("No se pudo generar la poblacion virtual bootstrap.")

    return pd.concat(virtual_frames, ignore_index=True)
