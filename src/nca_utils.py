# nca_utils.py
import numpy as np
import pandas as pd
import os

def load_nca_initial_values(nca_file, data, ids, cfg):
    """
    Lee el archivo de resultados NCA y calcula valores iniciales para theta, Omega y ETAs.
    
    Args:
        nca_file (str): ruta al archivo SolNCA.txt
        data (pd.DataFrame): datos originales (para obtener IDs)
        ids (list): lista de IDs
        cfg (dict): configuración del modelo (con n_par, obs_column, etc.)
    
    Returns:
        theta0 (np.array): valores iniciales para theta
        Omega0 (np.array): matriz de covarianza inicial
        sigma0 (np.array): parámetros de error iniciales (por defecto [0.1])
        eta_init_dict (dict): diccionario ID -> eta inicial
    """
    if not os.path.exists(nca_file):
        raise FileNotFoundError(f"Archivo NCA {nca_file} no encontrado.")
    
    # Leer el archivo buscando la línea de encabezado de la tabla
    with open(nca_file, 'r', encoding='latin1') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Individual") and "N_Lambda_Z_Samples" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No se encontró la tabla de resultados NCA en el archivo.")
    
    # Cargar el dataframe desde esa línea
    df_nca = pd.read_csv(nca_file, sep='\t', skiprows=header_idx, encoding='latin1')
    df_nca = df_nca.dropna(how='all')
    df_nca = df_nca[pd.to_numeric(df_nca["Individual"], errors='coerce').notna()]
    df_nca["Individual"] = df_nca["Individual"].astype(int)

    # Seleccionar columnas de interés
    col_cl = "CL (l/h)"
    col_vz = "V_z (l)"
    if col_cl not in df_nca.columns or col_vz not in df_nca.columns:
        raise ValueError("Columnas CL o V_z no encontradas en el archivo NCA.")
    
    params_nca = df_nca[["Individual", col_cl, col_vz]].copy()
    params_nca.columns = ["ID", "CL", "Vz"]
    for col in ["CL", "Vz"]:
        params_nca[col] = pd.to_numeric(params_nca[col], errors='coerce')
    params_nca = params_nca.dropna()

    # Parámetros poblacionales (theta) - media geométrica
    theta_CL = np.exp(np.mean(np.log(params_nca["CL"])))
    theta_Vz = np.exp(np.mean(np.log(params_nca["Vz"])))
    
    # Descomposición de Vz en V1 y V2 (ajusta según tu modelo)
    frac_v1 = 0.3   # volumen central
    frac_v2 = 0.7   # volumen periférico
    theta_V1 = frac_v1 * theta_Vz
    theta_V2 = frac_v2 * theta_Vz
    theta_Q = 10.0   # valor inicial para Q (ajústalo)
    
    theta0 = np.array([theta_CL, theta_V1, theta_V2, theta_Q])

    # Matriz de covarianza Omega (en escala log)
    log_cl = np.log(params_nca["CL"])
    log_vz = np.log(params_nca["Vz"])
    omega2_cl = np.var(log_cl, ddof=1)
    omega2_vz = np.var(log_vz, ddof=1)
    cov_log = np.cov(log_cl, log_vz)[0, 1]
    corr_log = cov_log / np.sqrt(omega2_cl * omega2_vz) if (omega2_cl > 0 and omega2_vz > 0) else 0.0

    # Asumimos que V1 y V2 tienen la misma varianza relativa que Vz
    omega2_v1 = omega2_vz
    omega2_v2 = omega2_vz
    omega2_q = 0.09   # para Q, CV ~30%

    # Construir matriz con correlaciones
    Omega0 = np.array([
        [omega2_cl, corr_log * np.sqrt(omega2_cl * omega2_v1), corr_log * np.sqrt(omega2_cl * omega2_v2), 0.0],
        [corr_log * np.sqrt(omega2_cl * omega2_v1), omega2_v1, omega2_v1, 0.0],
        [corr_log * np.sqrt(omega2_cl * omega2_v2), omega2_v1, omega2_v2, 0.0],
        [0.0, 0.0, 0.0, omega2_q]
    ])
    Omega0 = 0.5 * (Omega0 + Omega0.T)
    # Regularización mínima
    Omega0[np.diag_indices_from(Omega0)] += 1e-6

    # ETAs iniciales
    eta_init_dict = {}
    for _, row in params_nca.iterrows():
        iid = int(row["ID"])
        cl_i = row["CL"]
        vz_i = row["Vz"]
        eta_cl = np.log(cl_i / theta_CL) if cl_i > 0 else 0.0
        eta_v1 = np.log((frac_v1 * vz_i) / theta_V1) if vz_i > 0 else 0.0
        eta_v2 = np.log((frac_v2 * vz_i) / theta_V2) if vz_i > 0 else 0.0
        eta_q = 0.0
        eta_init_dict[iid] = np.array([eta_cl, eta_v1, eta_v2, eta_q])

    # sigma inicial (por defecto)
    sigma0 = np.array([0.1])

    return theta0, Omega0, sigma0, eta_init_dict


'''
Si quieres mayor precisión, implementa la estimación basada en residuos. No es obligatorio, pero puede ayudar en problemas donde la escala de los datos sea muy distinta a la típica.
# Después de obtener theta0_nca y eta_init_dict, calcular sigma0
residuals = []
for i_id in ids:
    data_ind = data[data["ID"] == i_id].copy()
    eta = eta_init_dict[i_id]
    pars = individual_params(theta0, eta, cfg["re_type"])
    pred_all = run_deck_with_params(pars, data_ind)  # o usar el ejecutor
    mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
    y_obs = data_ind.loc[mask, cfg["obs_column"]].values
    pred = pred_all[mask]
    if cfg["transform"] == "log":
        y_obs = np.log(np.clip(y_obs, 1e-12, None))
        pred = np.log(np.clip(pred, 1e-12, None))
    residuals.extend(y_obs - pred)
sigma0 = np.array([np.std(residuals)])
'''
