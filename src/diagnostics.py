# diagnostics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import logging
from scipy.optimize import minimize
from models import individual_params, residual_sd
from deck_interface import deck_executor as global_deck_executor

logger = logging.getLogger(__name__)

def compute_ebes(theta, Omega, sigma, data, ids, cfg, etas_dict=None, deck_executor=None):
    """
    Calcula las EBEs (η̂) para cada individuo mediante optimización de la posteriori.

    Args:
        theta (array): Parámetros poblacionales.
        Omega (array): Matriz de covarianza de efectos aleatorios.
        sigma (array): Parámetros de error.
        data (pd.DataFrame): Datos completos.
        ids (list): IDs de los individuos.
        cfg (ModelConfig): Configuración del modelo.
        etas_dict (dict, opcional): Diccionario ID -> eta para warm-start.

    Returns:
        pd.DataFrame: Columnas ID, ETA1..ETAn, OBJ (valor de la función objetivo).
    """
    if deck_executor is None:
        deck_executor = global_deck_executor

    n_par = cfg.n_par
    Omega_inv = np.linalg.inv(Omega)
    rows = []
    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        eta0 = etas_dict.get(i_id, np.zeros(n_par)) if etas_dict else np.zeros(n_par)

        def obj(eta):
            pars = individual_params(theta, eta, cfg.re_type)
            pred_all = deck_executor.run(pars, data_ind)
            mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
            y_obs = data_ind.loc[mask, cfg.obs_column].values
            pred = pred_all[mask]
            if len(y_obs) == 0:
                return 0.0
            if cfg.transform == "log":
                y_t = np.log(np.clip(y_obs, 1e-12, None))
                pred_t = np.log(np.clip(pred, 1e-12, None))
            else:
                y_t, pred_t = y_obs, pred
            sd = residual_sd(pred_t, sigma, cfg.err_model, cfg.transform)
            resid = (y_t - pred_t) / sd
            ll = -0.5 * np.sum(resid**2 + 2*np.log(sd) + np.log(2*np.pi))
            prior = -0.5 * (eta @ Omega_inv @ eta)
            return -(ll + prior)   # neg log posterior

        from scipy.optimize import minimize
        res = minimize(obj, x0=eta0, method="Powell", options={"maxiter": 20, "disp": False})
        eta_hat = res.x
        obj_val = res.fun if res.success else np.nan
        row = {"ID": i_id}
        for j in range(n_par):
            row[f"ETA{j+1}"] = eta_hat[j]
        row["OBJ"] = obj_val
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("EBEs calculadas para %d individuos.", len(df))
    return df

def compute_shrinkage(ebes_df, Omega):
    """
    Calcula η‑shrinkage por componente (varianza y desviación típica).

    Args:
        ebes_df (pd.DataFrame): DataFrame con columnas ETA1..ETAn.
        Omega (array): Matriz de covarianza de efectos aleatorios.

    Returns:
        pd.DataFrame: Columnas ETA, OMEGA_SD, EBE_SD, SHRINKAGE_VAR, SHRINKAGE_SD.
    """
    eta_cols = [c for c in ebes_df.columns if c.startswith("ETA")]
    if not eta_cols:
        raise ValueError("No ETA columns found")

    var_ebe = ebes_df[eta_cols].var(ddof=1).values
    sd_ebe = np.sqrt(var_ebe)
    omega_diag = np.diag(Omega)
    omega_sd = np.sqrt(omega_diag)

    shrink_var = 1.0 - var_ebe / omega_diag
    shrink_sd = 1.0 - sd_ebe / omega_sd

    rows = []
    for j, col in enumerate(eta_cols):
        rows.append({
            "ETA": col,
            "OMEGA_SD": omega_sd[j],
            "EBE_SD": sd_ebe[j],
            "SHRINKAGE_VAR": shrink_var[j],
            "SHRINKAGE_SD": shrink_sd[j]
        })
    df = pd.DataFrame(rows)
    logger.info("Shrinkage calculado.")
    return df

def compute_iwres(theta, sigma, data, ids, cfg, ebes_df, deck_executor=None):
    """
    Calcula IWRES (Individual Weighted Residuals) en la escala transformada.

    Args:
        theta (array): Parámetros poblacionales.
        sigma (array): Parámetros de error.
        data (pd.DataFrame): Datos completos.
        ids (list): IDs de los individuos.
        cfg (ModelConfig): Configuración del modelo.
        ebes_df (pd.DataFrame): DataFrame con las ETAs estimadas.

    Returns:
        tuple: (df con columnas ID, TIME, OBS_TRANS, PRED_TRANS, IWRES,
                diccionario con epsilon shrinkage global)
    """
    if deck_executor is None:
        deck_executor = global_deck_executor

    n_par = cfg.n_par
    eta_cols = [f"ETA{k+1}" for k in range(n_par)]
    eta_dict = {row["ID"]: row[eta_cols].values for _, row in ebes_df.iterrows()}

    rows = []
    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        eta_hat = eta_dict[i_id]
        pars = individual_params(theta, eta_hat, cfg.re_type)
        pred_all = deck_executor.run(pars, data_ind)
        mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
        y_obs = data_ind.loc[mask, cfg.obs_column].values
        pred = pred_all[mask]
        time_obs = data_ind.loc[mask, "TIME"].values

        if cfg.transform == "log":
            y_t = np.log(np.clip(y_obs, 1e-12, None))
            pred_t = np.log(np.clip(pred, 1e-12, None))
        else:
            y_t, pred_t = y_obs, pred

        sd = residual_sd(pred_t, sigma, cfg.err_model, cfg.transform)
        iwres = (y_t - pred_t) / sd

        for t, yv, fv, iw in zip(time_obs, y_t, pred_t, iwres):
            rows.append({
                "ID": i_id,
                "TIME": t,
                "OBS_TRANS": yv,
                "PRED_TRANS": fv,
                "IWRES": iw
            })
    df = pd.DataFrame(rows)
    # epsilon shrinkage global
    iw = df["IWRES"].values
    var_iw = np.var(iw, ddof=1)
    sd_iw = np.std(iw, ddof=1)
    eps_shrink = {
        "var_shrinkage": 1.0 - var_iw,
        "sd_shrinkage": 1.0 - sd_iw,
        "iwres_var": var_iw,
        "iwres_sd": sd_iw
    }
    logger.info("IWRES calculados para %d observaciones.", len(df))
    return df, eps_shrink

def generate_diagnostic_plots(ebes_df, data, theta, Omega, sigma, cfg, output_dir=None, deck_executor=None, ids=None):
    """
    Genera gráficos de diagnóstico: histogramas de ETAs, QQ-plot de IWRES, IWRES vs TIME.

    Args:
        ebes_df (pd.DataFrame): DataFrame con ETAs estimadas.
        data (pd.DataFrame): Datos completos.
        theta (array): Parámetros poblacionales.
        Omega (array): Matriz de covarianza.
        sigma (array): Parámetros de error.
        cfg (ModelConfig): Configuración del modelo.
        output_dir (str, opcional): Directorio para guardar los gráficos. Si es None, se muestran en pantalla.

    Returns:
        tuple: (iwres_df, eps_shrink) – DataFrame de IWRES y diccionario de shrinkage de epsilon.
    """
    # 1) Histogramas de ETAs
    eta_cols = [c for c in ebes_df.columns if c.startswith("ETA")]
    for col in eta_cols:
        plt.figure()
        plt.hist(ebes_df[col].values, bins=20, edgecolor="black")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col}")
        plt.grid(True)
        if output_dir:
            plt.savefig(f"{output_dir}/hist_{col}.png")
            plt.close()
        else:
            plt.show()

    # 2) IWRES y QQ-plot
    if ids is None:
        ids = data["ID"].unique()
    iwres_df, eps_shrink = compute_iwres(
        theta, sigma, data, ids, cfg, ebes_df, deck_executor=deck_executor
    )
    iw = iwres_df["IWRES"].dropna().values
    if len(iw) > 0:
        # QQ-plot
        n = len(iw)
        p = (np.arange(1, n+1) - 0.5) / n
        q_theor = norm.ppf(p)
        iw_sorted = np.sort(iw)
        plt.figure()
        plt.plot(q_theor, iw_sorted, "o", label="IWRES")
        lim = max(abs(q_theor).max(), abs(iw_sorted).max())
        plt.plot([-lim, lim], [-lim, lim], "r--", label="y=x")
        plt.xlabel("Theoretical N(0,1) quantiles")
        plt.ylabel("Ordered IWRES")
        plt.title("QQ-plot of IWRES")
        plt.grid(True)
        plt.legend()
        if output_dir:
            plt.savefig(f"{output_dir}/qqplot_IWRES.png")
            plt.close()
        else:
            plt.show()

        # IWRES vs TIME
        plt.figure()
        plt.plot(iwres_df["TIME"], iwres_df["IWRES"], "o")
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("IWRES")
        plt.title("IWRES vs TIME")
        plt.grid(True)
        if output_dir:
            plt.savefig(f"{output_dir}/iwres_vs_time.png")
            plt.close()
        else:
            plt.show()

    logger.info("Gráficos de diagnóstico generados.")
    return iwres_df, eps_shrink

def neg_log_lik_population(pop_vec, data, ids, cfg, deck_executor=None):
    """
    Calcula -2 log-verosimilitud aproximada (Laplace sin corrección) para un vector de parámetros poblacionales.
    pop_vec: [theta, L_flat, log_sigma]
    """
    if deck_executor is None:
        deck_executor = global_deck_executor

    n_par = cfg["n_par"]
    n_sigma = cfg["n_sigma"]
    n_L = n_par * (n_par + 1) // 2
    
    theta = pop_vec[:n_par]
    L_flat = pop_vec[n_par:n_par + n_L]
    log_sigma = pop_vec[n_par + n_L:]
    
    L = np.zeros((n_par, n_par))
    idx = 0
    for i in range(n_par):
        for j in range(i+1):
            L[i, j] = L_flat[idx]
            idx += 1
    Omega = L @ L.T
    Omega_inv = np.linalg.inv(Omega + 1e-8 * np.eye(n_par))
    sigma = np.exp(log_sigma)
    
    total = 0.0
    for i_id in ids:
        data_ind = data[data["ID"] == i_id].copy()
        
        # Función objetivo para eta
        def obj_eta(eta):
            pars = individual_params(theta, eta, cfg["re_type"])
            pred_all = deck_executor.run(pars, data_ind)
            mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
            y_obs = data_ind.loc[mask, cfg["obs_column"]].values
            pred = pred_all[mask]
            if len(y_obs) == 0:
                return 0.0
            if cfg["transform"] == "log":
                y_t = np.log(np.clip(y_obs, 1e-12, None))
                pred_t = np.log(np.clip(pred, 1e-12, None))
            else:
                y_t, pred_t = y_obs, pred
            sd = residual_sd(pred_t, sigma, cfg["err_model"], cfg["transform"])
            resid = (y_t - pred_t) / sd
            ll = -0.5 * np.sum(resid**2 + 2*np.log(sd) + np.log(2*np.pi))
            prior = -0.5 * (eta @ Omega_inv @ eta)
            return -(ll + prior)   # neg log posterior
        
        # Encontrar modo condicional
        res = minimize(obj_eta, x0=np.zeros(n_par), method='Powell', options={'maxiter': 20, 'disp': False})
        eta_hat = res.x
        total += 2 * obj_eta(eta_hat)  # -2 log posterior en el modo
    return total


def hessian_fd(f, x, eps=1e-4):
    """Aproximación del Hessiano por diferencias finitas centrales."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        x_i_plus = x.copy()
        x_i_plus[i] += eps
        f_i_plus = f(x_i_plus)
        
        x_i_minus = x.copy()
        x_i_minus[i] -= eps
        f_i_minus = f(x_i_minus)
        
        H[i, i] = (f_i_plus - 2*f0 + f_i_minus) / (eps**2)
        
        for j in range(i+1, n):
            x_ij_plus = x.copy()
            x_ij_plus[i] += eps
            x_ij_plus[j] += eps
            f_ij_plus = f(x_ij_plus)
            
            x_i_plus_j_minus = x.copy()
            x_i_plus_j_minus[i] += eps
            x_i_plus_j_minus[j] -= eps
            f_i_plus_j_minus = f(x_i_plus_j_minus)
            
            x_i_minus_j_plus = x.copy()
            x_i_minus_j_plus[i] -= eps
            x_i_minus_j_plus[j] += eps
            f_i_minus_j_plus = f(x_i_minus_j_plus)
            
            x_ij_minus = x.copy()
            x_ij_minus[i] -= eps
            x_ij_minus[j] -= eps
            f_ij_minus = f(x_ij_minus)
            
            H[i, j] = (f_ij_plus - f_i_plus_j_minus - f_i_minus_j_plus + f_ij_minus) / (4 * eps**2)
            H[j, i] = H[i, j]
    return H


def compute_population_se(theta_hat, Omega_hat, sigma_hat, data, ids, cfg, deck_executor=None):
    """
    Calcula errores estándar aproximados para los parámetros poblacionales.
    Devuelve un DataFrame con nombres, estimaciones, SE y CV%.
    """
    n_par = cfg["n_par"]
    n_sigma = cfg["n_sigma"]
    # Construir vector plano en la parametrización: [theta, L_flat, log_sigma]
    L = np.linalg.cholesky(Omega_hat + 1e-8 * np.eye(n_par))
    L_flat = L[np.tril_indices(n_par)]
    log_sigma = np.log(sigma_hat)
    pop_vec = np.concatenate([theta_hat, L_flat, log_sigma])
    
    # Función objetivo que depende solo de pop_vec
    def obj(p):
        return neg_log_lik_population(p, data, ids, cfg, deck_executor)
    
    # Calcular Hessiano
    H = hessian_fd(obj, pop_vec, eps=1e-4)
    H = 0.5 * (H + H.T)
    
    # Invertir con jitter
    cov = None
    for eps in [0.0, 1e-6, 1e-5, 1e-4]:
        try:
            cov = np.linalg.inv(H + eps * np.eye(len(pop_vec)))
            break
        except np.linalg.LinAlgError:
            continue
    if cov is None:
        print("No se pudo invertir el Hessiano.")
        return None
    
    se = np.sqrt(np.diag(cov))
    
    # Construir DataFrame de resultados
    n_L = len(L_flat)
    names = [f"THETA{i+1}" for i in range(n_par)] + \
            [f"L_{k+1}" for k in range(n_L)] + \
            [f"LOGSIGMA{j+1}" for j in range(n_sigma)]
    df_se = pd.DataFrame({
        "parameter": names,
        "estimate": pop_vec,
        "se": se,
        "cv_percent": 100 * se / np.abs(pop_vec)
    })
    return df_se
