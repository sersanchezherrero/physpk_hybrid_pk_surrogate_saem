# models.py
import numpy as np

def individual_params(theta, eta, re_type="log-additive"):
    theta = np.asarray(theta)
    eta = np.asarray(eta)
    if re_type == "additive":
        return theta + eta
    elif re_type == "proportional":
        return theta * (1.0 + eta)
    elif re_type in ("log-additive", "log-normal", "log-proportional"):
        return theta * np.exp(eta)
    else:
        raise ValueError(f"Unknown RE type: {re_type}")

def residual_sd(pred, sigma, err_model="add", transform="identity"):
    pred = np.asarray(pred)
    if err_model == "add":
        sd = np.full_like(pred, sigma[0])
    elif err_model == "prop":
        # En escala transformada, el error proporcional se aplica sobre f_t
        # Si transform == "log", f_t = log(f) y entonces sd = sigma * |f_t|? 
        # Normalmente en log, el error proporcional equivale a aditivo en log.
        # Mejor: definimos siempre sd en la escala de y_t.
        if transform == "identity":
            sd = sigma[0] * np.abs(pred)
        else:  # log
            sd = np.full_like(pred, sigma[0])  # En log, proporcional se vuelve aditivo
    elif err_model == "add+prop":
        if transform == "identity":
            sd = np.sqrt(sigma[0]**2 + (sigma[1] * pred)**2)
        else:
            sd = np.sqrt(sigma[0]**2 + sigma[1]**2)  # En log, ambos son aditivos
    else:
        raise ValueError(f"Unknown error model: {err_model}")
    return np.clip(sd, 1e-12, None)
