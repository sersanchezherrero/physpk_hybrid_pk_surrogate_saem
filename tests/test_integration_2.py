# test_integration.py
import numpy as np
import pandas as pd
import os
import sys
import time
from scipy.stats import pearsonr
from config import ModelConfig, SAEMConfig
from saem import SAEM
from deck_interface import DeckExecutor
from models import individual_params

# ---------------------------
# Modelo PK simple (2 compartimentos, solución analítica aproximada)
# Usamos la misma función que en el test anterior
def simple_pk_model(params, times, dose):
    """
    params: [CL, V1, V2, Q]
    """
    CL, V1, V2, Q = params
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    beta = 0.5 * (k12 + k21 + k10 - np.sqrt((k12 + k21 + k10)**2 - 4*k21*k10))
    alpha = (k21 * k10) / beta
    A = (dose / V1) * (alpha - k21) / (alpha - beta)
    B = (dose / V1) * (beta - k21) / (beta - alpha)
    return A * np.exp(-alpha * times) + B * np.exp(-beta * times)

class MockDeckExecutor(DeckExecutor):
    """Ejecutor que usa el modelo analítico en lugar del DECK real."""
    def run(self, params, data_ind):
        dose = data_ind["Dose"].iloc[0]  # asumimos dosis única al tiempo 0
        times = data_ind["TIME"].values
        return simple_pk_model(params, times, dose)

def generate_synthetic_data(n_ids=20, times=None, true_theta=None, true_Omega=None, true_sigma=None, seed=123):
    """Genera datos sintéticos con estructura similar a Monoclonal_Antibody.txt."""
    if times is None:
        times = np.array([0.5, 2.0, 5.0, 12.0, 24.0])
    if true_theta is None:
        true_theta = np.array([2.0, 40.0, 20.0, 10.0])  # CL, V1, V2, Q
    if true_Omega is None:
        true_Omega = np.diag([0.09, 0.09, 0.09, 0.09])  # CV ~30%
    if true_sigma is None:
        true_sigma = np.array([0.15])  # error aditivo

    rng = np.random.default_rng(seed)
    ids = list(range(1, n_ids + 1))
    data_list = []
    true_etas = {}

    for i_id in ids:
        eta = rng.multivariate_normal(np.zeros(4), true_Omega)
        true_etas[i_id] = eta
        pars = true_theta * np.exp(eta)  # log-additive
        dose = 100.0
        pred = simple_pk_model(pars, times, dose)
        y = pred + rng.normal(0, true_sigma[0], size=len(times))
        df = pd.DataFrame({
            "ID": i_id,
            "TIME": times,
            "Dose": [dose] + [0] * (len(times)-1),
            "Duration": [0.0] * len(times),
            "Meas": [1] * len(times),
            "LNCONC": y
        })
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)
    return data, ids, true_theta, true_Omega, true_sigma, true_etas

def test_pipeline_on_synthetic():
    """Prueba completa del pipeline con datos sintéticos."""
    print("Generando datos sintéticos...")
    data, ids, true_theta, true_Omega, true_sigma, true_etas = generate_synthetic_data(n_ids=30)

    # Configuración
    model_cfg = ModelConfig(
        n_par=4,
        re_type="log-additive",
        err_model="add",
        transform="identity",
        obs_column="LNCONC"
    )
    saem_cfg = SAEMConfig(
        n_iter=200,
        n_burn=100,
        n_mcmc=20,
        step_size_eta=0.2,
        use_neural_sstep=False,  # desactivamos para prueba rápida
        retrain_emulator_every=0,
        conv_epsilon=1e-3,
        conv_window=20
    )

    # Valores iniciales (ligeramente alejados de la verdad)
    theta0 = true_theta * 1.2
    Omega0 = true_Omega * 1.5
    sigma0 = true_sigma * 1.5

    # Ejecutor simulado (sin emulador)
    mock_executor = MockDeckExecutor()

    # Crear SAEM
    saem = SAEM(
        data, ids, model_cfg, saem_cfg,
        theta0, Omega0, sigma0,
        seed=123,
        deck_executor=mock_executor
    )

    # Medir tiempo
    t0 = time.time()
    theta_hat, Omega_hat, sigma_hat, etas_hat = saem.fit()
    elapsed = time.time() - t0
    print(f"SAEM completado en {elapsed:.2f} segundos.")

    # Comparar con valores verdaderos
    print("\n--- Comparación de parámetros poblacionales ---")
    print(f"THETA verdadero: {true_theta}")
    print(f"THETA estimado : {theta_hat}")
    rel_error_theta = np.abs((theta_hat - true_theta) / true_theta)
    print(f"Error relativo THETA: {rel_error_theta}")
    assert np.all(rel_error_theta < 0.2), "Error en THETA superior al 20%"

    print(f"\nOMEGA diagonal verdadero: {np.diag(true_Omega)}")
    print(f"OMEGA diagonal estimado : {np.diag(Omega_hat)}")
    rel_error_omega = np.abs((np.diag(Omega_hat) - np.diag(true_Omega)) / (np.diag(true_Omega) + 1e-8))
    print(f"Error relativo OMEGA diag: {rel_error_omega}")
    assert np.all(rel_error_omega < 0.5), "Error en OMEGA superior al 50%"

    print(f"\nSIGMA verdadero: {true_sigma}")
    print(f"SIGMA estimado : {sigma_hat}")
    rel_error_sigma = np.abs((sigma_hat - true_sigma) / true_sigma)
    print(f"Error relativo SIGMA: {rel_error_sigma}")
    assert rel_error_sigma[0] < 0.3, "Error en SIGMA superior al 30%"

    # Correlación entre ETAs verdaderas y estimadas
    eta_true_list = np.array([true_etas[i] for i in ids])
    eta_hat_list = np.array([etas_hat[i] for i in ids])
    corrs = [pearsonr(eta_true_list[:, j], eta_hat_list[:, j])[0] for j in range(4)]
    print(f"\nCorrelación ETAs verdaderas vs estimadas: {corrs}")
    assert np.all(np.array(corrs) > 0.7), "Baja correlación en ETAs"

    # Verificar que la traza se guardó
    assert hasattr(saem, 'trace') and len(saem.trace) > 0

    # Opcional: ejecutar diagnósticos
    from diagnostics import compute_ebes, compute_shrinkage, compute_iwres
    ebes_df = compute_ebes(
        theta_hat, Omega_hat, sigma_hat, data, ids, model_cfg,
        etas_hat, deck_executor=mock_executor
    )
    assert not ebes_df.empty
    shrinkage_df = compute_shrinkage(ebes_df, Omega_hat)
    assert not shrinkage_df.empty
    iwres_df, eps_shrink = compute_iwres(
        theta_hat, sigma_hat, data, ids, model_cfg, ebes_df,
        deck_executor=mock_executor
    )
    assert not iwres_df.empty

    print("\n✅ Todos los tests pasaron. El pipeline funciona correctamente en datos sintéticos.")

if __name__ == "__main__":
    test_pipeline_on_synthetic()
