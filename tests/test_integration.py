# test_integration.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import SAEMConfig
from saem import SAEM

def simple_pk_model(params, times, dose):
    """
    Modelo de un compartimento, bolus IV.
    params: [CL, V] (aclaramiento, volumen)
    """
    CL, V = params
    return (dose / V) * np.exp(-CL / V * times)

class MockDeckExecutor:
    """
    Ejecutor simulado que utiliza el modelo PK simple en lugar del DECK real o emulador.
    """
    def run(self, params, data_ind):
        # Extraer dosis (asumimos que está en la primera fila)
        dose = data_ind["Dose"].iloc[0]
        times = data_ind["TIME"].values
        return simple_pk_model(params, times, dose)

def test_saem_converges_on_synthetic():
    """
    Prueba de integración: genera datos sintéticos con un modelo simple,
    ejecuta SAEM y verifica que las estimaciones se acercan a los valores reales.
    """
    # ---- Parámetros verdaderos ----
    true_theta = np.array([2.0, 40.0])       # CL, V
    true_Omega = np.diag([0.1, 0.1])         # varianza de ETAs
    true_sigma = np.array([0.2])              # error aditivo

    # ---- Generar datos sintéticos ----
    n_ids = 10
    ids = list(range(1, n_ids + 1))
    rng = np.random.default_rng(123)
    true_etas = {}
    data_list = []

    for i_id in ids:
        # Muestrear ETAs
        eta = rng.multivariate_normal(np.zeros(2), true_Omega)
        true_etas[i_id] = eta
        # Parámetros individuales (log‑aditivos)
        pars = true_theta * np.exp(eta)
        # Tiempos de muestreo (típicos en PK)
        times = np.array([0.5, 2.0, 5.0, 12.0, 24.0])
        dose = 100.0
        # Concentración sin ruido
        conc = simple_pk_model(pars, times, dose)
        # Añadir ruido
        y = conc + rng.normal(0, true_sigma[0], size=len(times))
        # Crear DataFrame para el individuo
        df = pd.DataFrame({
            "ID": i_id,
            "TIME": times,
            "Dose": [dose] + [0] * (len(times) - 1),
            "Duration": [0.0] * len(times),
            "Meas": [1] * len(times),
            "LNCONC": y
        })
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    # ---- Configuración del modelo (adaptada a 2 parámetros) ----
    class TestModelConfig:
        n_par = 2
        re_type = "log-additive"
        err_model = "add"
        transform = "identity"
        obs_column = "LNCONC"

        @property
        def n_sigma(self):
            return 1

    model_cfg = TestModelConfig()

    # ---- Configuración del SAEM (iteraciones reducidas para test) ----
    saem_cfg = SAEMConfig(
        n_iter=50,
        n_burn=25,
        n_mcmc=5,
        step_size_eta=0.1,
        use_neural_sstep=False,          # desactivamos red neuronal por simplicidad
        retrain_emulator_every=0,        # no reentrenar emulador
        conv_epsilon=1e-2,
        conv_window=10
    )

    # ---- Ejecutor simulado ----
    mock_executor = MockDeckExecutor()

    # ---- Valores iniciales (ligeramente alejados de la verdad) ----
    theta0 = np.array([1.5, 30.0])
    Omega0 = np.diag([0.2, 0.2])
    sigma0 = np.array([0.3])

    # ---- Crear instancia SAEM y ejecutar ----
    saem = SAEM(
        data, ids, model_cfg, saem_cfg,
        theta0, Omega0, sigma0,
        seed=123,
        deck_executor=mock_executor
    )

    theta_hat, Omega_hat, sigma_hat, etas_hat = saem.fit()

    # ---- Verificar que las estimaciones se acercan a los valores verdaderos ----
    np.testing.assert_allclose(theta_hat, true_theta, rtol=0.2,
                               err_msg="Theta estimado no concuerda con el verdadero")
    np.testing.assert_allclose(np.diag(Omega_hat), np.diag(true_Omega), rtol=0.5,
                               err_msg="Omega estimado no concuerda con el verdadero")
    np.testing.assert_allclose(sigma_hat, true_sigma, rtol=0.2,
                               err_msg="Sigma estimado no concuerda con el verdadero")

    # Opcional: comprobar que las ETAs estimadas están correlacionadas con las verdaderas
    # (No lo hacemos aquí para mantener el test simple)

if __name__ == "__main__":
    test_saem_converges_on_synthetic()
    print("Test de integración pasado correctamente.")
