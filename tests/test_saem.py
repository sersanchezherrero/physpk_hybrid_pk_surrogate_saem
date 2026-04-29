import numpy as np
import pandas as pd
import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ModelConfig, SAEMConfig
from ml_utils import build_eta_posterior_net
from saem import SAEM

# Mock del DeckExecutor que devuelve predicciones ficticias
class MockDeckExecutor:
    def run(self, params, data_ind):
        # Devuelve un array de ceros del mismo tamaño que data_ind
        return np.zeros(len(data_ind))

@pytest.fixture
def dummy_data():
    data = pd.DataFrame({
        "ID": [1, 1, 1],
        "TIME": [0.5, 2.0, 5.0],
        "Dose": [100, 0, 0],
        "Duration": [0.1, 0, 0],
        "Meas": [1, 1, 1],
        "LNCONC": [4.5, 3.8, 2.9]
    })
    ids = [1]
    return data, ids

def test_saem_initialization(dummy_data):
    data, ids = dummy_data
    model_cfg = ModelConfig()
    saem_cfg = SAEMConfig(n_iter=10, n_burn=5, n_mcmc=5, step_size_eta=0.1)
    theta0 = np.array([2.0, 40.0, 20.0, 20.0])
    Omega0 = np.diag([0.212, 0.212, 0.212, 0.212])
    sigma0 = np.array([0.1])
    # Crear SAEM con el mock
    mock_executor = MockDeckExecutor()
    saem = SAEM(data, ids, model_cfg, saem_cfg, theta0, Omega0, sigma0, deck_executor=mock_executor)
    assert saem.theta is not None
    assert saem.Omega is not None
    # Probar loglik
    data_ind = data[data["ID"] == 1]
    ll = saem._loglik_individual(theta0, np.zeros(4), sigma0, data_ind)
    assert np.isfinite(ll)


def test_neural_mcmc_proposal_returns_finite_eta(dummy_data):
    data, ids = dummy_data
    model_cfg = ModelConfig()
    saem_cfg = SAEMConfig(n_iter=5, n_burn=2, n_mcmc=3, step_size_eta=0.1)
    theta0 = np.array([2.0, 40.0, 20.0, 20.0])
    Omega0 = np.diag([0.212, 0.212, 0.212, 0.212])
    sigma0 = np.array([0.1])
    mock_executor = MockDeckExecutor()

    saem = SAEM(data, ids, model_cfg, saem_cfg, theta0, Omega0, sigma0, deck_executor=mock_executor)
    saem.nn = build_eta_posterior_net(input_dim=12, n_par=model_cfg.n_par)

    data_ind = data[data["ID"] == 1]
    eta_new, accept = saem._sample_eta_mcmc(
        np.zeros(model_cfg.n_par),
        data_ind,
        step_size=saem_cfg.step_size_eta,
        n_steps=saem_cfg.n_mcmc,
        use_nn=True
    )

    assert eta_new.shape == (model_cfg.n_par,)
    assert np.all(np.isfinite(eta_new))
    assert 0 <= accept <= saem_cfg.n_mcmc


def test_empirical_eta_covariance_updates_from_buffer():
    data = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "TIME": [0.5] * 5,
        "Dose": [100.0] * 5,
        "Duration": [0.1] * 5,
        "Meas": [1] * 5,
        "LNCONC": [4.5, 4.2, 4.0, 3.7, 3.5],
    })
    ids = [1, 2, 3, 4, 5]
    model_cfg = ModelConfig()
    saem_cfg = SAEMConfig(
        n_iter=5,
        n_burn=2,
        n_mcmc=3,
        step_size_eta=0.1,
        use_empirical_eta_cov=True,
        emp_cov_update_freq=1,
        max_emp_cov_buffer=32,
    )
    theta0 = np.array([2.0, 40.0, 20.0, 20.0])
    Omega0 = np.diag([0.212, 0.212, 0.212, 0.212])
    sigma0 = np.array([0.1])
    mock_executor = MockDeckExecutor()

    saem = SAEM(data, ids, model_cfg, saem_cfg, theta0, Omega0, sigma0, deck_executor=mock_executor)
    saem.etas = {
        1: np.array([0.0, 0.1, 0.2, 0.3]),
        2: np.array([0.1, 0.2, 0.3, 0.4]),
        3: np.array([0.2, 0.3, 0.4, 0.5]),
        4: np.array([0.3, 0.4, 0.5, 0.6]),
        5: np.array([0.4, 0.5, 0.6, 0.7]),
    }
    emp_cov_before = saem.emp_cov.copy()

    saem._update_empirical_eta_cov_buffer(k=1)

    assert len(saem.emp_cov_buffer) == len(ids)
    assert saem.emp_cov.shape == Omega0.shape
    assert np.all(np.isfinite(saem.emp_cov))
    assert np.all(np.diag(saem.emp_cov) > 0.0)
    assert not np.allclose(saem.emp_cov, emp_cov_before)


def test_theta_trust_region_projects_theta_inside_emulator_domain(dummy_data):
    data, ids = dummy_data
    model_cfg = ModelConfig()
    saem_cfg = SAEMConfig(n_iter=5, n_burn=2, n_mcmc=3, step_size_eta=0.1)
    theta0 = np.array([6.9, 51.3, 44.1, 30.2], dtype=float)
    Omega0 = np.diag([0.01, 0.01, 0.01, 0.01])
    sigma0 = np.array([0.1])
    mock_executor = MockDeckExecutor()

    saem = SAEM(
        data,
        ids,
        model_cfg,
        saem_cfg,
        theta0,
        Omega0,
        sigma0,
        deck_executor=mock_executor,
        theta_trust_center=theta0,
        theta_trust_log_radius=0.5,
    )

    theta_far = np.array([6.9, 2980.0, 10.0, 9.0], dtype=float)
    theta_projected = saem._project_theta_to_trust_region(theta_far)

    phi_center = np.log(theta0)
    phi_projected = np.log(theta_projected)
    assert np.all(phi_projected <= phi_center + 0.5 + 1e-12)
    assert np.all(phi_projected >= phi_center - 0.5 - 1e-12)
    assert theta_projected[1] < 100.0


def test_saem_checkpoint_writes_partial_outputs(dummy_data, tmp_path):
    data, ids = dummy_data
    model_cfg = ModelConfig()
    saem_cfg = SAEMConfig(n_iter=5, n_burn=2, n_mcmc=3, step_size_eta=0.1)
    theta0 = np.array([2.0, 40.0, 20.0, 20.0])
    Omega0 = np.diag([0.212, 0.212, 0.212, 0.212])
    sigma0 = np.array([0.1])
    mock_executor = MockDeckExecutor()

    saem = SAEM(
        data,
        ids,
        model_cfg,
        saem_cfg,
        theta0,
        Omega0,
        sigma0,
        deck_executor=mock_executor,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=1,
    )
    saem.etas = {1: np.array([0.1, 0.2, 0.3, 0.4])}
    saem.trace.append({
        "iter": 1,
        "gamma": 1.0,
        "theta": theta0.copy(),
        "omega_diag": np.diag(Omega0).copy(),
        "sigma": sigma0.copy(),
        "acc_rate": 0.5,
        "step_size": 0.1,
    })

    saem.save_checkpoint(force=True, status="running")

    checkpoint_path = tmp_path / "final_estimates.partial.json"
    trace_path = tmp_path / "SAEM_trace.partial.csv"
    etas_path = tmp_path / "ETAs_partial.csv"

    assert checkpoint_path.exists()
    assert trace_path.exists()
    assert etas_path.exists()

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["iter"] == 1
    assert payload["status"] == "running"
    assert payload["theta_hat"][0] == pytest.approx(theta0[0])

    trace_df = pd.read_csv(trace_path)
    assert "theta1" in trace_df.columns
    assert trace_df.loc[0, "omega_diag1"] == pytest.approx(Omega0[0, 0])

    etas_df = pd.read_csv(etas_path)
    assert etas_df.loc[0, "ETA4"] == pytest.approx(0.4)


def test_saem_fit_resumes_from_existing_trace_without_restarting(dummy_data, tmp_path):
    data, ids = dummy_data
    model_cfg = ModelConfig()
    saem_cfg = SAEMConfig(n_iter=2, n_burn=1, n_mcmc=2, step_size_eta=0.1)
    theta0 = np.array([2.0, 40.0, 20.0, 20.0])
    Omega0 = np.diag([0.212, 0.212, 0.212, 0.212])
    sigma0 = np.array([0.1])
    mock_executor = MockDeckExecutor()

    saem = SAEM(
        data,
        ids,
        model_cfg,
        saem_cfg,
        theta0,
        Omega0,
        sigma0,
        deck_executor=mock_executor,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=1,
    )
    saem.etas = {1: np.array([0.1, 0.2, 0.3, 0.4])}
    saem.trace = [{
        "iter": 2,
        "gamma": 1.0,
        "theta": theta0.copy(),
        "omega_diag": np.diag(Omega0).copy(),
        "sigma": sigma0.copy(),
        "acc_rate": 0.5,
        "step_size": 0.25,
    }]

    theta_hat, Omega_hat, sigma_hat, etas_hat = saem.fit()

    assert theta_hat[0] == pytest.approx(theta0[0])
    assert sigma_hat[0] == pytest.approx(sigma0[0])
    assert etas_hat[1][3] == pytest.approx(0.4)

    payload = json.loads((tmp_path / "final_estimates.partial.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["iter"] == 2


def test_saem_fit_stops_cleanly_at_stop_after_iter(dummy_data, tmp_path):
    data, ids = dummy_data
    model_cfg = ModelConfig()
    saem_cfg = SAEMConfig(n_iter=4, n_burn=1, n_mcmc=2, step_size_eta=0.1, stop_after_iter=2)
    theta0 = np.array([2.0, 40.0, 20.0, 20.0])
    Omega0 = np.diag([0.212, 0.212, 0.212, 0.212])
    sigma0 = np.array([0.1])
    mock_executor = MockDeckExecutor()

    saem = SAEM(
        data,
        ids,
        model_cfg,
        saem_cfg,
        theta0,
        Omega0,
        sigma0,
        deck_executor=mock_executor,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=1,
        stop_after_iter=2,
    )

    theta_hat, Omega_hat, sigma_hat, etas_hat = saem.fit()

    assert saem.stop_reason == "stop_after_iter"
    assert len(saem.trace) == 2
    assert saem.trace[-1]["iter"] == 2
    assert theta_hat.shape == (4,)
    assert Omega_hat.shape == (4, 4)
    assert sigma_hat.shape == (1,)
    assert 1 in etas_hat

    payload = json.loads((tmp_path / "final_estimates.partial.json").read_text(encoding="utf-8"))
    assert payload["status"] == "stopped_after_iter"
    assert payload["iter"] == 2
