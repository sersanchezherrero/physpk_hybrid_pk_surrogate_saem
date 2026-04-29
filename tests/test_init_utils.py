import json

import numpy as np
import pandas as pd

import init_utils
from config import ModelConfig


def test_load_previous_run_initialization_from_final_estimates(tmp_path):
    run_dir = tmp_path / "prev_run"
    run_dir.mkdir()

    theta = [3.0, 30.0, 60.0, 9.0]
    omega = np.diag([0.1, 0.2, 0.3, 0.4]).tolist()
    sigma = [0.15]

    with open(run_dir / "final_estimates.json", "w") as f:
        json.dump({"theta_hat": theta, "Omega_hat": omega, "sigma_hat": sigma}, f)

    ebes_df = pd.DataFrame({
        "ID": [1, 2],
        "ETA1": [0.1, -0.1],
        "ETA2": [0.2, -0.2],
        "ETA3": [0.05, -0.05],
        "ETA4": [0.3, -0.3],
    })
    ebes_df.to_csv(run_dir / "EBEs.csv", index=False)

    theta0, Omega0, sigma0, eta_init = init_utils.load_previous_run_initialization(str(run_dir), ModelConfig())

    np.testing.assert_allclose(theta0, theta)
    np.testing.assert_allclose(Omega0, np.diag([0.1, 0.2, 0.3, 0.4]))
    np.testing.assert_allclose(sigma0, sigma)
    np.testing.assert_allclose(eta_init[1], np.array([0.1, 0.2, 0.05, 0.3]))


def test_load_previous_run_initialization_from_trace_and_ebes(tmp_path):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()

    trace_df = pd.DataFrame({
        "iter": [1, 2],
        "theta": ["[ 2.0 40.0 20.0 10.0]", "[ 3.0 50.0 25.0 11.0]"],
        "sigma": ["[0.2]", "[0.3]"],
    })
    trace_df.to_csv(run_dir / "SAEM_trace.csv", index=False)

    ebes_df = pd.DataFrame({
        "ID": [1, 2, 3],
        "ETA1": [0.1, 0.0, -0.1],
        "ETA2": [0.2, 0.0, -0.2],
        "ETA3": [0.05, 0.0, -0.05],
        "ETA4": [0.3, 0.0, -0.3],
    })
    ebes_df.to_csv(run_dir / "EBEs.csv", index=False)

    theta0, Omega0, sigma0, eta_init = init_utils.load_previous_run_initialization(str(run_dir), ModelConfig())

    np.testing.assert_allclose(theta0, np.array([3.0, 50.0, 25.0, 11.0]))
    np.testing.assert_allclose(sigma0, np.array([0.3]))
    assert np.all(np.diag(Omega0) > 0.0)
    np.testing.assert_allclose(eta_init[3], np.array([-0.1, -0.2, -0.05, -0.3]))


def test_load_previous_run_initialization_from_config_init_source(tmp_path):
    run_dir = tmp_path / "config_run"
    run_dir.mkdir()

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "theta0": [6.9, 51.3, 44.1, 30.2],
                "Omega0": np.diag([0.01, 0.02, 0.03, 0.04]).tolist(),
                "sigma0": [0.16],
            },
            f,
        )

    theta0, Omega0, sigma0, eta_init = init_utils.load_previous_run_initialization(
        str(run_dir),
        ModelConfig(),
        source="init",
    )

    np.testing.assert_allclose(theta0, np.array([6.9, 51.3, 44.1, 30.2]))
    np.testing.assert_allclose(Omega0, np.diag([0.01, 0.02, 0.03, 0.04]))
    np.testing.assert_allclose(sigma0, np.array([0.16]))
    assert eta_init == {}


def test_estimate_two_stage_initial_values_recovers_reasonable_center(monkeypatch):
    def fake_run_deck(params, data_ind):
        time = data_ind["TIME"].to_numpy(dtype=float)
        dose = data_ind["Dose"].to_numpy(dtype=float)
        duration = data_ind["Duration"].to_numpy(dtype=float)
        return (
            params[0]
            + 0.3 * params[1] * time
            + 0.01 * params[2] * dose
            + 0.5 * params[3] * duration
        )

    monkeypatch.setattr(init_utils, "original_run_deck", fake_run_deck)

    phi_id1 = np.array([2.0, 30.0, 10.0, 5.0], dtype=float)
    phi_id2 = np.array([4.0, 40.0, 20.0, 8.0], dtype=float)

    base = pd.DataFrame({
        "TIME": [0.0, 1.0, 2.0, 4.0],
        "Dose": [100.0, 50.0, 0.0, 0.0],
        "Duration": [1.0, 1.0, 0.5, 0.0],
        "Meas": [1, 1, 1, 1],
    })

    data_1 = base.copy()
    data_1["ID"] = 1
    data_1["LNCONC"] = fake_run_deck(phi_id1, data_1)

    data_2 = base.copy()
    data_2["ID"] = 2
    data_2["LNCONC"] = fake_run_deck(phi_id2, data_2)

    data = pd.concat([data_1, data_2], ignore_index=True)
    cfg = ModelConfig()

    theta0, Omega0, sigma0, eta_init = init_utils.estimate_two_stage_initial_values(
        data=data,
        ids=np.array([1, 2]),
        cfg=cfg,
        theta_seed=np.array([3.0, 35.0, 15.0, 6.0]),
        method="L-BFGS-B",
        maxiter=80,
    )

    expected_theta = np.exp(np.mean(np.log(np.vstack([phi_id1, phi_id2])), axis=0))
    np.testing.assert_allclose(theta0, expected_theta, rtol=0.15, atol=0.2)
    assert np.all(np.diag(Omega0) > 0.0)
    assert sigma0.shape == (1,)
    assert sigma0[0] <= 0.2
    assert 1 in eta_init and 2 in eta_init
