import numpy as np
import pandas as pd

from config import ModelConfig
from diagnostics import compute_ebes, compute_iwres


class MockDeckExecutor:
    def run(self, params, data_ind):
        return np.full(len(data_ind), 1.0, dtype=float)


def test_diagnostics_use_injected_executor():
    data = pd.DataFrame({
        "ID": [1, 1, 1],
        "TIME": [0.5, 2.0, 5.0],
        "Dose": [100.0, 0.0, 0.0],
        "Duration": [0.0, 0.0, 0.0],
        "Meas": [1, 1, 1],
        "LNCONC": [1.0, 1.0, 1.0],
    })
    ids = [1]
    theta = np.array([2.0, 40.0, 20.0, 10.0])
    Omega = np.diag([0.1, 0.1, 0.1, 0.1])
    sigma = np.array([0.1])
    cfg = ModelConfig()
    executor = MockDeckExecutor()

    ebes_df = compute_ebes(theta, Omega, sigma, data, ids, cfg, deck_executor=executor)
    assert list(ebes_df["ID"]) == [1]

    iwres_df, eps_shrink = compute_iwres(
        theta, sigma, data, ids, cfg, ebes_df, deck_executor=executor
    )
    assert len(iwres_df) == 3
    assert "IWRES" in iwres_df.columns
    assert "iwres_sd" in eps_shrink
