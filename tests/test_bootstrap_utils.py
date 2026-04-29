import numpy as np
import pandas as pd

from bootstrap_utils import (
    apply_sparse_sampling_windows,
    generate_parametric_bootstrap_population,
    parse_sparse_windows,
    patient_bootstrap_population,
)
from config import ModelConfig


def test_patient_bootstrap_population_resamples_full_patients():
    data = pd.DataFrame({
        "ID": [10, 10, 20, 20],
        "TIME": [0.0, 1.0, 0.0, 1.0],
        "Dose": [100.0, 0.0, 120.0, 0.0],
        "Duration": [1.0, 0.0, 1.0, 0.0],
        "LNCONC": [1.0, 0.5, 2.0, 1.0],
    })

    out = patient_bootstrap_population(data, ids=[10, 20], n_virtual=3, seed=123)

    assert sorted(out["ID"].unique().tolist()) == [1, 2, 3]
    assert set(out["SOURCE_ID"].unique().tolist()).issubset({10, 20})
    assert set(out["BOOTSTRAP_KIND"].unique().tolist()) == {"patient"}
    counts = out.groupby("ID").size().to_dict()
    assert counts == {1: 2, 2: 2, 3: 2}


def test_patient_bootstrap_population_balanced_counts_sources_evenly():
    data = pd.DataFrame({
        "ID": [10, 10, 20, 20],
        "TIME": [0.0, 1.0, 0.0, 1.0],
        "Dose": [100.0, 0.0, 120.0, 0.0],
        "Duration": [1.0, 0.0, 1.0, 0.0],
        "LNCONC": [1.0, 0.5, 2.0, 1.0],
    })

    out = patient_bootstrap_population(
        data, ids=[10, 20], n_virtual=4, seed=123, balanced=True
    )

    source_counts = out.groupby("SOURCE_ID")["ID"].nunique().to_dict()
    assert source_counts == {10: 2, 20: 2}


def test_generate_parametric_bootstrap_population_simulates_new_profiles():
    def fake_simulate_fn(params, data_ind):
        return np.linspace(1.0, float(len(data_ind)), len(data_ind), dtype=float)

    data = pd.DataFrame({
        "ID": [10, 10, 10],
        "TIME": [0.0, 1.0, 2.0],
        "Dose": [100.0, 0.0, 0.0],
        "Duration": [1.0, 0.0, 0.0],
        "Meas": [1, 0, 1],
        "LNCONC": [50.0, 60.0, 70.0],
    })
    cfg = ModelConfig()

    out = generate_parametric_bootstrap_population(
        data=data,
        ids=[10],
        theta_pop=np.array([2.0, 40.0, 20.0, 10.0], dtype=float),
        Omega=np.zeros((4, 4), dtype=float),
        sigma=np.array([0.0], dtype=float),
        cfg=cfg,
        n_virtual=2,
        seed=123,
        simulate_fn=fake_simulate_fn,
        include_residual_noise=False,
    )

    assert sorted(out["ID"].unique().tolist()) == [1, 2]
    assert set(out["SOURCE_ID"].unique().tolist()) == {10}
    assert set(out["BOOTSTRAP_KIND"].unique().tolist()) == {"parametric"}
    assert "SIM_PRED" in out.columns
    assert "ETA_01" in out.columns
    assert "PARAM_01" in out.columns

    first_patient = out[out["ID"] == 1].reset_index(drop=True)
    np.testing.assert_allclose(first_patient["SIM_PRED"].to_numpy(dtype=float), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(first_patient.loc[[0, 2], "LNCONC"].to_numpy(dtype=float), np.array([1.0, 3.0]))
    np.testing.assert_allclose(first_patient.loc[[1], "LNCONC"].to_numpy(dtype=float), np.array([60.0]))


def test_generate_parametric_bootstrap_population_balanced_counts_sources_evenly():
    def fake_simulate_fn(params, data_ind):
        return np.full(len(data_ind), 2.0, dtype=float)

    data = pd.DataFrame({
        "ID": [10, 10, 20, 20],
        "TIME": [0.0, 1.0, 0.0, 1.0],
        "Dose": [100.0, 0.0, 120.0, 0.0],
        "Duration": [1.0, 0.0, 1.0, 0.0],
        "Meas": [1, 1, 1, 1],
        "LNCONC": [1.0, 0.5, 2.0, 1.0],
    })
    cfg = ModelConfig()

    out = generate_parametric_bootstrap_population(
        data=data,
        ids=[10, 20],
        theta_pop=np.array([2.0, 40.0, 20.0, 10.0], dtype=float),
        Omega=np.zeros((4, 4), dtype=float),
        sigma=np.array([0.0], dtype=float),
        cfg=cfg,
        n_virtual=4,
        seed=123,
        simulate_fn=fake_simulate_fn,
        include_residual_noise=False,
        balanced=True,
    )

    source_counts = out.groupby("SOURCE_ID")["ID"].nunique().to_dict()
    assert source_counts == {10: 2, 20: 2}


def test_parse_sparse_windows_reads_comma_separated_ranges():
    windows = parse_sparse_windows("0-1,1-24,24-120")

    assert windows == [(0.0, 1.0), (1.0, 24.0), (24.0, 120.0)]


def test_apply_sparse_sampling_windows_keeps_schedule_and_selects_one_per_window():
    data = pd.DataFrame({
        "ID": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "TIME": [0.25, 8.0, 24.0, 168.0, 840.0, 0.25, 8.0, 24.0, 168.0, 840.0],
        "Dose": [10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        "Duration": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "Meas": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "LNCONC": np.linspace(5.0, 1.0, 10),
    })

    out = apply_sparse_sampling_windows(
        data,
        windows=[(0.0, 24.0), (24.0, 500.0), (500.0, 840.0)],
        points_per_window=1,
        seed=123,
    )

    assert out.shape[0] == data.shape[0]
    assert out["SPARSE_SELECTED"].sum() == 6

    selected_counts = out[out["Meas"] != 0].groupby("ID").size().to_dict()
    assert selected_counts == {1: 3, 2: 3}

    selected_windows = (
        out[out["Meas"] != 0]
        .groupby("ID")["SPARSE_WINDOW"]
        .apply(list)
        .to_dict()
    )
    assert set(selected_windows[1]) == {"0-24", "24-500", "500-840"}
    assert set(selected_windows[2]) == {"0-24", "24-500", "500-840"}
