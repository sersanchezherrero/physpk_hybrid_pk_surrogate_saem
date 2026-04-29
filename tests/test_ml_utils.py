import numpy as np
import pandas as pd

import deck_interface
import ml_utils
from config import ModelConfig
from ml_utils import (
    build_virtual_dataset,
    build_xgb_baseline_cache,
    build_xgb_pointwise_sample_weights,
    build_sequence_mask,
    fit_target_scaler,
    flatten_xgb_pointwise_dataset,
    generate_emulator_data,
    generate_emulator_data_local,
    inverse_scale_target_sequences,
    masked_mae,
    scale_target_sequences,
    validate_emulator,
    validate_emulator_with_etas,
)


def test_generate_emulator_data_preserves_raw_deck_output(monkeypatch):
    deck_output = np.array([0.5, -0.5], dtype=float)

    def fake_run_deck(params, data_ind):
        return deck_output[: len(data_ind)].copy()

    monkeypatch.setattr(deck_interface, "original_run_deck", fake_run_deck)

    data = pd.DataFrame({
        "ID": [1, 1],
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
        "Meas": [1, 1],
        "LNCONC": [0.0, 0.0],
    })
    cfg = ModelConfig()

    _, _, y, lens, max_len = generate_emulator_data(
        data=data,
        ids=[1],
        cfg=cfg,
        n_pop=1,
        n_ind_per_pop=1,
        theta_center=np.array([2.0, 40.0, 20.0, 10.0]),
        theta_sigma=0.0,
        omega_range=(0.01, 0.01),
        sigma_range=(0.1, 0.1),
        dose_range=(100.0, 100.0),
        seed=123,
    )

    assert max_len == 2
    np.testing.assert_array_equal(lens, np.array([2]))
    np.testing.assert_allclose(y[0, :2, 0], deck_output)


def test_scaler_and_mae_ignore_padding():
    y = np.array([
        [[1.0], [2.0], [999.0]],
        [[10.0], [888.0], [777.0]],
    ])
    lengths = np.array([2, 1])

    mask = build_sequence_mask(lengths, max_len=3)
    np.testing.assert_array_equal(mask, np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))

    scaler = fit_target_scaler(y, lengths)
    np.testing.assert_allclose(scaler.mean_[0], (1.0 + 2.0 + 10.0) / 3.0)

    y_scaled = scale_target_sequences(y, lengths, scaler)
    assert y_scaled[0, 2, 0] == 0.0
    assert y_scaled[1, 1, 0] == 0.0
    assert y_scaled[1, 2, 0] == 0.0

    y_restored = inverse_scale_target_sequences(y_scaled, lengths, scaler)
    np.testing.assert_allclose(y_restored[0, :2, 0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(y_restored[1, :1, 0], np.array([10.0]))

    y_pred = y.copy()
    y_pred[0, 2, 0] = -123.0
    y_pred[1, 1, 0] = -456.0
    y_pred[1, 2, 0] = -789.0
    assert masked_mae(y, y_pred, lengths) == 0.0


def test_generate_emulator_data_local_preserves_raw_deck_output(monkeypatch):
    deck_output = np.array([1.5, -0.25], dtype=float)

    def fake_run_deck(params, data_ind):
        return deck_output[: len(data_ind)].copy()

    monkeypatch.setattr(deck_interface, "original_run_deck", fake_run_deck)

    data = pd.DataFrame({
        "ID": [1, 1, 2, 2],
        "TIME": [0.0, 1.0, 0.0, 1.0],
        "Dose": [100.0, 0.0, 120.0, 0.0],
        "Duration": [1.0, 0.0, 1.0, 0.0],
        "Meas": [1, 1, 1, 1],
        "LNCONC": [0.0, 0.0, 0.0, 0.0],
    })
    cfg = ModelConfig()

    _, _, y, lens, max_len = generate_emulator_data_local(
        data=data,
        ids=[1, 2],
        theta_pop=np.array([2.0, 40.0, 20.0, 10.0]),
        Omega=np.diag([0.01, 0.01, 0.01, 0.01]),
        sigma=np.array([0.1]),
        cfg=cfg,
        n_virtual=3,
        seed=123,
    )

    assert max_len == 2
    np.testing.assert_array_equal(lens, np.array([2, 2, 2]))
    np.testing.assert_allclose(y[:, :2, 0], np.tile(deck_output, (3, 1)))


def test_build_virtual_dataset_uses_real_deck_by_default(monkeypatch):
    real_output = np.array([9.0, 8.0], dtype=float)
    emu_output = np.array([1.0, 0.5], dtype=float)

    def fake_real(params, data_ind):
        return real_output[: len(data_ind)].copy()

    def fake_active_backend(params, data_ind):
        return emu_output[: len(data_ind)].copy()

    monkeypatch.setattr(ml_utils, "original_run_deck", fake_real)
    monkeypatch.setattr(ml_utils, "run_deck", fake_active_backend)

    data = pd.DataFrame({
        "ID": [1, 1],
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
        "Meas": [1, 1],
        "LNCONC": [0.0, 0.0],
    })
    cfg = ModelConfig()

    X, Y = build_virtual_dataset(
        data=data,
        ids=[1],
        theta=np.array([2.0, 40.0, 20.0, 10.0]),
        Omega=np.zeros((4, 4), dtype=float),
        sigma=np.array([0.0]),
        cfg=cfg,
        n_per_id=1,
        seed=123,
    )

    assert X.shape == (1, 12)
    assert Y.shape == (1, 4)
    np.testing.assert_allclose(X[0, 3], 9.0)  # cmax
    np.testing.assert_allclose(X[0, 5], 8.0)  # cmin
    np.testing.assert_allclose(X[0, 6], 8.5)  # auc trapezoidal


def test_flatten_xgb_pointwise_dataset_uses_only_real_lengths():
    Xp = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    Xs = np.array([[
        [0.0, 100.0, 1.0],
        [1.0, 0.0, 0.0],
        [999.0, 999.0, 999.0],
    ]], dtype=float)
    y = np.array([[[10.0], [20.0], [999.0]]], dtype=float)
    lens = np.array([2], dtype=int)

    X_rows, y_rows = flatten_xgb_pointwise_dataset(Xp, Xs, y, lens)

    assert X_rows.shape == (2, 15)
    np.testing.assert_allclose(y_rows, np.array([10.0, 20.0]))


def test_flatten_xgb_pointwise_dataset_supports_residual_targets():
    Xp = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    Xs = np.array([[
        [0.0, 100.0, 1.0],
        [1.0, 0.0, 0.0],
        [999.0, 999.0, 999.0],
    ]], dtype=float)
    y = np.array([[[10.0], [20.0], [999.0]]], dtype=float)
    lens = np.array([2], dtype=int)
    signature = deck_interface.build_schedule_signature_from_arrays(
        Xs[0, :2, 0],
        Xs[0, :2, 1],
        Xs[0, :2, 2],
    )
    baseline_cache = {signature: np.array([7.0, 18.0], dtype=float)}

    X_rows, y_rows = flatten_xgb_pointwise_dataset(
        Xp,
        Xs,
        y,
        lens,
        baseline_cache=baseline_cache,
        residual_mode=True,
        baseline_feature=True,
    )

    assert X_rows.shape == (2, 16)
    np.testing.assert_allclose(X_rows[:, -1], np.array([7.0, 18.0], dtype=np.float32))
    np.testing.assert_allclose(y_rows, np.array([3.0, 2.0]))


def test_build_xgb_baseline_cache_uses_real_deck_center(monkeypatch):
    calls = []

    def fake_real(params, data_ind):
        calls.append((params.copy(), data_ind.shape[0]))
        return np.arange(data_ind.shape[0], dtype=float) + params[0]

    monkeypatch.setattr(ml_utils, "original_run_deck", fake_real)

    data = pd.DataFrame({
        "ID": [1, 1, 2, 2],
        "TIME": [0.0, 1.0, 0.0, 1.0],
        "Dose": [100.0, 0.0, 100.0, 0.0],
        "Duration": [1.0, 0.0, 1.0, 0.0],
    })

    cache = build_xgb_baseline_cache(
        data=data,
        ids=[1, 2],
        theta_center=np.array([6.0, 40.0, 20.0, 10.0], dtype=float),
    )

    assert len(cache) == 1
    assert len(calls) == 1
    baseline = next(iter(cache.values()))
    np.testing.assert_allclose(baseline, np.array([6.0, 7.0]))


def test_build_xgb_pointwise_sample_weights_emphasizes_peak_and_dose():
    Xs = np.array([[
        [0.0, 100.0, 1.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]], dtype=float)
    y = np.array([[
        [0.1],
        [0.6],
        [1.3],
        [0.4],
    ]], dtype=float)
    lens = np.array([4], dtype=int)

    weights = build_xgb_pointwise_sample_weights(
        Xs,
        y,
        lens,
        peak_emphasis=4.0,
        peak_window=1,
        dose_emphasis=1.5,
    )

    np.testing.assert_allclose(weights, np.array([2.5, 5.0, 5.0, 5.0]))


def test_validate_emulator_with_etas_handles_log_scale_outputs(monkeypatch):
    class DummyDeckExecutor:
        def run_real(self, params, data_ind):
            return np.array([2.0, 1.0, 0.0, -1.0], dtype=float)[: len(data_ind)]

    def fake_run_deck_emulated(params, data_ind, emulator, max_len, scaler_y):
        return np.array([2.0, 1.0, 0.0, -1.0], dtype=float)[: len(data_ind)]

    monkeypatch.setattr(ml_utils, "run_deck_emulated", fake_run_deck_emulated)

    data = pd.DataFrame({
        "ID": [1, 1, 1, 1],
        "TIME": [0.0, 1.0, 2.0, 3.0],
        "Dose": [100.0, 0.0, 0.0, 0.0],
        "Duration": [0.1, 0.0, 0.0, 0.0],
        "Meas": [1, 1, 1, 1],
        "LNCONC": [2.0, 1.0, 0.0, -1.0],
    })
    cfg = ModelConfig()

    results = validate_emulator_with_etas(
        emulator=object(),
        max_len=4,
        scaler_y=None,
        data=data,
        ids=[1],
        theta=np.array([2.0, 40.0, 20.0, 10.0]),
        Omega=np.diag([1e-6, 1e-6, 1e-6, 1e-6]),
        sigma=np.array([0.1]),
        cfg=cfg,
        n_etas_per_id=2,
        n_val=1,
        deck_executor=DummyDeckExecutor(),
    )

    assert results["n_profiles"] == 2
    assert results["n_ids"] == 1
    assert results["mean_rel_error"] == 0.0
    assert results["mean_auc_error"] == 0.0
    assert results["mean_cmax_error"] == 0.0
    assert results["mean_tmax_abs_error"] == 0.0
    assert results["mean_terminal_slope_error"] == 0.0


def test_validate_emulator_computes_auc_on_natural_scale(monkeypatch):
    ln2 = np.log(2.0)
    ln4 = np.log(4.0)

    def fake_real(params, data_ind):
        return np.array([0.0, ln2], dtype=float)[: len(data_ind)]

    def fake_run_deck_emulated(params, data_ind, emulator, max_len, scaler_y):
        return np.array([0.0, ln4], dtype=float)[: len(data_ind)]

    monkeypatch.setattr(deck_interface, "original_run_deck", fake_real)
    monkeypatch.setattr(ml_utils, "run_deck_emulated", fake_run_deck_emulated)

    data = pd.DataFrame({
        "ID": [1, 1],
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [0.1, 0.0],
        "Meas": [1, 1],
        "LNCONC": [0.0, ln2],
    })
    cfg = ModelConfig()

    results = validate_emulator(
        emulator=object(),
        max_len=2,
        scaler_y=None,
        data=data,
        ids=[1],
        theta=np.array([2.0, 40.0, 20.0, 10.0]),
        sigma=np.array([0.1]),
        cfg=cfg,
        n_val=1,
    )

    np.testing.assert_allclose(results["mean_auc_error"], 2.0 / 3.0, atol=1e-6)
