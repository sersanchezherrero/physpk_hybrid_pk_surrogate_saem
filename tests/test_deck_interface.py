import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import deck_interface


def test_run_real_ignores_active_emulator(monkeypatch):
    real_output = np.array([7.0, 7.0], dtype=float)
    emu_output = np.array([3.0, 3.0], dtype=float)

    def fake_real(params, data_ind):
        return real_output[: len(data_ind)].copy()

    def fake_emu(params, data_ind, emulator, max_len, scaler_y):
        return emu_output[: len(data_ind)].copy()

    monkeypatch.setattr(deck_interface, "original_run_deck", fake_real)
    monkeypatch.setattr(deck_interface, "run_deck_emulated", fake_emu)

    executor = deck_interface.DeckExecutor()
    executor.use_emulator(emulator=object(), max_len=2, scaler_y=object())

    data_ind = pd.DataFrame({
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
    })

    np.testing.assert_allclose(executor.run(np.ones(4), data_ind), emu_output)
    np.testing.assert_allclose(executor.run_real(np.ones(4), data_ind), real_output)
    assert executor.use_real_deck is False


def test_run_deck_emulated_uses_model_call_and_manual_inverse_scaling():
    class DummyTensor:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)

        def numpy(self):
            return self._array

    class DummyModel:
        def __call__(self, inputs, training=False):
            seq, params = inputs
            assert training is False
            assert seq.shape == (1, 2, 3)
            assert params.shape == (1, 4)
            return DummyTensor([[[1.0], [2.0]]])

    class DummyScaler:
        mean_ = np.array([10.0], dtype=float)
        scale_ = np.array([2.0], dtype=float)

    data_ind = pd.DataFrame({
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
    })

    pred = deck_interface.run_deck_emulated(
        np.array([1.0, 2.0, 3.0, 4.0]),
        data_ind,
        DummyModel(),
        max_len=2,
        scaler_y=DummyScaler(),
    )

    np.testing.assert_allclose(pred, np.array([12.0, 14.0]))


def test_run_deck_emulated_uses_pointwise_backend():
    class DummyPointwiseEmulator:
        def predict_sequence(self, params, data_ind):
            np.testing.assert_allclose(params, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
            assert list(data_ind["TIME"]) == [0.0, 1.0]
            return np.array([5.0, 6.0], dtype=float)

    data_ind = pd.DataFrame({
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
    })

    pred = deck_interface.run_deck_emulated(
        np.array([1.0, 2.0, 3.0, 4.0]),
        data_ind,
        DummyPointwiseEmulator(),
        max_len=2,
        scaler_y=None,
    )

    np.testing.assert_allclose(pred, np.array([5.0, 6.0]))


def test_original_run_deck_rejects_too_many_params(monkeypatch):
    monkeypatch.setattr(deck_interface, "_load_real_deck", lambda: object())
    data_ind = pd.DataFrame({
        "TIME": [0.0],
        "Dose": [100.0],
        "Duration": [1.0],
    })

    try:
        deck_interface.original_run_deck(np.ones(deck_interface.MAX_DECK_PARAMS + 1), data_ind)
    except ValueError as exc:
        assert "n_params" in str(exc)
        assert str(deck_interface.MAX_DECK_PARAMS) in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError por exceso de parametros")


def test_original_run_deck_rejects_too_many_samples(monkeypatch):
    monkeypatch.setattr(deck_interface, "_load_real_deck", lambda: object())
    n_rows = deck_interface.MAX_DECK_SAMPLES + 1
    data_ind = pd.DataFrame({
        "TIME": np.arange(n_rows, dtype=float),
        "Dose": np.zeros(n_rows, dtype=float),
        "Duration": np.zeros(n_rows, dtype=float),
    })

    try:
        deck_interface.original_run_deck(np.ones(4), data_ind)
    except ValueError as exc:
        assert "num_samples" in str(exc)
        assert str(deck_interface.MAX_DECK_SAMPLES) in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError por exceso de muestras")


def test_xgb_pointwise_emulator_inverts_signed_log1p_predictions():
    class DummyModel:
        def predict(self, features):
            assert features.shape == (2, 15)
            return np.array([np.log1p(4.0), -np.log1p(2.0)], dtype=float)

    emulator = deck_interface.XGBPointwiseEmulator(
        DummyModel(),
        target_transform="signed_log1p",
    )
    data_ind = pd.DataFrame({
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
    })

    pred = emulator.predict_sequence(np.array([1.0, 2.0, 3.0, 4.0]), data_ind)

    np.testing.assert_allclose(pred, np.array([4.0, -2.0]), atol=1e-6)


def test_xgb_residual_pointwise_emulator_adds_cached_baseline():
    class DummyModel:
        def predict(self, features):
            assert features.shape == (2, 16)
            np.testing.assert_allclose(features[:, -1], np.array([10.0, 12.0], dtype=np.float32))
            return np.array([np.log1p(1.0), -np.log1p(2.0)], dtype=float)

    data_ind = pd.DataFrame({
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
    })
    signature = deck_interface.build_schedule_signature(data_ind)
    emulator = deck_interface.XGBPointwiseEmulator(
        DummyModel(),
        target_transform="signed_log1p",
        residual_mode=True,
        baseline_feature=True,
        baseline_cache={signature: np.array([10.0, 12.0], dtype=float)},
    )

    pred = emulator.predict_sequence(np.array([1.0, 2.0, 3.0, 4.0]), data_ind)

    np.testing.assert_allclose(pred, np.array([11.0, 10.0]), atol=1e-6)


def test_xgb_residual_pointwise_emulator_falls_back_to_real_for_unknown_schedule(monkeypatch):
    class DummyModel:
        def predict(self, features):
            raise AssertionError("No deberia invocarse el modelo cuando el schedule no esta en cache")

    real_pred = np.array([7.5, 6.5], dtype=float)

    def fake_real(params, data_ind):
        np.testing.assert_allclose(params, np.array([1.0, 2.0, 3.0, 4.0]))
        assert list(data_ind["TIME"]) == [0.0, 1.0]
        return real_pred

    monkeypatch.setattr(deck_interface, "original_run_deck", fake_real)

    data_ind = pd.DataFrame({
        "TIME": [0.0, 1.0],
        "Dose": [100.0, 0.0],
        "Duration": [1.0, 0.0],
    })
    emulator = deck_interface.XGBPointwiseEmulator(
        DummyModel(),
        target_transform="signed_log1p",
        residual_mode=True,
        baseline_feature=True,
        baseline_cache={},
        unknown_schedule_policy="fallback_real",
    )

    pred = emulator.predict_sequence(np.array([1.0, 2.0, 3.0, 4.0]), data_ind)

    np.testing.assert_allclose(pred, real_pred)


def test_load_emulator_supports_xgb_pointwise(tmp_path):
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    y = np.array([1.0, 1.0, 1.0], dtype=float)
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    model_path = tmp_path / "emulator.h5"
    joblib.dump(model, tmp_path / "emulator.joblib")
    with open(tmp_path / "emulator_meta.json", "w") as f:
        json.dump(
            {
                "backend": "xgb_pointwise",
                "max_len": 11,
                "n_params": 4,
                "target_transform": "signed_log1p",
            },
            f,
        )

    emulator, max_len = deck_interface.load_emulator(str(model_path), n_params=4)

    assert isinstance(emulator, deck_interface.XGBPointwiseEmulator)
    assert max_len == 11
    assert emulator.target_transform == "signed_log1p"


def test_load_emulator_supports_xgb_residual_pointwise(tmp_path):
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    y = np.array([1.0, 1.0, 1.0], dtype=float)
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    model_path = tmp_path / "emulator.h5"
    joblib.dump(model, tmp_path / "emulator.joblib")
    joblib.dump({"abc": np.array([1.0, 2.0], dtype=float)}, tmp_path / "emulator_baseline_cache.joblib")
    with open(tmp_path / "emulator_meta.json", "w") as f:
        json.dump(
            {
                "backend": "xgb_residual_pointwise",
                "max_len": 11,
                "n_params": 4,
                "target_transform": "signed_log1p",
                "residual_mode": True,
                "baseline_feature": True,
            },
            f,
        )

    emulator, max_len = deck_interface.load_emulator(str(model_path), n_params=4)

    assert isinstance(emulator, deck_interface.XGBPointwiseEmulator)
    assert max_len == 11
    assert emulator.residual_mode is True
    assert emulator.baseline_feature is True
    assert "abc" in emulator.baseline_cache
