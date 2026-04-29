import os
import json

import numpy as np
import pandas as pd

from main import (
    count_measured_rows,
    export_bootstrap_dataset,
    load_saem_checkpoint,
    load_analysis_data,
    resolve_bootstrap_output_path,
    resolve_emulator_artifact_paths,
    resolve_n_burn,
    resolve_output_dir,
)
from config import ModelConfig


def test_resolve_n_burn_preserves_valid_value():
    assert resolve_n_burn(n_iter=100, requested_n_burn=20, default_n_burn=300) == 20


def test_resolve_n_burn_adjusts_short_runs():
    assert resolve_n_burn(n_iter=10, requested_n_burn=None, default_n_burn=300) == 5
    assert resolve_n_burn(n_iter=10, requested_n_burn=300, default_n_burn=300) == 5
    assert resolve_n_burn(n_iter=1, requested_n_burn=300, default_n_burn=300) == 0


def test_resolve_emulator_artifact_paths_defaults_to_output_dir():
    load_path, save_path, reuse_active = resolve_emulator_artifact_paths(
        output_dir="outputs/run_a",
        reuse_emulator_from=None,
        train_emulator=False,
    )
    assert load_path == os.path.join("outputs/run_a", "emulator.h5")
    assert save_path == os.path.join("outputs/run_a", "emulator.h5")
    assert reuse_active is False


def test_resolve_emulator_artifact_paths_supports_reuse_without_training():
    load_path, save_path, reuse_active = resolve_emulator_artifact_paths(
        output_dir="outputs/run_b",
        reuse_emulator_from="outputs/emu_good",
        train_emulator=False,
    )
    assert load_path == os.path.join("outputs/emu_good", "emulator.h5")
    assert save_path == os.path.join("outputs/run_b", "emulator.h5")
    assert reuse_active is True


def test_resolve_emulator_artifact_paths_ignores_reuse_when_training():
    load_path, save_path, reuse_active = resolve_emulator_artifact_paths(
        output_dir="outputs/run_c",
        reuse_emulator_from="outputs/emu_good",
        train_emulator=True,
    )
    assert load_path == os.path.join("outputs/run_c", "emulator.h5")
    assert save_path == os.path.join("outputs/run_c", "emulator.h5")
    assert reuse_active is False


def test_resolve_output_dir_uses_resume_dir_when_no_exp_name():
    assert resolve_output_dir(exp_name=None, resume_from_checkpoint="outputs/run_resume") == os.path.normpath("outputs/run_resume")


def test_load_saem_checkpoint_reads_partial_artifacts(tmp_path):
    checkpoint_dir = tmp_path / "resume_run"
    checkpoint_dir.mkdir()

    payload = {
        "status": "running",
        "iter": 7,
        "gamma": 0.5,
        "acc_rate": 0.2,
        "step_size": 0.12,
        "theta_hat": [1.0, 2.0, 3.0, 4.0],
        "Omega_hat": np.eye(4).tolist(),
        "sigma_hat": [0.3],
    }
    (checkpoint_dir / "final_estimates.partial.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "iter": 7,
            "gamma": 0.5,
            "acc_rate": 0.2,
            "step_size": 0.12,
            "theta1": 1.0,
            "theta2": 2.0,
            "theta3": 3.0,
            "theta4": 4.0,
            "omega_diag1": 1.0,
            "omega_diag2": 1.0,
            "omega_diag3": 1.0,
            "omega_diag4": 1.0,
            "sigma1": 0.3,
        }
    ]).to_csv(checkpoint_dir / "SAEM_trace.partial.csv", index=False)
    pd.DataFrame([
        {"ID": 1, "ETA1": 0.1, "ETA2": 0.2, "ETA3": 0.3, "ETA4": 0.4}
    ]).to_csv(checkpoint_dir / "ETAs_partial.csv", index=False)

    state = load_saem_checkpoint(str(checkpoint_dir), n_par=4, n_sigma=1)

    assert state["iter"] == 7
    assert state["status"] == "running"
    assert state["theta_hat"][3] == 4.0
    assert 1 in state["etas"]
    assert state["etas"][1][2] == 0.3
    assert state["trace"][-1]["iter"] == 7


def test_load_saem_checkpoint_falls_back_to_legacy_run_log(tmp_path):
    checkpoint_dir = tmp_path / "legacy_run"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "run.log").write_text(
        "2026-03-16 09:01:46,930 - saem - INFO - Iter 33: theta=[ 6.79921829 48.3859582  50.12543938 30.90836386], sigma=0.168, acc=0.00\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "Omega0": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )

    state = load_saem_checkpoint(str(checkpoint_dir), n_par=4, n_sigma=1)

    assert state["status"] == "legacy_bootstrap"
    assert state["iter"] == 33
    assert state["theta_hat"][0] == 6.79921829
    assert state["sigma_hat"][0] == 0.168
    assert state["trace"][-1]["acc_rate"] == 0.0


def test_load_analysis_data_drops_unnamed_and_filters_outl(tmp_path):
    data_path = tmp_path / "virtual_data.txt"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "ID": [1, 2],
            "TIME": [0.0, 1.0],
            "Dose": [10.0, 0.0],
            "Duration": [0.2, 0.0],
            "LNCONC": [1.0, 2.0],
            "OUTL": [0, 1],
        }
    ).to_csv(data_path, sep="\t", index=False)

    loaded = load_analysis_data(str(data_path))

    assert "Unnamed: 0" not in loaded.columns
    assert loaded.shape[0] == 1
    assert int(loaded["ID"].iloc[0]) == 1


def test_load_analysis_data_keeps_rows_when_outl_missing(tmp_path):
    data_path = tmp_path / "virtual_data_no_outl.txt"
    pd.DataFrame(
        {
            "ID": [1, 2],
            "TIME": [0.0, 1.0],
            "Dose": [10.0, 0.0],
            "Duration": [0.2, 0.0],
            "LNCONC": [1.0, 2.0],
        }
    ).to_csv(data_path, sep="\t", index=False)

    loaded = load_analysis_data(str(data_path))

    assert loaded.shape[0] == 2


def test_resolve_bootstrap_output_path_uses_default_name():
    output_path = resolve_bootstrap_output_path("outputs/run_a", "parametric")
    assert output_path == os.path.join("outputs/run_a", "bootstrap_parametric_virtual.txt")


def test_export_bootstrap_dataset_patient_writes_tabular_output(tmp_path):
    data = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "TIME": [0.0, 1.0, 0.0, 1.0],
            "Dose": [10.0, 0.0, 20.0, 0.0],
            "Duration": [0.2, 0.0, 0.3, 0.0],
            "LNCONC": [1.0, 0.8, 2.0, 1.6],
        }
    )
    output_path = tmp_path / "bootstrap_patient.txt"

    written_path, boot_df = export_bootstrap_dataset(
        data=data,
        ids=np.array([1, 2]),
        bootstrap_kind="patient",
        output_path=str(output_path),
        data_sep="\t",
        cfg=ModelConfig(),
        n_virtual=3,
        seed=123,
        balanced=False,
    )

    assert written_path == str(output_path)
    assert output_path.exists()
    loaded = pd.read_csv(output_path, sep="\t")
    assert loaded.shape[0] == boot_df.shape[0]
    assert set(loaded["BOOTSTRAP_KIND"].unique().tolist()) == {"patient"}


def test_export_bootstrap_dataset_parametric_uses_injected_generator(monkeypatch, tmp_path):
    captured = {}

    def fake_parametric_generator(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "ID": [1, 1],
                "TIME": [0.0, 1.0],
                "LNCONC": [1.5, 1.0],
                "BOOTSTRAP_KIND": ["parametric", "parametric"],
            }
        )

    monkeypatch.setattr("main.generate_parametric_bootstrap_population", fake_parametric_generator)
    output_path = tmp_path / "bootstrap_parametric.txt"

    written_path, boot_df = export_bootstrap_dataset(
        data=pd.DataFrame({"ID": [10], "TIME": [0.0], "Dose": [10.0], "Duration": [0.5], "LNCONC": [1.0]}),
        ids=np.array([10]),
        bootstrap_kind="parametric",
        output_path=str(output_path),
        data_sep="\t",
        cfg=ModelConfig(),
        theta=np.array([1.0, 2.0, 3.0, 4.0]),
        Omega=np.eye(4),
        sigma=np.array([0.2]),
        n_virtual=5,
        seed=321,
        balanced=True,
        include_residual_noise=False,
    )

    assert written_path == str(output_path)
    assert output_path.exists()
    assert boot_df.shape[0] == 2
    assert captured["n_virtual"] == 5
    assert captured["balanced"] is True
    assert captured["include_residual_noise"] is False


def test_export_bootstrap_dataset_applies_sparse_windows(monkeypatch, tmp_path):
    def fake_parametric_generator(**kwargs):
        return pd.DataFrame(
            {
                "ID": [1, 1, 1, 2, 2, 2],
                "TIME": [0.25, 24.0, 840.0, 0.25, 24.0, 840.0],
                "Dose": [10.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                "Duration": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "Meas": [1, 1, 1, 1, 1, 1],
                "LNCONC": [5.0, 4.0, 1.0, 5.5, 4.5, 1.5],
                "BOOTSTRAP_KIND": ["parametric"] * 6,
            }
        )

    monkeypatch.setattr("main.generate_parametric_bootstrap_population", fake_parametric_generator)
    output_path = tmp_path / "bootstrap_sparse.txt"

    _, boot_df = export_bootstrap_dataset(
        data=pd.DataFrame({"ID": [10], "TIME": [0.0], "Dose": [10.0], "Duration": [0.5], "LNCONC": [1.0]}),
        ids=np.array([10]),
        bootstrap_kind="parametric",
        output_path=str(output_path),
        data_sep="\t",
        cfg=ModelConfig(),
        theta=np.array([1.0, 2.0, 3.0, 4.0]),
        Omega=np.eye(4),
        sigma=np.array([0.2]),
        n_virtual=2,
        seed=321,
        sparse_windows=[(0.0, 100.0), (100.0, 900.0)],
        sparse_points_per_window=1,
        sparse_seed=7,
    )

    assert output_path.exists()
    assert count_measured_rows(boot_df) == 4
    assert "SPARSE_SELECTED" in boot_df.columns
