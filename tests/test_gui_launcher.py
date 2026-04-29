from gui_launcher import (
    ANALYSIS_PRESETS,
    LauncherConfig,
    build_main_command,
    format_command,
    resolve_sparse_windows,
)


def test_build_main_command_for_surrogate_fit_includes_core_flags():
    cfg = LauncherConfig(
        exp_name="real_xgb_run",
        use_emulator=True,
        reuse_emulator_from=r"outputs\emu_xgb_residual_local_5k_peak",
        validate_emulator_etas=True,
    )

    cmd = build_main_command(cfg)
    joined = format_command(cmd)

    assert "main.py" in joined
    assert "--use_emulator" in joined
    assert "--reuse_emulator_from" in joined
    assert "--validate_emulator_etas" in joined
    assert "--exp_name real_xgb_run" in joined


def test_resolve_sparse_windows_uses_preset_mapping():
    cfg = LauncherConfig(sparse_preset="Peak-aware 2-point", sparse_windows="")

    assert resolve_sparse_windows(cfg) == "0-2,168-840"


def test_build_main_command_supports_bootstrap_and_sparse():
    cfg = LauncherConfig(
        bootstrap_kind="parametric",
        bootstrap_n_virtual="10",
        bootstrap_seed=202,
        bootstrap_balanced=True,
        sparse_preset="Generic 5-point",
        exp_name="sparse5_export",
    )

    cmd = build_main_command(cfg)
    joined = format_command(cmd)

    assert "--export_bootstrap parametric" in joined
    assert "--bootstrap_n_virtual 10" in joined
    assert "--bootstrap_seed 202" in joined
    assert "--bootstrap_balanced" in joined
    assert "--sparse_windows 0-24,24-168,168-336,336-672,672-840" in joined


def test_analysis_presets_cover_expected_common_flows():
    assert "Real data - exact fit" in ANALYSIS_PRESETS
    assert "Real data - surrogate-assisted fit" in ANALYSIS_PRESETS
    assert "Export sparse 5-point cohort" in ANALYSIS_PRESETS
