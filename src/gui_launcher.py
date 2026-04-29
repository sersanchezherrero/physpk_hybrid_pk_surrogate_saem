from __future__ import annotations

import os
import queue
import shlex
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "main.py"

SPARSE_PRESETS = {
    "None": "",
    "Generic 5-point": "0-24,24-168,168-336,336-672,672-840",
    "Generic 2-point": "0-120,168-840",
    "Peak-aware 5-point": "0-2,2-24,24-168,168-504,504-840",
    "Peak-aware 2-point": "0-2,168-840",
    "Custom": "",
}


@dataclass
class LauncherConfig:
    python_command: str = "py -3.11"
    data_file: str = "Monoclonal_Antibody.txt"
    data_sep: str = "\t"
    exp_name: str = ""
    init_method: str = "prev_run"
    prev_run_dir: str = r"outputs\bench_real_prev_run_80"
    prev_run_source: str = "final"
    n_iter: int = 40
    mstep_method: str = "powell"
    use_emulator: bool = False
    train_emulator: bool = False
    emulator_only: bool = False
    reuse_emulator_from: str = ""
    emulator_model_type: str = "xgb_residual_pointwise"
    validate_emulator_etas: bool = False
    use_empirical_eta_cov: bool = False
    bootstrap_kind: str = ""
    bootstrap_n_virtual: str = ""
    bootstrap_seed: int = 123
    bootstrap_balanced: bool = False
    bootstrap_without_residual_noise: bool = False
    sparse_preset: str = "None"
    sparse_windows: str = ""
    sparse_points_per_window: int = 1
    sparse_seed: int = 123
    extra_args: str = ""


ANALYSIS_PRESETS: dict[str, dict[str, Any]] = {
    "Real data - exact fit": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "reuse_emulator_from": "",
        "validate_emulator_etas": False,
        "bootstrap_kind": "",
        "bootstrap_n_virtual": "",
        "bootstrap_balanced": False,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "None",
        "sparse_windows": "",
        "n_iter": 40,
        "exp_name": "real_exact_run",
    },
    "Real data - surrogate-assisted fit": {
        "use_emulator": True,
        "train_emulator": False,
        "emulator_only": False,
        "reuse_emulator_from": r"outputs\emu_xgb_residual_local_5k_peak",
        "validate_emulator_etas": True,
        "emulator_model_type": "xgb_residual_pointwise",
        "bootstrap_kind": "",
        "bootstrap_n_virtual": "",
        "bootstrap_balanced": False,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "None",
        "sparse_windows": "",
        "n_iter": 40,
        "exp_name": "real_xgb_run",
    },
    "Export parametric bootstrap cohort": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "bootstrap_kind": "parametric",
        "bootstrap_n_virtual": "30",
        "bootstrap_balanced": False,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "None",
        "sparse_windows": "",
        "n_iter": 40,
        "exp_name": "bootstrap_parametric_export",
    },
    "Export balanced noisy cohort": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "bootstrap_kind": "parametric",
        "bootstrap_n_virtual": "30",
        "bootstrap_balanced": True,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "None",
        "sparse_windows": "",
        "n_iter": 40,
        "exp_name": "bootstrap_balanced_export",
    },
    "Export clean cohort": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "bootstrap_kind": "parametric",
        "bootstrap_n_virtual": "30",
        "bootstrap_balanced": False,
        "bootstrap_without_residual_noise": True,
        "sparse_preset": "None",
        "sparse_windows": "",
        "n_iter": 40,
        "exp_name": "bootstrap_clean_export",
    },
    "Export sparse 5-point cohort": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "bootstrap_kind": "parametric",
        "bootstrap_n_virtual": "10",
        "bootstrap_balanced": True,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "Generic 5-point",
        "sparse_windows": SPARSE_PRESETS["Generic 5-point"],
        "n_iter": 40,
        "exp_name": "sparse5_export",
    },
    "Export sparse 2-point cohort": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "bootstrap_kind": "parametric",
        "bootstrap_n_virtual": "10",
        "bootstrap_balanced": True,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "Generic 2-point",
        "sparse_windows": SPARSE_PRESETS["Generic 2-point"],
        "n_iter": 40,
        "exp_name": "sparse2_export",
    },
    "Export peak-aware sparse 5-point cohort": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "bootstrap_kind": "parametric",
        "bootstrap_n_virtual": "10",
        "bootstrap_balanced": True,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "Peak-aware 5-point",
        "sparse_windows": SPARSE_PRESETS["Peak-aware 5-point"],
        "n_iter": 40,
        "exp_name": "sparse5peak_export",
    },
    "Export peak-aware sparse 2-point cohort": {
        "use_emulator": False,
        "train_emulator": False,
        "emulator_only": False,
        "bootstrap_kind": "parametric",
        "bootstrap_n_virtual": "10",
        "bootstrap_balanced": True,
        "bootstrap_without_residual_noise": False,
        "sparse_preset": "Peak-aware 2-point",
        "sparse_windows": SPARSE_PRESETS["Peak-aware 2-point"],
        "n_iter": 40,
        "exp_name": "sparse2peak_export",
    },
}


def _append_flag(command: list[str], name: str, value: str | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    command.extend([name, text])


def resolve_sparse_windows(config: LauncherConfig) -> str:
    if config.sparse_preset == "Custom":
        return config.sparse_windows.strip()
    return SPARSE_PRESETS.get(config.sparse_preset, config.sparse_windows).strip()


def build_main_command(config: LauncherConfig) -> list[str]:
    python_parts = shlex.split(config.python_command, posix=False)
    if not python_parts:
        raise ValueError("Python command cannot be empty.")

    command = list(python_parts)
    command.append(str(MAIN_SCRIPT.name))

    _append_flag(command, "--data_file", config.data_file)
    if config.data_sep != "\t":
        _append_flag(command, "--data_sep", config.data_sep)
    _append_flag(command, "--init_method", config.init_method)
    _append_flag(command, "--prev_run_dir", config.prev_run_dir)
    _append_flag(command, "--prev_run_source", config.prev_run_source)
    _append_flag(command, "--n_iter", str(config.n_iter))
    _append_flag(command, "--mstep_method", config.mstep_method)
    _append_flag(command, "--exp_name", config.exp_name)

    if config.use_emulator:
        command.append("--use_emulator")
    if config.train_emulator:
        command.append("--train_emulator")
    if config.emulator_only:
        command.append("--emulator_only")
    if config.validate_emulator_etas:
        command.append("--validate_emulator_etas")
    if config.use_empirical_eta_cov:
        command.append("--use_empirical_eta_cov")

    if config.use_emulator or config.train_emulator or config.emulator_only or config.reuse_emulator_from.strip():
        _append_flag(command, "--emulator_model_type", config.emulator_model_type)
        _append_flag(command, "--reuse_emulator_from", config.reuse_emulator_from)

    if config.bootstrap_kind.strip():
        _append_flag(command, "--export_bootstrap", config.bootstrap_kind)
        _append_flag(command, "--bootstrap_n_virtual", config.bootstrap_n_virtual)
        _append_flag(command, "--bootstrap_seed", str(config.bootstrap_seed))
        if config.bootstrap_balanced:
            command.append("--bootstrap_balanced")
        if config.bootstrap_without_residual_noise:
            command.append("--bootstrap_without_residual_noise")

    sparse_windows = resolve_sparse_windows(config)
    if sparse_windows:
        _append_flag(command, "--sparse_windows", sparse_windows)
        _append_flag(command, "--sparse_points_per_window", str(config.sparse_points_per_window))
        _append_flag(command, "--sparse_seed", str(config.sparse_seed))

    extra_args = config.extra_args.strip()
    if extra_args:
        command.extend(shlex.split(extra_args, posix=False))

    return command


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


class PipelineLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PhysPK pipeline launcher")
        self.root.geometry("1220x860")
        self.root.minsize(1080, 760)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.analysis_preset = tk.StringVar(value="Real data - surrogate-assisted fit")
        self.vars: dict[str, tk.Variable] = {
            "python_command": tk.StringVar(value="py -3.11"),
            "data_file": tk.StringVar(value="Monoclonal_Antibody.txt"),
            "data_sep": tk.StringVar(value="\\t"),
            "exp_name": tk.StringVar(value="real_xgb_run"),
            "init_method": tk.StringVar(value="prev_run"),
            "prev_run_dir": tk.StringVar(value=r"outputs\bench_real_prev_run_80"),
            "prev_run_source": tk.StringVar(value="final"),
            "n_iter": tk.IntVar(value=40),
            "mstep_method": tk.StringVar(value="powell"),
            "use_emulator": tk.BooleanVar(value=True),
            "train_emulator": tk.BooleanVar(value=False),
            "emulator_only": tk.BooleanVar(value=False),
            "reuse_emulator_from": tk.StringVar(value=r"outputs\emu_xgb_residual_local_5k_peak"),
            "emulator_model_type": tk.StringVar(value="xgb_residual_pointwise"),
            "validate_emulator_etas": tk.BooleanVar(value=True),
            "use_empirical_eta_cov": tk.BooleanVar(value=False),
            "bootstrap_kind": tk.StringVar(value=""),
            "bootstrap_n_virtual": tk.StringVar(value=""),
            "bootstrap_seed": tk.IntVar(value=123),
            "bootstrap_balanced": tk.BooleanVar(value=False),
            "bootstrap_without_residual_noise": tk.BooleanVar(value=False),
            "sparse_preset": tk.StringVar(value="None"),
            "sparse_windows": tk.StringVar(value=""),
            "sparse_points_per_window": tk.IntVar(value=1),
            "sparse_seed": tk.IntVar(value=123),
            "extra_args": tk.StringVar(value=""),
        }

        self._build_ui()
        self._bind_preview_updates()
        self.apply_analysis_preset()
        self.root.after(150, self._drain_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=14)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        preset_frame = ttk.LabelFrame(main_frame, text="Analysis preset", padding=10)
        preset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        preset_frame.columnconfigure(1, weight=1)

        ttk.Label(preset_frame, text="Preset").grid(row=0, column=0, sticky="w", padx=(0, 8))
        preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.analysis_preset,
            values=list(ANALYSIS_PRESETS.keys()),
            state="readonly",
            width=44,
        )
        preset_combo.grid(row=0, column=1, sticky="ew")
        preset_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_analysis_preset())

        ttk.Button(preset_frame, text="Apply preset", command=self.apply_analysis_preset).grid(row=0, column=2, padx=(8, 0))

        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky="nsew")

        basic_tab = ttk.Frame(notebook, padding=12)
        advanced_tab = ttk.Frame(notebook, padding=12)
        run_tab = ttk.Frame(notebook, padding=12)
        notebook.add(basic_tab, text="Basic")
        notebook.add(advanced_tab, text="Advanced")
        notebook.add(run_tab, text="Run")

        self._build_basic_tab(basic_tab)
        self._build_advanced_tab(advanced_tab)
        self._build_run_tab(run_tab)

    def _build_basic_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        data_frame = ttk.LabelFrame(parent, text="Data and output", padding=10)
        data_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        data_frame.columnconfigure(1, weight=1)

        self._entry_with_browse(data_frame, 0, "Data file", "data_file", browse="file")
        self._entry_row(data_frame, 1, "Experiment name", "exp_name")
        self._entry_row(data_frame, 2, "Iterations", "n_iter")

        init_frame = ttk.LabelFrame(parent, text="Initialization and estimation", padding=10)
        init_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        init_frame.columnconfigure(1, weight=1)

        self._combo_row(init_frame, 0, "Init method", "init_method", ["default", "nca", "two_stage", "prev_run", "hybrid"])
        self._entry_with_browse(init_frame, 1, "Previous run dir", "prev_run_dir", browse="dir")
        self._combo_row(init_frame, 2, "Previous run source", "prev_run_source", ["auto", "final", "init"])
        self._combo_row(init_frame, 3, "M-step method", "mstep_method", ["powell", "lbfgsb", "hybrid"])

        emu_frame = ttk.LabelFrame(parent, text="Emulator options", padding=10)
        emu_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        emu_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(emu_frame, text="Use emulator inside SAEM", variable=self.vars["use_emulator"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(emu_frame, text="Validate emulator with sampled ETAs", variable=self.vars["validate_emulator_etas"]).grid(row=1, column=0, columnspan=2, sticky="w")
        self._entry_with_browse(emu_frame, 2, "Reuse emulator from", "reuse_emulator_from", browse="dir")
        self._combo_row(emu_frame, 3, "Emulator model", "emulator_model_type", ["xgb_residual_pointwise", "xgb_pointwise", "lstm", "transformer"])

        bootstrap_frame = ttk.LabelFrame(parent, text="Bootstrap and sparse options", padding=10)
        bootstrap_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        bootstrap_frame.columnconfigure(1, weight=1)

        self._combo_row(bootstrap_frame, 0, "Bootstrap export", "bootstrap_kind", ["", "patient", "parametric"])
        self._entry_row(bootstrap_frame, 1, "Bootstrap n_virtual", "bootstrap_n_virtual")
        self._entry_row(bootstrap_frame, 2, "Bootstrap seed", "bootstrap_seed")
        ttk.Checkbutton(bootstrap_frame, text="Balanced bootstrap", variable=self.vars["bootstrap_balanced"]).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(bootstrap_frame, text="Without residual noise", variable=self.vars["bootstrap_without_residual_noise"]).grid(row=4, column=0, columnspan=2, sticky="w")
        sparse_combo = self._combo_row(bootstrap_frame, 5, "Sparse preset", "sparse_preset", list(SPARSE_PRESETS.keys()))
        sparse_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_sparse_preset())
        self._entry_row(bootstrap_frame, 6, "Sparse windows", "sparse_windows")
        self._entry_row(bootstrap_frame, 7, "Points per window", "sparse_points_per_window")
        self._entry_row(bootstrap_frame, 8, "Sparse seed", "sparse_seed")

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        env_frame = ttk.LabelFrame(parent, text="Execution environment", padding=10)
        env_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        env_frame.columnconfigure(1, weight=1)

        self._entry_row(env_frame, 0, "Python command", "python_command")
        self._entry_row(env_frame, 1, "Data separator", "data_sep")
        ttk.Label(
            env_frame,
            text="Use py -3.11 by default. You can change it if another interpreter is needed.",
            wraplength=420,
            foreground="#52606D",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        adv_frame = ttk.LabelFrame(parent, text="Advanced toggles", padding=10)
        adv_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        adv_frame.columnconfigure(0, weight=1)

        ttk.Checkbutton(adv_frame, text="Train emulator in this run", variable=self.vars["train_emulator"]).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(adv_frame, text="Emulator-only mode", variable=self.vars["emulator_only"]).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(adv_frame, text="Use empirical ETA covariance", variable=self.vars["use_empirical_eta_cov"]).grid(row=2, column=0, sticky="w")

        extra_frame = ttk.LabelFrame(parent, text="Extra CLI arguments", padding=10)
        extra_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        extra_frame.columnconfigure(0, weight=1)

        ttk.Label(
            extra_frame,
            text="Optional free-form arguments for advanced cases not covered by the form, for example --force_use_emulator or --bootstrap_output_file outputs\\my_dataset.txt",
            wraplength=920,
            foreground="#52606D",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(extra_frame, textvariable=self.vars["extra_args"]).grid(row=1, column=0, sticky="ew")

    def _build_run_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        ttk.Label(parent, text="Generated command").grid(row=0, column=0, sticky="w")
        self.command_preview = ScrolledText(parent, height=5, wrap="word")
        self.command_preview.grid(row=1, column=0, sticky="ew", pady=(4, 10))

        button_row = ttk.Frame(parent)
        button_row.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(button_row, text="Refresh preview", command=self.update_preview).pack(side="left")
        ttk.Button(button_row, text="Run pipeline", command=self.run_pipeline).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Stop current run", command=self.stop_pipeline).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Open workspace folder", command=self.open_workspace).pack(side="left", padx=(8, 0))

        ttk.Label(parent, text="Run log").grid(row=3, column=0, sticky="nw")
        self.log_text = ScrolledText(parent, wrap="word")
        self.log_text.grid(row=4, column=0, sticky="nsew", pady=(4, 0))

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, key: str) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=self.vars[key])
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        return entry

    def _entry_with_browse(self, parent: ttk.Frame, row: int, label: str, key: str, browse: str) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=self.vars[key])
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        button = ttk.Button(parent, text="Browse", command=lambda: self.browse_path(key, browse))
        button.grid(row=row, column=2, padx=(8, 0), pady=4)
        return entry

    def _combo_row(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        return combo

    def _bind_preview_updates(self) -> None:
        for variable in self.vars.values():
            variable.trace_add("write", lambda *_args: self.update_preview())
        self.analysis_preset.trace_add("write", lambda *_args: self.update_preview())

    def browse_path(self, key: str, browse: str) -> None:
        if browse == "file":
            path = filedialog.askopenfilename(initialdir=str(ROOT))
        else:
            path = filedialog.askdirectory(initialdir=str(ROOT))
        if path:
            rel_path = os.path.relpath(path, start=ROOT)
            self.vars[key].set(rel_path if not rel_path.startswith("..") else path)

    def apply_sparse_preset(self) -> None:
        preset = self.vars["sparse_preset"].get()
        if preset != "Custom":
            self.vars["sparse_windows"].set(SPARSE_PRESETS.get(preset, ""))

    def apply_analysis_preset(self) -> None:
        preset_name = self.analysis_preset.get()
        updates = ANALYSIS_PRESETS.get(preset_name, {})
        for key, value in updates.items():
            if key == "sparse_preset":
                self.vars[key].set(value)
                self.apply_sparse_preset()
            else:
                self.vars[key].set(value)
        self.update_preview()

    def collect_config(self) -> LauncherConfig:
        raw = {key: var.get() for key, var in self.vars.items()}
        if raw["data_sep"] == r"\t":
            raw["data_sep"] = "\t"
        return LauncherConfig(**raw)

    def update_preview(self) -> None:
        try:
            command = build_main_command(self.collect_config())
            preview = format_command(command)
        except Exception as exc:  # pragma: no cover - GUI feedback
            preview = f"Error building command: {exc}"

        self.command_preview.configure(state="normal")
        self.command_preview.delete("1.0", tk.END)
        self.command_preview.insert(tk.END, preview)
        self.command_preview.configure(state="disabled")

    def run_pipeline(self) -> None:
        if self.process is not None:
            messagebox.showwarning("Run in progress", "A pipeline run is already active.")
            return

        try:
            command = build_main_command(self.collect_config())
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self.log_text.insert(tk.END, f"$ {format_command(command)}\n\n")
        self.log_text.see(tk.END)

        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.process = None
            messagebox.showerror("Launch failed", str(exc))
            return

        thread = threading.Thread(target=self._stream_process_output, daemon=True)
        thread.start()

    def _stream_process_output(self) -> None:
        assert self.process is not None
        try:
            if self.process.stdout is not None:
                for line in self.process.stdout:
                    self.log_queue.put(line)
        finally:
            return_code = self.process.wait()
            self.log_queue.put(f"\n[Process finished with exit code {return_code}]\n")
            self.process = None

    def _drain_log_queue(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
        self.root.after(150, self._drain_log_queue)

    def stop_pipeline(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        self.log_text.insert(tk.END, "\n[Termination requested]\n")
        self.log_text.see(tk.END)

    def open_workspace(self) -> None:
        try:
            os.startfile(str(ROOT))  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - OS dependent
            messagebox.showerror("Cannot open folder", str(exc))

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno("Close launcher", "A run is still active. Do you want to close the launcher anyway?"):
                return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = PipelineLauncherApp(root)
    app.update_preview()
    root.mainloop()


if __name__ == "__main__":
    main()
