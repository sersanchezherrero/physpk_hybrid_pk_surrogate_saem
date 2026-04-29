# config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_par: int = 4
    re_type: str = "log-additive"
    err_model: str = "add"
    transform: str = "identity"
    obs_column: str = "LNCONC"

    @property
    def n_sigma(self):
        """Número de parámetros de error según el modelo."""
        return 2 if self.err_model == "add+prop" else 1

@dataclass
class SAEMConfig:
    n_iter: int = 1000
    n_burn: int = 300
    n_mcmc: int = 20
    checkpoint_every: int = 1
    stop_after_iter: int | None = None
    step_size_eta: float = 0.12
    adapt_step: bool = True
    target_acc: float = 0.3
    use_neural_sstep: bool = True
    neural_start_after: int = 250
    neural_retrain_every: int = 50
    neural_clip_eta: float = 4.0
    retrain_emulator_every: int = 50   # Reentrenar cada 50 iteraciones (0 = no reentrenar)
    emulator_path: str = "emulator.h5" # Ruta donde se guarda el emulador
    n_virtual_retrain: int = 1000      # Nº de muestras virtuales para reentrenar
    use_ml_pop_warmstart: bool = False     # ML warm‑start poblacional
    ml_pop_train: bool = True
    ml_pop_path: str = "ml_pop_warmstart.pkl"
    n_virtual_pop_sets: int = 10
    n_ind_per_pop_set: int = 3
    conv_epsilon: float = 1e-4
    conv_window: int = 30
    mstep_method: str = "powell"   # "powell", "lbfgsb", "hybrid"
    use_empirical_eta_cov: bool = False
    emp_cov_update_freq: int = 10
    max_emp_cov_buffer: int = 2000
