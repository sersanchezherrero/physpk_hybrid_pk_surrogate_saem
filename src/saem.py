# saem.py
import json
import os
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import logging
from scipy.optimize import minimize
from ml_utils import (
    build_pk_features_from_individual,
    build_eta_posterior_net,
    train_eta_posterior_net,
    sample_from_eta_net,
    validate_emulator,
    train_emulator
)
from models import individual_params, residual_sd
from deck_interface import deck_executor as global_deck_executor

logger = logging.getLogger(__name__)

class SAEM:
    def __init__(self, data, ids, cfg_model, cfg_saem, theta0, Omega0, sigma0,
                 seed=123, emulator_path=None, retrain_emulator_every=None,
                 n_virtual_retrain=1000, deck_executor=None, emulator_train_kwargs=None,
                 theta_trust_center=None, theta_trust_log_radius=None,
                 checkpoint_dir=None, checkpoint_every=None, stop_after_iter=None):
        self.data = data
        self.ids = ids
        self.cfg_model = cfg_model
        self.cfg_saem = cfg_saem
        self.theta = np.asarray(theta0, dtype=float)
        self.Omega = np.asarray(Omega0, dtype=float)
        self.Omega_inv = np.linalg.inv(self.Omega)
        self.sigma = np.asarray(sigma0, dtype=float)
        self.n_par = cfg_model.n_par
        self.n_ids = len(ids)
        self.etas = {i_id: np.zeros(self.n_par) for i_id in ids}
        self.data_by_id = {i_id: data[data["ID"] == i_id].copy() for i_id in ids}
        self.trace = []
        self.rng = np.random.default_rng(seed)
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_every is None:
            checkpoint_every = getattr(cfg_saem, "checkpoint_every", 0)
        self.checkpoint_every = max(0, int(checkpoint_every))
        if stop_after_iter is None:
            stop_after_iter = getattr(cfg_saem, "stop_after_iter", None)
        self.stop_after_iter = None if stop_after_iter is None else max(0, int(stop_after_iter))
        self.stop_reason = None
        # Usar el ejecutor proporcionado o el global
        self.deck_executor = deck_executor if deck_executor is not None else global_deck_executor

        # Para S‑step neuronal
        self.nn = None
        self.nn_scaler = None
        self.nn_buffer_X = []
        self.nn_buffer_mu = []

        # Para reentrenamiento del emulador
        self.retrain_emulator_every = retrain_emulator_every
        self.emulator_path = emulator_path
        self.n_virtual_retrain = n_virtual_retrain
        self.emulator_train_kwargs = dict(emulator_train_kwargs or {})

        # Para SAEM adaptativo
        self.step_size_rw = float(cfg_saem.step_size_eta)
        self.adapt_cov = Omega0.copy()
        self.adapt_step = 0.05
        self.target_acc = cfg_saem.target_acc
        self.epsilon_conv = getattr(cfg_saem, 'epsilon_conv', 1e-4)
        self.window_conv = getattr(cfg_saem, 'window_conv', 30)
        self.mstep_method = str(getattr(cfg_saem, 'mstep_method', 'powell')).lower()
        self.use_empirical_eta_cov = bool(getattr(cfg_saem, 'use_empirical_eta_cov', False))
        self.emp_cov_update_freq = max(1, int(getattr(cfg_saem, 'emp_cov_update_freq', 10)))
        self.max_emp_cov_buffer = max(self.n_par + 1, int(getattr(cfg_saem, 'max_emp_cov_buffer', 2000)))
        self.emp_cov = np.asarray(Omega0, dtype=float).copy()
        self.emp_cov_buffer = []
        self.theta_trust_center = None
        self.theta_trust_center_log = None
        self.theta_trust_log_radius = None
        if theta_trust_center is not None and theta_trust_log_radius is not None:
            radius = float(theta_trust_log_radius)
            if radius > 0.0:
                self.theta_trust_center = np.asarray(theta_trust_center, dtype=float)
                self.theta_trust_center_log = np.log(np.clip(self.theta_trust_center, 1e-8, None))
                self.theta_trust_log_radius = radius
                theta_projected = self._project_theta_to_trust_region(self.theta)
                if not np.allclose(theta_projected, self.theta):
                    logger.info(
                        "theta0 proyectado al trust region del emulador: %s -> %s",
                        self.theta,
                        theta_projected,
                    )
                    self.theta = theta_projected

    def trace_as_dataframe(self):
        rows = []
        for item in self.trace:
            row = {
                "iter": int(item["iter"]),
                "gamma": float(item["gamma"]),
                "acc_rate": float(item["acc_rate"]),
                "step_size": float(item["step_size"]),
            }
            theta = np.asarray(item.get("theta", []), dtype=float).reshape(-1)
            omega_diag = np.asarray(item.get("omega_diag", []), dtype=float).reshape(-1)
            sigma = np.asarray(item.get("sigma", []), dtype=float).reshape(-1)
            for idx, value in enumerate(theta, start=1):
                row[f"theta{idx}"] = float(value)
            for idx, value in enumerate(omega_diag, start=1):
                row[f"omega_diag{idx}"] = float(value)
            for idx, value in enumerate(sigma, start=1):
                row[f"sigma{idx}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)

    def etas_as_dataframe(self):
        rows = []
        for i_id in self.ids:
            eta = np.asarray(self.etas.get(i_id, np.zeros(self.n_par)), dtype=float).reshape(-1)
            row = {"ID": int(i_id)}
            for idx, value in enumerate(eta, start=1):
                row[f"ETA{idx}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)

    def _checkpoint_payload(self, status):
        last_trace = self.trace[-1] if self.trace else {}
        return {
            "status": str(status),
            "iter": int(last_trace.get("iter", 0)),
            "gamma": float(last_trace.get("gamma", np.nan)) if last_trace else None,
            "acc_rate": float(last_trace.get("acc_rate", np.nan)) if last_trace else None,
            "step_size": float(last_trace.get("step_size", np.nan)) if last_trace else None,
            "theta_hat": np.asarray(self.theta, dtype=float).tolist(),
            "Omega_hat": np.asarray(self.Omega, dtype=float).tolist(),
            "sigma_hat": np.asarray(self.sigma, dtype=float).tolist(),
        }

    @staticmethod
    def _atomic_write_text(path, text):
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_path, path)

    def save_checkpoint(self, force=False, status="running"):
        if not self.checkpoint_dir:
            return
        if not force:
            if self.checkpoint_every <= 0 or not self.trace:
                return
            if int(self.trace[-1]["iter"]) % self.checkpoint_every != 0:
                return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "final_estimates.partial.json")
        trace_path = os.path.join(self.checkpoint_dir, "SAEM_trace.partial.csv")
        etas_path = os.path.join(self.checkpoint_dir, "ETAs_partial.csv")

        payload = self._checkpoint_payload(status=status)
        self._atomic_write_text(checkpoint_path, json.dumps(payload, indent=4, default=str))
        self._atomic_write_text(trace_path, self.trace_as_dataframe().to_csv(index=False))
        self._atomic_write_text(etas_path, self.etas_as_dataframe().to_csv(index=False))

    def _loglik_individual(self, theta, eta, sigma, data_ind):
        pars = individual_params(theta, eta, self.cfg_model.re_type)
        pred_all = self.deck_executor.run(pars, data_ind)   # <--- USAMOS EL EJECUTOR
        mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
        y_obs = data_ind.loc[mask, self.cfg_model.obs_column].values
        pred = pred_all[mask]
        if len(y_obs) == 0:
            return 0.0
        if self.cfg_model.transform == "log":
            y_t = np.log(np.clip(y_obs, 1e-12, None))
            pred_t = np.log(np.clip(pred, 1e-12, None))
        else:
            y_t, pred_t = y_obs, pred
        sd = residual_sd(pred_t, sigma, self.cfg_model.err_model, self.cfg_model.transform)
        resid = (y_t - pred_t) / sd
        return -0.5 * np.sum(resid**2 + 2*np.log(sd) + np.log(2*np.pi))

    def _neg_log_post_eta(self, eta, theta, Omega_inv, sigma, data_ind):
        ll = self._loglik_individual(theta, eta, sigma, data_ind)
        if not np.isfinite(ll):
            return 1e12
        prior = -0.5 * (eta @ Omega_inv @ eta)
        return -(ll + prior)

    def _sample_eta_mcmc(self, eta0, data_ind, step_size, n_steps, use_nn=False):
        eta = eta0.copy()
        accept = 0
        feats = None
        if not use_nn:
            prop_base = self.emp_cov if self.use_empirical_eta_cov else self.Omega
            prop_cov = step_size * prop_base + 1e-6 * np.eye(self.n_par)
            try:
                L = cholesky(prop_cov, lower=True)
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.diag(prop_cov)))
        else:
            feats = self._extract_features(data_ind)
        f_current = self._neg_log_post_eta(eta, self.theta, self.Omega_inv, self.sigma, data_ind)

        for _ in range(n_steps):
            if use_nn:
                eta_prop = sample_from_eta_net(self.nn, feats, self.rng, scaler=self.nn_scaler)
                log_q_cur = self._log_prob_eta_net(eta, feats=feats)
                log_q_prop = self._log_prob_eta_net(eta_prop, feats=feats)
            else:
                eta_prop = eta + L @ self.rng.normal(size=self.n_par)
                log_q_cur = 0.0
                log_q_prop = 0.0

            f_prop = self._neg_log_post_eta(eta_prop, self.theta, self.Omega_inv, self.sigma, data_ind)
            if np.isfinite(f_prop) and np.isfinite(f_current):
                log_accept = -(f_prop - f_current) + (log_q_prop - log_q_cur)
                if np.log(self.rng.random()) < log_accept:
                    eta = eta_prop
                    f_current = f_prop
                    accept += 1
        return eta, accept

    def fit(self):
        step_size = self.cfg_saem.step_size_eta
        self.stop_reason = None
        if self.trace:
            last_trace = self.trace[-1]
            start_iter = max(1, int(last_trace.get("iter", 0)) + 1)
            last_step_size = last_trace.get("step_size", step_size)
            if np.isfinite(last_step_size):
                step_size = float(last_step_size)
            self.save_checkpoint(force=True, status="resumed")
        else:
            start_iter = 1
            self.save_checkpoint(force=True, status="initialized")
        if start_iter > self.cfg_saem.n_iter:
            logger.info("No hay iteraciones pendientes: el checkpoint ya llega a iter %d.", start_iter - 1)
            self.save_checkpoint(force=True, status="completed")
            return self.theta, self.Omega, self.sigma, self.etas
        for k in range(start_iter, self.cfg_saem.n_iter + 1):
            # Gamma
            if k <= self.cfg_saem.n_burn:
                gamma_k = 1.0
            else:
                gamma_k = 1.0 / (k - self.cfg_saem.n_burn)

            # ---- S‑step: muestrear ETAs ----
            accepts = 0
            proposals = 0
            new_etas = {}
            for i_id in self.ids:
                data_ind = self.data_by_id[i_id]
                eta0 = self.etas[i_id]
                use_nn = (self.cfg_saem.use_neural_sstep and
                          self.nn is not None and
                          k >= self.cfg_saem.neural_start_after and
                          (k % self.cfg_saem.neural_retrain_every != 0))
                eta_new, acc = self._sample_eta_mcmc(
                    eta0, data_ind, step_size, self.cfg_saem.n_mcmc, use_nn=use_nn
                )
                new_etas[i_id] = eta_new
                accepts += acc
                proposals += self.cfg_saem.n_mcmc
                # Acumular para entrenar la red
                if self.cfg_saem.use_neural_sstep and k >= self.cfg_saem.neural_start_after:
                    self.nn_buffer_X.append(self._extract_features(data_ind))
                    self.nn_buffer_mu.append(eta_new)

            self.etas = new_etas
            if self.use_empirical_eta_cov:
                self._update_empirical_eta_cov_buffer(k)

            # Adaptar paso del MCMC
            if k <= self.cfg_saem.n_burn and proposals > 0:
                acc_rate = accepts / proposals
                if acc_rate < 0.15:
                    step_size *= 0.8
                elif acc_rate > 0.4:
                    step_size *= 1.2
                step_size = np.clip(step_size, 0.01, 2.0)

            # ---- Reentrenar red neuronal si toca ----
            if (self.cfg_saem.use_neural_sstep and
                k >= self.cfg_saem.neural_start_after and
                k % self.cfg_saem.neural_retrain_every == 0 and
                len(self.nn_buffer_X) > 100):
                self._train_neural_posterior()

            # ---- SA‑step: actualizar Omega ----
            eta_mat = np.array(list(self.etas.values()))
            S_omega = eta_mat.T @ eta_mat / self.n_ids
            self.Omega = self.Omega + gamma_k * (S_omega - self.Omega)
            self.Omega = 0.5 * (self.Omega + self.Omega.T)
            diag = np.diag(self.Omega).copy()
            diag = np.clip(diag, 1e-6, None)
            np.fill_diagonal(self.Omega, diag)
            try:
                self.Omega_inv = np.linalg.inv(self.Omega)
            except np.linalg.LinAlgError:
                logger.warning("Iter %d: Omega no invertible, se usa pseudoinversa.", k)
                self.Omega_inv = np.linalg.pinv(self.Omega)

            # ---- SA‑step: actualizar sigma según modelo de error ----
            if self.cfg_model.err_model == "add":
                rss = 0.0
                nobs = 0
                for i_id in self.ids:
                    data_ind = self.data_by_id[i_id]
                    eta = self.etas[i_id]
                    pars = individual_params(self.theta, eta, self.cfg_model.re_type)
                    pred_all = self.deck_executor.run(pars, data_ind)
                    mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
                    y_obs = data_ind.loc[mask, self.cfg_model.obs_column].values
                    pred = pred_all[mask]
                    if self.cfg_model.transform == "log":
                        y_t = np.log(np.clip(y_obs, 1e-12, None))
                        pred_t = np.log(np.clip(pred, 1e-12, None))
                    else:
                        y_t, pred_t = y_obs, pred
                    resid = y_t - pred_t
                    rss += np.sum(resid**2)
                    nobs += len(resid)
                if nobs > 0:
                    sigma2_hat = rss / nobs
                    self.sigma[0] = np.sqrt(sigma2_hat)

            elif self.cfg_model.err_model == "prop":
                sum_sq_prop = 0.0
                nobs = 0
                for i_id in self.ids:
                    data_ind = self.data_by_id[i_id]
                    eta = self.etas[i_id]
                    pars = individual_params(self.theta, eta, self.cfg_model.re_type)
                    pred_all = self.deck_executor.run(pars, data_ind)
                    mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
                    y_obs = data_ind.loc[mask, self.cfg_model.obs_column].values
                    pred = pred_all[mask]
                    if self.cfg_model.transform == "log":
                        y_t = np.log(np.clip(y_obs, 1e-12, None))
                        pred_t = np.log(np.clip(pred, 1e-12, None))
                    else:
                        y_t, pred_t = y_obs, pred
                    resid = y_t - pred_t
                    if self.cfg_model.transform == "identity":
                        with np.errstate(divide='ignore', invalid='ignore'):
                            weighted_resid = resid / np.abs(pred_t)
                    else:
                        weighted_resid = resid
                    valid = np.isfinite(weighted_resid)
                    weighted_resid = weighted_resid[valid]
                    sum_sq_prop += np.sum(weighted_resid**2)
                    nobs += len(weighted_resid)
                if nobs > 0:
                    sigma2_hat = sum_sq_prop / nobs
                    self.sigma[0] = np.sqrt(sigma2_hat)

            elif self.cfg_model.err_model == "add+prop":
                def neg_log_lik(sigma_vec):
                    sigma_a, sigma_p = sigma_vec
                    ll = 0.0
                    for i_id in self.ids:
                        data_ind = self.data_by_id[i_id]
                        eta = self.etas[i_id]
                        pars = individual_params(self.theta, eta, self.cfg_model.re_type)
                        pred_all = self.deck_executor.run(pars, data_ind)
                        mask = data_ind["Meas"] != 0 if "Meas" in data_ind.columns else np.ones(len(data_ind), bool)
                        y_obs = data_ind.loc[mask, self.cfg_model.obs_column].values
                        pred = pred_all[mask]
                        if self.cfg_model.transform == "log":
                            y_t = np.log(np.clip(y_obs, 1e-12, None))
                            pred_t = np.log(np.clip(pred, 1e-12, None))
                        else:
                            y_t, pred_t = y_obs, pred
                        sd = residual_sd(
                            pred_t,
                            np.array([sigma_a, sigma_p], dtype=float),
                            "add+prop",
                            self.cfg_model.transform
                        )
                        resid = y_t - pred_t
                        ll += -0.5 * np.sum( resid**2 / sd**2 + 2*np.log(sd) + np.log(2*np.pi) )
                    return -ll
                bounds = [(1e-6, None), (1e-6, None)]
                res = minimize(neg_log_lik, self.sigma, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
                self.sigma = res.x

            # ---- M‑step para theta (Powell con paso estocástico) ----
            self._mstep_theta(k, gamma_k)

            # ---- Guardar traza ----
            acc_rate = accepts / proposals if proposals > 0 else np.nan
            self.trace.append({
                "iter": k,
                "gamma": gamma_k,
                "theta": self.theta.copy(),
                "omega_diag": np.diag(self.Omega).copy(),
                "sigma": self.sigma.copy(),
                "acc_rate": acc_rate,
                "step_size": step_size
            })
            self.save_checkpoint(status="running")
            logger.info("Iter %d: theta=%s, sigma=%.3f, acc=%.2f", k, self.theta, self.sigma[0], acc_rate)

            if (
                self.stop_after_iter is not None
                and self.stop_after_iter > 0
                and self.stop_after_iter < self.cfg_saem.n_iter
                and k >= self.stop_after_iter
            ):
                self.stop_reason = "stop_after_iter"
                logger.info(
                    "Parada controlada de SAEM en iter %d por stop_after_iter=%d.",
                    k,
                    self.stop_after_iter,
                )
                self.save_checkpoint(force=True, status="stopped_after_iter")
                return self.theta, self.Omega, self.sigma, self.etas

            # ---- Criterio de convergencia ----
            if k > self.cfg_saem.n_burn and k >= self.cfg_saem.conv_window:
                thetas = np.array([t['theta'] for t in self.trace[-self.cfg_saem.conv_window:]])
                rel_change = np.mean(np.abs(np.diff(thetas, axis=0)) / (np.abs(thetas[:-1]) + 1e-8))
                if rel_change < self.cfg_saem.conv_epsilon:
                    logger.info("Convergencia alcanzada en iteración %d", k)
                    break

            # ---- Reentrenar emulador si toca ----
            if (self.retrain_emulator_every is not None and
                self.retrain_emulator_every > 0 and
                k >= self.cfg_saem.n_burn and
                k % self.retrain_emulator_every == 0):
                self._retrain_emulator()

        self.save_checkpoint(force=True, status="completed")
        return self.theta, self.Omega, self.sigma, self.etas

    def _extract_features(self, data_ind):
        return build_pk_features_from_individual(data_ind, self.cfg_model.obs_column)

    def _theta_objective(self, theta):
        total = 0.0
        for i_id in self.ids:
            data_ind = self.data_by_id[i_id]
            eta = self.etas[i_id]
            ll = self._loglik_individual(theta, eta, self.sigma, data_ind)
            total -= ll
        return total if np.isfinite(total) else 1e12

    def _theta_objective_log(self, phi, phi_ref=None):
        theta = np.exp(phi)
        total = 0.0
        for i_id in self.ids:
            data_ind = self.data_by_id[i_id]
            try:
                ll = self._loglik_individual(theta, self.etas[i_id], self.sigma, data_ind)
            except Exception:
                penalty = 0.0 if phi_ref is None else 1000.0 * np.sum((phi - phi_ref) ** 2)
                return 1e12 + penalty
            if not np.isfinite(ll):
                penalty = 0.0 if phi_ref is None else 1000.0 * np.sum((phi - phi_ref) ** 2)
                return 1e12 + penalty
            total -= ll
        return total if np.isfinite(total) else 1e12

    def _trust_region_bounds_log(self):
        if self.theta_trust_center_log is None or self.theta_trust_log_radius is None:
            return None
        lower = np.maximum(-8.0, self.theta_trust_center_log - self.theta_trust_log_radius)
        upper = np.minimum(8.0, self.theta_trust_center_log + self.theta_trust_log_radius)
        return lower, upper

    def _project_theta_to_trust_region(self, theta):
        theta = np.asarray(theta, dtype=float)
        trust_bounds = self._trust_region_bounds_log()
        if trust_bounds is None:
            return theta
        lower, upper = trust_bounds
        phi = np.log(np.clip(theta, 1e-8, None))
        phi = np.clip(phi, lower, upper)
        return np.exp(phi)

    def _mstep_theta(self, k, gamma_k):
        phi0 = np.log(np.clip(self.theta, 1e-8, None))
        trust_bounds = self._trust_region_bounds_log()
        if trust_bounds is not None:
            trust_lower, trust_upper = trust_bounds
            phi_anchor = np.clip(phi0, trust_lower, trust_upper)
        else:
            phi_anchor = phi0.copy()
        local_radius = 1.5
        lower = np.maximum(-8.0, phi_anchor - local_radius)
        upper = np.minimum(8.0, phi_anchor + local_radius)
        if trust_bounds is not None:
            lower = np.maximum(lower, trust_lower)
            upper = np.minimum(upper, trust_upper)
        bounds = list(zip(lower, upper))

        if self.mstep_method == "hybrid":
            if k <= self.cfg_saem.n_burn:
                method = "Powell"
                options = {"maxiter": 20, "ftol": 1e-4, "disp": False}
            else:
                method = "L-BFGS-B"
                options = {"maxiter": 10, "ftol": 1e-6}
        elif self.mstep_method == "powell":
            method = "Powell"
            options = {"maxiter": 20, "ftol": 1e-4, "disp": False}
        else:
            method = "L-BFGS-B"
            options = {"maxiter": 10, "ftol": 1e-6}

        res = minimize(
            lambda phi: self._theta_objective_log(phi, phi_ref=phi_anchor),
            phi_anchor,
            method=method,
            bounds=bounds,
            options=options
        )
        phi_new = res.x
        theta_new = np.exp(phi_new)
        negL_new = self._theta_objective_log(phi_new, phi_ref=phi_anchor)

        if np.isfinite(negL_new) and np.all(np.isfinite(theta_new)):
            theta_candidate = self.theta + gamma_k * (theta_new - self.theta)
            self.theta = self._project_theta_to_trust_region(theta_candidate)
        else:
            logger.debug(
                "Iter %d: theta update descartado por resultado no finito (negL_new=%.4f)",
                k, negL_new
            )

    def _log_prob_eta_net(self, eta, data_ind=None, feats=None):
        """Log densidad de eta bajo la gaussiana predicha por la red neuronal."""
        if self.nn is None:
            return -np.inf
        if feats is None:
            if data_ind is None:
                raise ValueError("Se requiere data_ind o feats para evaluar la densidad de la red.")
            feats = self._extract_features(data_ind)
        if self.nn_scaler is not None:
            feats_scaled = self.nn_scaler.transform(feats.reshape(1, -1))
        else:
            feats_scaled = feats.reshape(1, -1)
        out = self.nn(feats_scaled, training=False).numpy().ravel()
        n_par = out.shape[0] // 2
        mu = out[:n_par]
        logvar = np.clip(out[n_par:], -10.0, 4.0)
        sigma = np.clip(np.exp(0.5 * logvar), 1e-6, None)
        return -0.5 * np.sum(((eta - mu) / sigma)**2 + np.log(2 * np.pi) + 2 * np.log(sigma))

    def _update_empirical_eta_cov_buffer(self, k):
        for eta in self.etas.values():
            self.emp_cov_buffer.append(np.asarray(eta, dtype=float).copy())
        if len(self.emp_cov_buffer) > self.max_emp_cov_buffer:
            self.emp_cov_buffer = self.emp_cov_buffer[-self.max_emp_cov_buffer:]
        if k % self.emp_cov_update_freq == 0:
            self._refresh_empirical_eta_cov()

    def _refresh_empirical_eta_cov(self):
        if len(self.emp_cov_buffer) < self.n_par + 1:
            return
        eta_array = np.asarray(self.emp_cov_buffer, dtype=float)
        emp_cov_new = np.cov(eta_array.T)
        if emp_cov_new.shape != self.emp_cov.shape or not np.all(np.isfinite(emp_cov_new)):
            logger.debug("Se descarta una actualizacion no finita de la covarianza empirica de eta.")
            return
        emp_cov_new = 0.5 * (emp_cov_new + emp_cov_new.T)
        diag = np.clip(np.diag(emp_cov_new), 1e-6, None)
        np.fill_diagonal(emp_cov_new, diag)
        self.emp_cov = 0.8 * emp_cov_new + 0.2 * self.emp_cov
        self.emp_cov = 0.5 * (self.emp_cov + self.emp_cov.T) + 1e-6 * np.eye(self.n_par)

    def _train_neural_posterior(self):
        X = np.vstack(self.nn_buffer_X)
        mu = np.vstack(self.nn_buffer_mu)
        from sklearn.preprocessing import StandardScaler
        self.nn_scaler = StandardScaler()
        X_scaled = self.nn_scaler.fit_transform(X)
        if self.nn is None:
            self.nn = build_eta_posterior_net(X.shape[1], self.n_par)
        self.nn = train_eta_posterior_net(self.nn, X_scaled, mu, epochs=30)
        self.nn_buffer_X = []
        self.nn_buffer_mu = []

    def _retrain_emulator(self):
        logger.info("Reentrenando emulador con theta=%s, Omega diag=%s, sigma=%s",
                    self.theta, np.diag(self.Omega), self.sigma)
        retrain_kwargs = {
            "validation_split": 0.15,
            "epochs": 20,
            "batch_size": 32,
        }
        retrain_kwargs.update(self.emulator_train_kwargs)

        new_emulator, new_max_len, new_scaler_y = train_emulator(
            data=self.data,
            ids=self.ids,
            theta_pop=self.theta,
            Omega=self.Omega,
            sigma=self.sigma,
            cfg=self.cfg_model,
            n_virtual=self.n_virtual_retrain,
            model_path=self.emulator_path,
            force_retrain=True,
            seed=self.rng.integers(0, 2**31),
            **retrain_kwargs
        )
        # Usar el ejecutor para activar el emulador
        self.deck_executor.use_emulator(new_emulator, new_max_len, new_scaler_y)
        logger.info("Emulador reentrenado y activado.")

        # Validación rápida
        val_results = validate_emulator(
            new_emulator, new_max_len, new_scaler_y,
            self.data, self.ids, self.theta, self.sigma, self.cfg_model, n_val=10
        )
        logger.info("Validación post‑reentrenamiento: %s", val_results)
        if val_results["mean_rel_error"] > 0.15:
            logger.warning("Error relativo medio alto (%.2f), considera reducir retrain_emulator_every",
                           val_results["mean_rel_error"])
