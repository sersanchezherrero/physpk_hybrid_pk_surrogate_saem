import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import individual_params, residual_sd

def test_individual_params_log_additive():
    theta = np.array([2.0, 40.0, 20.0, 20.0])
    eta = np.array([0.1, -0.2, 0.0, 0.3])
    result = individual_params(theta, eta, "log-additive")
    expected = theta * np.exp(eta)
    np.testing.assert_allclose(result, expected)

def test_residual_sd_additive():
    pred = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1])
    sd = residual_sd(pred, sigma, "add", "identity")
    np.testing.assert_allclose(sd, [0.1, 0.1, 0.1])

def test_residual_sd_prop_identity():
    pred = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1])
    sd = residual_sd(pred, sigma, "prop", "identity")
    expected = 0.1 * np.abs(pred)
    np.testing.assert_allclose(sd, expected)

def test_residual_sd_prop_log():
    pred = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1])
    sd = residual_sd(pred, sigma, "prop", "log")
    # En log, el error proporcional se vuelve aditivo
    np.testing.assert_allclose(sd, [0.1, 0.1, 0.1])
