import pytest
import numpy as np
from src.utils.metrics import compute_ber, compute_tau_convergence, compute_fse_metrics


class TestComputeBER:
    def test_perfect_match(self):
        a = np.array([1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1])
        ber, errors, shift = compute_ber(a, a)
        assert ber == 0.0
        assert errors == 0
        assert shift == 0

    def test_all_errors(self):
        a_tx = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1])
        a_rx = -a_tx.copy()
        ber, errors, shift = compute_ber(a_tx, a_rx, max_shift=0)
        assert ber == 1.0
        assert errors == len(a_tx)

    def test_partial_errors(self):
        a_tx = np.array([1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1])
        a_rx = a_tx.copy()
        a_rx[3] = -a_rx[3]
        a_rx[7] = -a_rx[7]
        ber, errors, shift = compute_ber(a_tx, a_rx, max_shift=0)  # No shift search
        assert errors == 2
        assert ber == 2.0 / len(a_tx)

    def test_shift_alignment(self):
        a_tx = np.array([1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1])
        a_rx = a_tx[:8]
        ber, errors, shift = compute_ber(a_tx, a_rx)
        assert shift == 0
        assert errors == 0

    def test_empty_sequences(self):
        ber, errors, shift = compute_ber(np.array([]), np.array([]))
        assert ber == 0.0


class TestTauConvergence:
    def test_constant_tau(self):
        tau = np.ones(100) * 0.5
        metrics = compute_tau_convergence(tau)
        assert abs(metrics['tau_steady_mean'] - 0.5) < 1e-10
        assert metrics['tau_steady_std'] == 0.0

    def test_converging_tau(self):
        tau = np.linspace(1.0, 0.1, 200)
        metrics = compute_tau_convergence(tau, window=20)
        assert abs(metrics['tau_steady_mean'] - 0.1) < 0.05
        assert abs(metrics['tau_final'] - 0.1) < 0.01

    def test_empty_tau(self):
        metrics = compute_tau_convergence(np.array([]))
        assert metrics['tau_final'] == 0.0

    def test_tau_range(self):
        tau = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        metrics = compute_tau_convergence(tau)
        assert abs(metrics['tau_range'] - 0.7) < 1e-10


class TestFSEMetrics:
    def test_converging_mse(self):
        errors = np.concatenate([np.ones(100) * 1.0, np.ones(100) * 0.01])
        metrics = compute_fse_metrics(errors)
        assert metrics['mse_initial'] > metrics['mse_final']
        assert metrics['convergence_ratio'] < 1.0

    def test_empty_errors(self):
        metrics = compute_fse_metrics(np.array([]))
        assert metrics['mse_initial'] == 0.0

    def test_zero_errors(self):
        metrics = compute_fse_metrics(np.zeros(100))
        assert metrics['mse_initial'] == 0.0
        assert metrics['mse_final'] == 0.0
