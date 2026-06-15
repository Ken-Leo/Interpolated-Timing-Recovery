import pytest
import numpy as np
from src.opsps_va.ted import early_late_ted, get_oversampled_samples, mueller_muller_ted


class TestEarlyLateTED:
    def test_zero_error_symmetric(self):
        # Symmetric: early == late => zero error
        eps = early_late_ted(1.0, 1.0, 1.0)
        assert eps == 0.0

    def test_positive_error_positive_bit(self):
        # Positive bit, early > late => positive error
        eps = early_late_ted(1.5, 0.5, 1.0)
        assert eps > 0.0

    def test_negative_error_positive_bit(self):
        # Positive bit, early < late => negative error
        eps = early_late_ted(0.5, 1.5, 1.0)
        assert eps < 0.0

    def test_polarity_flip_negative_bit(self):
        # Negative bit flips error sign
        eps_pos = early_late_ted(1.5, 0.5, 1.0)
        eps_neg = early_late_ted(1.5, 0.5, -1.0)
        assert eps_neg == -eps_pos

    def test_zero_bit_error(self):
        # d_hat = 0 => zero error regardless of samples
        eps = early_late_ted(1.5, 0.5, 0.0)
        assert eps == 0.0


class TestGetOversampledSamples:
    def setup_method(self):
        self.signal = np.zeros(200)
        for i in range(200):
            self.signal[i] = np.exp(-((i - 100) ** 2) / 200.0)
        self.sps = 100

    def test_center_sample_at_peak(self):
        x_early, x_center, x_late = get_oversampled_samples(
            self.signal, 1, 0.0, self.sps
        )
        assert abs(x_center - 1.0) < 0.01

    def test_tau_shift(self):
        x_early_0, x_center_0, x_late_0 = get_oversampled_samples(
            self.signal, 1, 0.0, self.sps
        )
        x_early_p, x_center_p, x_late_p = get_oversampled_samples(
            self.signal, 1, 0.1, self.sps
        )
        assert x_center_p < x_center_0

    def test_early_late_symmetry(self):
        x_early, x_center, x_late = get_oversampled_samples(
            self.signal, 1, 0.0, self.sps
        )
        assert abs(x_early - x_late) < 0.01

    def test_out_of_bounds(self):
        x_early, x_center, x_late = get_oversampled_samples(
            self.signal, 0, -0.5, self.sps
        )
        assert np.isfinite(x_early)
        assert np.isfinite(x_center)
        assert np.isfinite(x_late)


class TestMuellerMullerTED:
    def test_zero_error_balanced_case(self):
        eps = mueller_muller_ted(
            x_current=1.2,
            x_previous=1.2,
            d_current=1.0,
            d_previous=1.0,
        )
        assert eps == 0.0

    def test_expected_formula(self):
        eps = mueller_muller_ted(
            x_current=1.5,
            x_previous=0.5,
            d_current=1.0,
            d_previous=-1.0,
        )
        # epsilon = x_k * d_{k-1} - x_{k-1} * d_k
        assert eps == (-1.5 - 0.5)

    def test_zero_previous_symbol(self):
        eps = mueller_muller_ted(
            x_current=2.0,
            x_previous=0.0,
            d_current=1.0,
            d_previous=0.0,
        )
        assert eps == 0.0
