import pytest
import numpy as np
from src.frontend.filters import ButterworthFilter, create_lowpass_filter


class TestButterworthFilter:
    def setup_method(self):
        self.lpf = ButterworthFilter(cutoff_freq=0.5, fs=100.0)

    def test_order(self):
        assert self.lpf.order == 7

    def test_filter_preserves_length(self):
        signal = np.random.randn(1000)
        output = self.lpf.filter(signal)
        assert len(output) == len(signal)

    def test_dc_gain(self):
        # DC signal should pass through with ~unity gain
        dc_signal = np.ones(200)
        output = self.lpf.filter(dc_signal)
        assert np.allclose(output[100:], 1.0, atol=0.01)

    def test_high_freq_attenuation(self):
        # High frequency signal should be attenuated
        t = np.linspace(0, 10, 1000)
        high_freq = np.sin(2 * np.pi * 40 * t)  # 40 Hz, well above 0.5 Hz cutoff
        output = self.lpf.filter(high_freq)
        assert np.std(output[200:]) < 0.1

    def test_continuous_filtering(self):
        # Filtering in chunks should give same result as filtering all at once
        signal = np.random.randn(500)
        lpf1 = ButterworthFilter(cutoff_freq=0.5, fs=100.0)
        lpf2 = ButterworthFilter(cutoff_freq=0.5, fs=100.0)

        full_output = lpf1.filter(signal)
        chunk1 = lpf2.filter(signal[:250])
        chunk2 = lpf2.filter(signal[250:])

        np.testing.assert_array_almost_equal(full_output[:250], chunk1, decimal=10)
        np.testing.assert_array_almost_equal(full_output[250:], chunk2, decimal=10)


class TestCreateLowpassFilter:
    def test_cutoff_frequency(self):
        # Cutoff = N / (2T)
        lpf = create_lowpass_filter(T=1.0, N=2, fs=100.0)
        assert lpf.cutoff_freq == 1.0  # 2 / (2*1) = 1.0

    def test_different_T(self):
        lpf = create_lowpass_filter(T=0.5, N=2, fs=100.0)
        assert lpf.cutoff_freq == 2.0  # 2 / (2*0.5) = 2.0

    def test_different_N(self):
        lpf = create_lowpass_filter(T=1.0, N=1, fs=100.0)
        assert lpf.cutoff_freq == 0.5  # 1 / (2*1) = 0.5
