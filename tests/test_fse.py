import pytest
import numpy as np
from src.frontend.fse import FractionallySpacedEqualizer


class TestFractionallySpacedEqualizer:
    def setup_method(self):
        self.taps = [0.0, 0.5, 1.0, 0.5, 0.0]
        self.fse = FractionallySpacedEqualizer(self.taps, T=1.0, N=2)

    def test_initial_taps(self):
        np.testing.assert_array_equal(self.fse.taps, self.taps)

    def test_process_output_length(self):
        signal = np.random.randn(200)
        output = self.fse.process(signal, tau=0.0)
        expected_len = (len(signal) - len(self.taps)) // self.fse.N
        assert abs(len(output) - expected_len) <= 1

    def test_delta_response(self):
        signal = np.zeros(20)
        signal[10] = 1.0
        output = self.fse.process(signal, tau=0.0)
        assert np.max(np.abs(output)) == 1.0

    def test_set_taps(self):
        new_taps = [1.0, 2.0, 3.0]
        self.fse.set_taps(new_taps)
        np.testing.assert_array_equal(self.fse.taps, new_taps)

    def test_update_weights_lms(self):
        fse = FractionallySpacedEqualizer([0.5, 1.0, 0.5], T=1.0)
        old_taps = fse.taps.copy()
        window = np.array([1.0, 1.0, 1.0])
        error = 0.5
        mu = 0.01

        fse.update_weights(error, window, mu)

        assert not np.array_equal(fse.taps, old_taps)
        assert fse.taps[1] > old_taps[1]

    def test_center_symmetry_projection(self):
        w = np.array([0.1, 0.3, 1.0, 0.2, 0.4])
        w_proj = self.fse._project_taps_center_symmetric(w)

        assert abs(w_proj[0] - w_proj[4]) < 1e-10
        assert abs(w_proj[1] - w_proj[3]) < 1e-10

    def test_center_max_enforcement(self):
        w = np.array([0.5, 0.8, 0.3, 0.8, 0.5])
        w_proj = self.fse._project_taps_center_symmetric(w)

        assert w_proj[2] >= np.max(np.delete(w_proj, 2))
