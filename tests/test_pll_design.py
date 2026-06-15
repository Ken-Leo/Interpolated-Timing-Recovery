import numpy as np

from src.opsps_va.pll_design import design_pll_gains


class TestPLLDesign:
    def test_design_outputs_positive_gains(self):
        gains = design_pll_gains(convergence_bits=100, ted_slope=1.0)
        assert gains["alpha"] > 0.0
        assert gains["beta"] > 0.0
        assert 0.0 < gains["r"] < 1.0

    def test_closed_loop_det_trace_match(self):
        kd = 1.0
        gains = design_pll_gains(convergence_bits=100, ted_slope=kd)

        alpha = gains["alpha"]
        beta = gains["beta"]
        r = gains["r"]
        phi = gains["phi"]

        # For linearized state matrix:
        # A = [[1-aK_d-bK_d, 1], [-bK_d, 1]]
        a11 = 1.0 - alpha * kd - beta * kd
        a12 = 1.0
        a21 = -beta * kd
        a22 = 1.0

        det_a = a11 * a22 - a12 * a21
        tr_a = a11 + a22

        assert np.isclose(det_a, r * r, rtol=1e-6, atol=1e-9)
        assert np.isclose(tr_a, 2.0 * r * np.cos(phi), rtol=1e-6, atol=1e-9)
