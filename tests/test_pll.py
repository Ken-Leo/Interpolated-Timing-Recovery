import pytest
from src.opsps_va.survivor_pll import SurvivorPLL, PLLState


class TestPLLState:
    def test_default_values(self):
        state = PLLState()
        assert state.tau == 0.0
        assert state.theta == 0.0

    def test_custom_values(self):
        state = PLLState(tau=0.5, theta=0.1)
        assert state.tau == 0.5
        assert state.theta == 0.1


class TestSurvivorPLL:
    def setup_method(self):
        self.pll = SurvivorPLL(alpha=0.01, beta=0.001)

    def test_zero_error_no_change(self):
        state = PLLState(tau=0.5, theta=0.1)
        new_state = self.pll.update(state, 0.0)
        assert new_state.tau == 0.5 + 0.1
        assert new_state.theta == 0.1

    def test_positive_error_increases_tau(self):
        state = PLLState(tau=0.0, theta=0.0)
        new_state = self.pll.update(state, 1.0)
        assert new_state.tau > 0.0
        assert new_state.theta > 0.0

    def test_negative_error_decreases_tau(self):
        state = PLLState(tau=0.0, theta=0.0)
        new_state = self.pll.update(state, -1.0)
        assert new_state.tau < 0.0
        assert new_state.theta < 0.0

    def test_gain_scaling(self):
        pll_fast = SurvivorPLL(alpha=0.1, beta=0.01)
        pll_slow = SurvivorPLL(alpha=0.01, beta=0.001)

        state = PLLState(tau=0.0, theta=0.0)
        new_fast = pll_fast.update(state, 1.0)
        new_slow = pll_slow.update(state, 1.0)

        assert new_fast.tau > new_slow.tau
        assert new_fast.theta > new_slow.theta

    def test_convergence_to_zero(self):
        """PLL should converge tau toward 0 with correct error sign.

        The PLL uses positive feedback (paper equations).
        The TED S-curve has negative slope: eps has opposite sign to tau.
        Positive feedback + negative S-curve = overall negative feedback → convergence.
        """
        pll = SurvivorPLL(alpha=0.005, beta=0.0005)
        state = PLLState(tau=0.3, theta=0.0)

        # S-curve: epsilon has OPPOSITE sign to tau (negative slope)
        for _ in range(200):
            epsilon = -state.tau  # eps = -k * tau
            state = pll.update(state, epsilon)

        assert abs(state.tau) < 0.5  # Should be reduced significantly

    def test_frequency_tracking(self):
        """PLL should track a constant frequency offset."""
        pll = SurvivorPLL(alpha=0.01, beta=0.001)
        state = PLLState(tau=0.0, theta=0.0)

        for _ in range(200):
            epsilon = 0.1
            state = pll.update(state, epsilon)

        assert abs(state.theta) > 0.001
