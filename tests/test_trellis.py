import pytest
import numpy as np

from src.opsps_va.trellis import GPRTrellis, PRIVTrellis, Transition
from src.utils.gpr_coefficients import get_gpr_target


class TestPRIVTrellis:
    def setup_method(self):
        self.trellis = PRIVTrellis()

    def test_num_states(self):
        assert self.trellis.num_states == 4

    def test_state_history(self):
        assert self.trellis.state_history[0] == (1, 1)
        assert self.trellis.state_history[1] == (1, -1)
        assert self.trellis.state_history[2] == (-1, 1)
        assert self.trellis.state_history[3] == (-1, -1)

    def test_transitions_per_state(self):
        for s in range(4):
            transitions = self.trellis.get_transitions_to(s)
            assert len(transitions) == 2  # PR-IV: 2 incoming branches per state

    def test_transition_values(self):
        transitions = self.trellis.get_transitions_to(0)
        inputs = {t.input_bit for t in transitions}
        assert inputs == {1}  # Both transitions to state 0 have input +1

        outputs = {t.output_symbol for t in transitions}
        assert outputs == {0.0, 2.0}  # y = a_k - a_{k-2}

    def test_all_transitions_valid(self):
        for s in range(4):
            for t in self.trellis.get_transitions_to(s):
                assert t.input_bit in [1, -1]
                assert t.output_symbol in [-2.0, 0.0, 2.0]
                assert 0 <= t.prestate < 4

    def test_pr4_target(self):
        # Verify: y_k = a_k - a_{k-2}
        # State 3: (a_{k-1}, a_{k-2}) = (-1, -1), input a_k = 1
        # => y = 1 - (-1) = 2
        transitions = self.trellis.get_transitions_to(0)
        for t in transitions:
            if t.prestate == 3:
                assert t.output_symbol == 2.0
                assert t.input_bit == 1

        # State 0: (a_{k-1}, a_{k-2}) = (1, 1), input a_k = -1
        # => y = -1 - 1 = -2
        transitions = self.trellis.get_transitions_to(2)
        for t in transitions:
            if t.prestate == 0:
                assert t.output_symbol == -2.0
                assert t.input_bit == -1


class TestGPRTrellis:
    def test_lmr_oversampled_state_count(self):
        target = get_gpr_target(mode="lmr", oversampled=True)
        trellis = GPRTrellis(target)
        assert trellis.num_states == 16

    def test_lmr_oversampled_has_two_incoming_per_state(self):
        target = get_gpr_target(mode="lmr", oversampled=True)
        trellis = GPRTrellis(target)
        for s in range(trellis.num_states):
            assert len(trellis.get_transitions_to(s)) == 2

    def test_branch_output_matches_target_equation(self):
        target = get_gpr_target(mode="lmr", oversampled=False)
        trellis = GPRTrellis(target)

        # Pick one transition and verify y = sum_i h_i * a_{k-i}
        transitions = trellis.get_transitions_to(0)
        t = transitions[0]
        history = trellis.state_history[t.prestate]
        symbols = np.array([t.input_bit, *history], dtype=float)
        expected = float(np.dot(target, symbols))
        assert t.output_symbol == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("mode", "oversampled"),
        [("pr4", True), ("pmr", True), ("lmr", True), ("lmr", False)],
    )
    def test_all_branch_outputs_match_active_target(self, mode, oversampled):
        target = get_gpr_target(mode=mode, oversampled=oversampled)
        trellis = GPRTrellis(target)

        for state in range(trellis.num_states):
            for trans in trellis.get_transitions_to(state):
                history = trellis.state_history[trans.prestate]
                symbols = np.array([trans.input_bit, *history], dtype=float)
                expected = float(np.dot(target, symbols))
                assert trans.output_symbol == pytest.approx(expected)

    def test_noiseless_target_sequence_is_recovered_by_trellis_viterbi_core(self):
        rng = np.random.default_rng(0)
        target = get_gpr_target(mode="pmr", oversampled=True)
        trellis = GPRTrellis(target)
        memory = target.size - 1

        truth = rng.choice([-1.0, 1.0], size=32)
        padded_truth = np.concatenate([np.ones(memory), truth])

        target_output = np.zeros_like(padded_truth)
        for symbol_idx in range(padded_truth.size):
            for tap_idx, coeff in enumerate(target):
                bit_idx = symbol_idx - tap_idx
                if bit_idx >= 0:
                    target_output[symbol_idx] += coeff * padded_truth[bit_idx]

        phi = np.full(trellis.num_states, np.inf)
        phi[0] = 0.0
        paths = {state: [] for state in range(trellis.num_states)}

        for sample in target_output[memory:]:
            new_phi = np.full(trellis.num_states, np.inf)
            new_paths = {state: [] for state in range(trellis.num_states)}

            for state in range(trellis.num_states):
                best_metric = np.inf
                best_path = None

                for trans in trellis.get_transitions_to(state):
                    metric = phi[trans.prestate] + (sample - trans.output_symbol) ** 2
                    if metric < best_metric:
                        best_metric = metric
                        best_path = paths[trans.prestate] + [trans.input_bit]

                new_phi[state] = best_metric
                new_paths[state] = best_path if best_path is not None else []

            phi = new_phi
            paths = new_paths

        best_final_state = int(np.argmin(phi))
        detected = np.array(paths[best_final_state])

        assert np.array_equal(detected, truth)
