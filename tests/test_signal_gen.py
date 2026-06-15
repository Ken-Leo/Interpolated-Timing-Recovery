import pytest
import numpy as np
from src.channel.signal_gen import generate_binary_sequence, generate_transition_sequence


class TestGenerateBinarySequence:
    def test_length(self):
        a = generate_binary_sequence(100)
        assert len(a) == 100

    def test_values(self):
        a = generate_binary_sequence(50)
        assert set(np.unique(a)).issubset({1, -1})

    def test_reproducibility(self):
        a1 = generate_binary_sequence(20, seed=42)
        a2 = generate_binary_sequence(20, seed=42)
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds(self):
        a1 = generate_binary_sequence(20, seed=1)
        a2 = generate_binary_sequence(20, seed=2)
        assert not np.array_equal(a1, a2)


class TestGenerateTransitionSequence:
    def test_length(self):
        a = generate_binary_sequence(10, seed=0)
        b = generate_transition_sequence(a)
        assert len(b) == len(a)

    def test_values_subset(self):
        a = np.array([1, 1, -1, -1, 1])
        b = generate_transition_sequence(a)
        # First element can be 1 (since a_prev[0]=0), rest should be in {-2, 0, 2}
        assert set(np.unique(b[1:])).issubset({-2, 0, 2})

    def test_no_transition(self):
        a = np.array([1, 1, 1, 1])
        b = generate_transition_sequence(a)
        assert b[1] == 0
        assert b[2] == 0
        assert b[3] == 0

    def test_transition(self):
        a = np.array([1, -1, 1, -1])
        b = generate_transition_sequence(a)
        assert b[1] == -2
        assert b[2] == 2
        assert b[3] == -2
