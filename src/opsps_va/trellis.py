from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np


@dataclass
class Transition:
    """Represents a transition in the Viterbi trellis."""

    prestate: int  # Index of the previous state
    input_bit: int  # The binary input symbol (+1 or -1)
    output_symbol: float  # Branch target output from the selected GPR target


class GPRTrellis:
    """Trellis generated from an arbitrary binary-input GPR target.

    For target coefficients h = [h0, h1, ..., hL], branch output is:
        y_k = h0*a_k + h1*a_{k-1} + ... + hL*a_{k-L}

    State stores (a_{k-1}, ..., a_{k-L}), so number of states is 2^L.
    """

    def __init__(self, target_coeffs: np.ndarray):
        coeffs = np.asarray(target_coeffs, dtype=float)
        if coeffs.ndim != 1 or coeffs.size < 1:
            raise ValueError("target_coeffs must be a 1-D array with at least one tap")

        self.target_coeffs = coeffs
        self.memory = coeffs.size - 1

        if self.memory == 0:
            state_histories = [tuple()]
        else:
            # Keep +1/-1 ordering deterministic for reproducible state indexing.
            state_histories = list(product([1, -1], repeat=self.memory))

        self.num_states = len(state_histories)
        self.state_history = {idx: hist for idx, hist in enumerate(state_histories)}
        self._history_to_index = {hist: idx for idx, hist in self.state_history.items()}

        self.transitions = {s: [] for s in range(self.num_states)}
        self._build_trellis()

    def _build_trellis(self):
        for pre_s in range(self.num_states):
            history = self.state_history[pre_s]

            for a_curr in [1, -1]:
                symbols = [a_curr, *history]
                y_target = float(np.dot(self.target_coeffs, symbols))

                if self.memory == 0:
                    next_history = tuple()
                else:
                    next_history = tuple(symbols[:-1])

                next_s = self._history_to_index[next_history]
                self.transitions[next_s].append(
                    Transition(
                        prestate=pre_s,
                        input_bit=a_curr,
                        output_symbol=y_target,
                    )
                )

    def get_transitions_to(self, state: int) -> List[Transition]:
        return self.transitions[state]


PR_IV = np.array([1.0, 0.0, -1.0])


class PRIVTrellis:
    """
    Trellis structure for PR-IV target response H(D) = 1 - D^2.
    The target output is y_k = a_k - a_{k-2}.

    States are defined by the history (a_{k-1}, a_{k-2}):
    State 0: (+1, +1)
    State 1: (+1, -1)
    State 2: (-1, +1)
    State 3: (-1, -1)
    """

    def __init__(self):
        impl = GPRTrellis(PR_IV)
        self.num_states = impl.num_states
        self.state_history = impl.state_history
        self.transitions = impl.transitions

    def get_transitions_to(self, state: int) -> List[Transition]:
        """Return all transitions leading into the given state."""
        return self.transitions[state]
