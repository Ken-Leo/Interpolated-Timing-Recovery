import numpy as np
from typing import List, Tuple


class FractionallySpacedEqualizer:
    """
    Implementation of a Fractionally-Spaced Equalizer (FSE).
    The FSE samples the input signal at T/N intervals (N=2) and
    applies a FIR filter to produce a T-spaced output.
    """

    def __init__(
        self,
        taps: List[float],
        T: float,
        N: int = 2,
        project_taps: bool = True,
        symmetry_strength: float = 1.0,
        enforce_center_max: bool = True,
        use_nlms: bool = False,
        nlms_eps: float = 1e-8,
        error_clip: float | None = None,
        tap_leak: float = 0.0,
    ):
        """
        Initialize the FSE.

        Args:
            taps: FIR filter coefficients.
            T: Bit period.
            N: Oversampling ratio (default 2 for FSE).
        """
        self.taps = np.array(taps, dtype=float)
        self.T = T
        self.N = N
        self.Ts = T / N
        self.project_taps = project_taps
        self.symmetry_strength = symmetry_strength
        self.enforce_center_max = enforce_center_max
        self.use_nlms = bool(use_nlms)
        if nlms_eps <= 0.0:
            raise ValueError("nlms_eps must be positive")
        self.nlms_eps = float(nlms_eps)
        if error_clip is not None and error_clip <= 0.0:
            raise ValueError("error_clip must be positive or None")
        self.error_clip = None if error_clip is None else float(error_clip)
        if tap_leak < 0.0 or tap_leak >= 1.0:
            raise ValueError("tap_leak must be in [0, 1)")
        self.tap_leak = float(tap_leak)
        # Buffer to store the oversampled samples
        self.buffer = np.zeros(len(taps))

    def process(self, samples: np.ndarray, tau: float) -> np.ndarray:
        """
        Process the filtered signal using the current sampling phase offset tau.

        Args:
            samples: Filtered signal array (oversampled at Ts).
            tau: Current sampling phase offset.

        Returns:
            y_k: Equalized T-spaced sequence.
        """
        results = []
        # Process in steps of N to downsample to T-spaced output
        for m in range(0, len(samples) - len(self.taps), self.N):
            window = samples[m : m + len(self.taps)]
            y_k = np.dot(window, self.taps)
            results.append(y_k)

        return np.array(results)

    def update_weights(self, error: float, window: np.ndarray, mu: float):
        """
        Update equalizer coefficients using the LMS algorithm.

        Optional projection can enforce a center-symmetric, center-max shape when
        the active target response justifies it.
        """
        update_error = float(error)
        if self.error_clip is not None:
            update_error = float(
                np.clip(update_error, -self.error_clip, self.error_clip)
            )

        # Apply a mild leakage only when adaptation is active.
        if mu > 0.0 and self.tap_leak > 0.0:
            self.taps *= 1.0 - self.tap_leak

        if self.use_nlms:
            denom = float(np.dot(window, window)) + self.nlms_eps
            step = mu / denom
        else:
            step = mu

        # LMS/NLMS update: w = w + step * error * window
        self.taps += step * update_error * window

        # 2. Optionally project the taps to a constrained shape.
        if self.project_taps:
            self.taps = self._project_taps_center_symmetric(
                self.taps,
                symmetry_strength=self.symmetry_strength,
                enforce_center_max=self.enforce_center_max,
            )

    def _project_taps_center_symmetric(
        self,
        w: np.ndarray,
        symmetry_strength: float = 1.0,
        enforce_center_max: bool = True,
    ) -> np.ndarray:
        """Project taps to center symmetry and enforce center maximum.

        For odd tap length, enforce single-center maximum.
        For even tap length, enforce dual-center equal maximum.
        """
        w_proj = w.copy()
        if w_proj.size <= 1:
            return w_proj

        n_taps = w_proj.size

        # Enforce symmetry around center (or center gap for even length).
        for i in range(n_taps // 2):
            j = n_taps - 1 - i
            avg = 0.5 * (w_proj[i] + w_proj[j])
            w_proj[i] = (1.0 - symmetry_strength) * w_proj[i] + symmetry_strength * avg
            w_proj[j] = (1.0 - symmetry_strength) * w_proj[j] + symmetry_strength * avg

        # Enforce center-max
        if enforce_center_max:
            margin = 1e-9
            if n_taps % 2 == 1:
                center_idx = n_taps // 2
                others = np.delete(w_proj, center_idx)
                max_other = np.max(others) if others.size > 0 else -np.inf
                if w_proj[center_idx] <= max_other:
                    w_proj[center_idx] = max_other + margin
            else:
                left_center = n_taps // 2 - 1
                right_center = left_center + 1
                others = np.delete(w_proj, [left_center, right_center])
                max_other = np.max(others) if others.size > 0 else -np.inf
                center_val = max(
                    0.5 * (w_proj[left_center] + w_proj[right_center]),
                    max_other + margin,
                )
                w_proj[left_center] = center_val
                w_proj[right_center] = center_val

        return w_proj

    def set_taps(self, new_taps: List[float]):
        """Update equalizer coefficients."""
        self.taps = np.array(new_taps, dtype=float)
