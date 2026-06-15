import numpy as np
from typing import Tuple

class MMSEInterpolator:
    """
    Implementation of an MMSE Interpolation Filter for magnetic recording channels.
    Based on the principle of maximizing SSNR (minimizing MSE).
    """
    def __init__(self, channel_pulse, target_coeffs: np.ndarray, Ts: float, T: float, num_taps: int = 8):
        """
        Initialize the MMSE Interpolator.
        
        Args:
            channel_pulse: The pulse shape function p(t) or a discrete version.
            target_coeffs: GPR target coefficients H(D).
            Ts: Sampling period.
            T: Symbol period.
            num_taps: Number of taps for the interpolator.
        """
        self.Ts = Ts
        self.T = T
        self.num_taps = num_taps
        self.target = target_coeffs
        self.pulse = channel_pulse
        
        self.weight_lut = {}
        self._precompute_weights()

    def _get_rp(self, tau: float) -> float:
        """Compute the autocorrelation of the pulse p(t) at tau."""
        t = np.linspace(-5*self.T, 5*self.T, 2000)
        dt = t[1] - t[0]
        p_t = self.pulse(t)
        p_tau = np.interp(t + tau, t, p_t, left=0, right=0)
        return np.sum(p_t * p_tau) * dt

    def _precompute_weights(self):
        """Precompute weights for 64 quantized levels of tau in [0, Ts)."""
        print("Precomputing MMSE interpolation weights...")
        num_levels = 64
        for i in range(num_levels):
            tau = (i / num_levels) * self.Ts
            self.weight_lut[i] = self.compute_weights_raw(tau)
        print("Precomputation complete.")

    def compute_weights_raw(self, tau: float) -> np.ndarray:
        """The actual MMSE weight calculation logic."""
        taps_indices = np.arange(-self.num_taps // 2, self.num_taps // 2)
        
        def R_s(delta):
            res = 0.0
            for m in range(len(self.target)):
                for n in range(len(self.target)):
                    res += self.target[m] * self.target[n] * self._get_rp(delta - (m-n)*self.T)
            return res

        R = np.zeros((self.num_taps, self.num_taps))
        for i in range(self.num_taps):
            for j in range(self.num_taps):
                R[i, j] = R_s((i - j) * self.Ts)
        
        p = np.zeros(self.num_taps)
        for i in range(self.num_taps):
            p[i] = R_s(taps_indices[i] * self.Ts + tau)
            
        R += np.eye(self.num_taps) * 1e-9
        return np.linalg.solve(R, p)

    def compute_weights(self, tau: float) -> np.ndarray:
        """
        Retrieve weights from the LUT for a given phase offset tau.
        """
        tau_norm = tau % self.Ts
        idx = int(np.floor((tau_norm / self.Ts) * 64))
        idx = max(0, min(63, idx))
        return self.weight_lut[idx]

    def interpolate(self, signal: np.ndarray, symbol_idx: int, tau: float) -> float:
        """
        Interpolate the signal at symbol_idx with phase offset tau.
        """
        center_idx = int(np.floor((symbol_idx * self.T + tau * self.T) / self.Ts))
        weights = self.compute_weights(tau * self.T % self.Ts)
        
        val = 0.0
        for i in range(self.num_taps):
            idx = center_idx - self.num_taps // 2 + i
            if 0 <= idx < len(signal):
                val += weights[i] * signal[idx]
        
        return val
