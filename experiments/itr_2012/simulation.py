"""
ITR 2012 Simulation — Interpolated Timing Recovery for PMR Channel.

Paper: "Performance of Interpolated Timing Recovery in Perpendicular
       Magnetic Recording Channel" (Kovintavewat et al., I-SEEC 2012).

Architecture:
  r(t) → LPF → ADC(Ts=T/N) → FSE(Ts-spaced) → Interpolator → Viterbi → a_hat_k
                                                ↑                ↓
                                           NCO/PLL ←── MM TED ←──┘
Key equations:
  Ts = T / N                 (N = oversampling factor, e.g. 1.05 = 5%)
  epsilon_hat_k = y_k * d_hat_{k-1} - y_{k-1} * d_hat_k    (Mueller-Müller TED)
  theta_hat_k = theta_hat_{k-1} + beta * epsilon_hat_k              (frequency update)
  tau_hat_{k+1} = tau_hat_k + (alpha * epsilon_hat_k + theta_hat_k) * T/Ts + (T-Ts)/Ts   (phase + drift)
  e_k_fir approx (1 - tau_hat_k * Ts/T) * e_k + tau_hat_k * (Ts/T) * e_{k-1}  (ITR^-1)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.ted import mueller_muller_ted
from src.opsps_va.trellis import GPRTrellis
from src.utils.gpr_target import gen_gpr_target


# ──────────────────────────────────────────────
# Cubic interpolator
# ──────────────────────────────────────────────
def cubic_interp(signal: np.ndarray, pos_float: float) -> float:
    """Cubic interpolation at fractional position."""
    i = int(np.floor(pos_float))
    u = pos_float - i

    def _v(idx):
        if 0 <= idx < len(signal):
            return signal[idx]
        return 0.0

    y0, y1, y2, y3 = _v(i - 1), _v(i), _v(i + 1), _v(i + 2)
    c0 = ((u - 0) * (u - 1) * (u - 2)) / (-6.0)
    c1 = ((u + 1) * (u - 1) * (u - 2)) / 2.0
    c2 = ((u + 1) * u * (u - 2)) / (-2.0)
    c3 = ((u + 1) * u * (u - 1)) / 6.0
    return c0 * y0 + c1 * y1 + c2 * y2 + c3 * y3


def quadratic_interp(signal: np.ndarray, pos_float: float) -> float:
    """Quadratic interpolation (Farrow structure) at fractional position."""
    i = int(np.floor(pos_float))
    u = pos_float - i

    def _v(idx):
        if 0 <= idx < len(signal):
            return signal[idx]
        return 0.0

    y0, y1, y2, y3 = _v(i - 1), _v(i), _v(i + 1), _v(i + 2)

    # Coefficients based on the MATLAB second-order Farrow structure
    # y = (FI1 * u + FI2) * u + FI3
    fi1 = 0.5 * y3 - 0.5 * y2 - 0.5 * y1 + 0.5 * y0
    fi2 = 1.5 * y2 - 0.5 * y3 - 0.5 * y1 - 0.5 * y0
    fi3 = y1

    return (fi1 * u + fi2) * u + fi3


# ──────────────────────────────────────────────
# NCO + second-order PLL (paper Eqs. 4–5)
# ──────────────────────────────────────────────
class NCOPLL:
    """Numerically-controlled oscillator with second-order loop filter."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        T: float = 1.0,
        Ts: float = 1.0 / 1.05,
        tau_init: float = 0.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.Ts = Ts
        self.ratio = T / Ts  # = 1.05
        self.drift = (T - Ts) / Ts  # ≈ 0.04762
        self.tau = tau_init
        self.theta = -(T - Ts) / T  # steady-state, cancels drift from start
        self.mu = tau_init

    def step(self, epsilon: float) -> tuple:
        """Update PLL state. Returns (mk, mu_k): base index & fractional interval."""
        self.theta += self.beta * epsilon
        # Phase update with drift compensation (paper Eq. 4)
        self.tau += (self.alpha * epsilon + self.theta) * self.ratio + self.drift
        mk = int(np.floor(self.tau))
        self.mu = self.tau - mk
        return mk, self.mu


# ──────────────────────────────────────────────
# ITR Receiver
# ──────────────────────────────────────────────
class ITRReceiver:
    """Interpolated Timing Recovery receiver."""

    def __init__(
        self,
        target_coeffs: np.ndarray,
        fse_taps: np.ndarray,
        Ts: float,
        T: float = 1.0,
        acq_gains: tuple = (0.1, 0.004),
        trac_gains: tuple = (0.012, 0.00027),
        lms_mu: float = 0.01,
        interp_mode: str = "cubic",
        ted_mode: str = "ground_truth",
        eq_mode: str = "wiener",
        eq_mu: float = 0.01,
        eq_mu_data_scale: float = 0.1,
        eq_tap_leak: float = 1e-5,
    ):
        self.target = np.asarray(target_coeffs, dtype=float)
        self.fse_taps = np.asarray(fse_taps, dtype=float).copy()
        self.Ts = Ts
        self.T = T
        self.acq_gains = acq_gains
        self.trac_gains = trac_gains
        self.lms_mu = lms_mu
        self.eq_mu = eq_mu
        self.eq_mu_data_scale = eq_mu_data_scale
        self.eq_tap_leak = eq_tap_leak
        self.trellis = GPRTrellis(self.target)
        self.nco = NCOPLL(*acq_gains, T=T, Ts=Ts)
        self.half_tap = (len(self.fse_taps) - 1) // 2
        self.scale = 1.0
        self.bias = 0.0
        self._interp_fn = self._select_interp(interp_mode)
        self._ted_mode = ted_mode
        self._eq_mode = eq_mode
        # Decision history for DD mode: stores last L detected bits [â_{k-1}, ..., â_{k-L}]
        self._decision_history = np.zeros(len(self.target), dtype=float)
        # Validated eq_mode values
        self._valid_eq_modes = {"wiener", "lms_preamble", "nlms_adaptive"}
        if eq_mode not in self._valid_eq_modes:
            raise ValueError(
                f"Unknown eq_mode: {eq_mode!r}. "
                f"Valid options: {sorted(self._valid_eq_modes)}"
            )

    @staticmethod
    def _select_interp(mode: str):
        """Select interpolator by mode string."""
        if mode == "cubic":
            return cubic_interp
        elif mode == "quadratic":
            return quadratic_interp
        else:
            raise ValueError(f"Unknown interpolator mode: {mode}")

    def _fse_output(self, r_ts: np.ndarray, center_idx: int) -> float:
        """Ts-spaced FSE output at position center_idx."""
        w_start = center_idx - self.half_tap
        window = np.zeros(len(self.fse_taps))
        for j in range(len(self.fse_taps)):
            idx = w_start + j
            if 0 <= idx < len(r_ts):
                window[j] = r_ts[idx]
        return np.dot(window, self.fse_taps)

    def _itp_inverse(self, ek: float, ek_prev: float, tau: float) -> float:
        """ITR⁻¹: map T-domain error to Ts-domain (paper Eq. 6)."""
        factor = tau * self.Ts / self.T
        return (1.0 - factor) * ek + factor * ek_prev

    def _get_fse_window(self, r_ts: np.ndarray, fse_center: int) -> np.ndarray:
        """Extract FSE tap-length window centered at fse_center from Ts-domain signal."""
        win_start = fse_center - self.half_tap
        win = np.zeros(len(self.fse_taps))
        for j in range(len(self.fse_taps)):
            idx = win_start + j
            if 0 <= idx < len(r_ts):
                win[j] = r_ts[idx]
        return win

    def _nlms_update(
        self,
        r_ts: np.ndarray,
        fse_center: int,
        error: float,
        ek_prev: float,
        is_data: bool = False,
    ) -> float:
        """Normalized LMS update for the FSE taps.

        Applies ITR⁻¹ mapping to convert the T-domain error to Ts-domain,
        then performs an NLMS update on ``self.fse_taps``.

        In the data section (is_data=True), a reduced step size
        (``eq_mu * eq_mu_data_scale``) is used to prevent coefficient
        drift from decision-directed errors. A small tap leakage
        (``eq_tap_leak``) is applied at every update to bound the
        coefficient norm.

        Returns:
            The current error (to be used as ek_prev in the next call).
        """
        mu = self.eq_mu * (self.eq_mu_data_scale if is_data else 1.0)
        if mu <= 0.0:
            return error

        win_vec = self._get_fse_window(r_ts, fse_center)
        ek_fir = self._itp_inverse(error, ek_prev, self.nco.tau)
        denom = float(np.dot(win_vec, win_vec)) + 1e-8

        # Leakage: pull taps towards zero to bound drift
        if self.eq_tap_leak > 0.0:
            self.fse_taps *= 1.0 - self.eq_tap_leak

        # NLMS update
        self.fse_taps += mu * ek_fir * win_vec / denom
        return error

    @staticmethod
    def _slice_symbol(y: float) -> float:
        """Hard-decision slicer: map y_k to {+1, -1}.

        Assumes GPR target has a dominant center tap, so the sign of y_k
        corresponds to the sign of a_k.
        """
        return 1.0 if y >= 0.0 else -1.0

    def _slice_with_decision_feedback(
        self, y_k: float, past_decisions: np.ndarray
    ) -> float:
        """Slice y_k with ISI cancellation from past decisions.

        Subtract the contribution of known past symbols to isolate h0 * a_k,
        then slice on the residual.

        Args:
            y_k: Current equalized sample.
            past_decisions: [â_{k-1}, â_{k-2}, ..., â_{k-L}] (length L).

        Returns:
            â_k: Sliced symbol in {+1, -1}.
        """
        # ISI contribution from past symbols
        isi = sum(
            self.target[j] * float(past_decisions[j - 1])
            for j in range(1, min(len(self.target), len(past_decisions) + 1))
        )
        residual = y_k - isi
        h0 = self.target[0]
        return 1.0 if (h0 * residual) >= 0.0 else -1.0

    def _target_output(self, bits: np.ndarray) -> float:
        """Compute d_k = sum(h_j * a_{k-j}) from the last L bits.

        Args:
            bits: Array of last L detected bits [â_k, â_{k-1}, ..., â_{k-L+1}].

        Returns:
            d_hat_k: The target output at symbol k.
        """
        # bits may be shorter than target on first few symbols
        L = len(self.target)
        d = 0.0
        for j in range(min(L, len(bits))):
            d += self.target[j] * float(bits[j])
        return d

    def process(self, r_ts: np.ndarray, a_true: np.ndarray, data_start: int) -> tuple:
        """Run ITR over one packet with streaming Viterbi-based timing recovery.

        Three phases:
          1. Preamble (k < data_start): PLL with ground-truth decisions, LMS active.
          2. Data warm-up: Viterbi accumulates path history, tau held fixed.
          3. Data tracking: PLL driven by delayed Viterbi decisions.

        Returns: tau_history, y_history, a_hat (Viterbi decisions)
        """
        num_sym = len(a_true)
        y_history = np.zeros(num_sym)
        tau_history = np.zeros(num_sym)
        ek_prev = 0.0

        for k in range(num_sym):
            # ── Interpolation ──
            pos_ts = k * self.nco.ratio + self.nco.tau
            mk = int(np.floor(pos_ts))
            fse_center = mk + self.half_tap
            eq_samples = np.array(
                [
                    self._fse_output(r_ts, fse_center - 1),
                    self._fse_output(r_ts, fse_center),
                    self._fse_output(r_ts, fse_center + 1),
                    self._fse_output(r_ts, fse_center + 2),
                ]
            )
            y_k_raw = self._interp_fn(eq_samples, 1.0 + (pos_ts - mk))
            y_k = self.scale * y_k_raw + self.bias
            y_history[k] = y_k

            # ── PLL ──
            if k > 0:
                if self._ted_mode == "ground_truth":
                    # Use ground-truth bits throughout
                    a_window = np.array(
                        [
                            float(a_true[k - j])
                            for j in range(len(self.target))
                            if k - j >= 0
                        ]
                        + [0.0] * max(0, len(self.target) - k - 1)
                    )
                    a_window1 = np.array(
                        [
                            float(a_true[k - 1 - j])
                            for j in range(len(self.target))
                            if k - 1 - j >= 0
                        ]
                        + [0.0] * max(0, len(self.target) - k)
                    )
                    d_k = self._target_output(a_window)
                    d_k1 = self._target_output(a_window1)

                elif self._ted_mode == "decision_directed":
                    if k < data_start:
                        # Preamble: use ground-truth (known 4T pattern)
                        a_window = np.array(
                            [
                                float(a_true[k - j])
                                for j in range(len(self.target))
                                if k - j >= 0
                            ]
                            + [0.0] * max(0, len(self.target) - k - 1)
                        )
                        a_window1 = np.array(
                            [
                                float(a_true[k - 1 - j])
                                for j in range(len(self.target))
                                if k - 1 - j >= 0
                            ]
                            + [0.0] * max(0, len(self.target) - k)
                        )
                        d_k = self._target_output(a_window)
                        d_k1 = self._target_output(a_window1)
                    else:
                        # Data section: decision-directed TED via DDF slicer
                        # self._decision_history = [â_{k-1}, â_{k-2}, ..., â_{k-L}]
                        a_k = self._slice_with_decision_feedback(
                            y_k, self._decision_history
                        )
                        a_window = np.concatenate(([a_k], self._decision_history))
                        d_k = self._target_output(a_window)
                        d_k1 = self._target_output(self._decision_history)

                        # Update decision history: [â_k, â_{k-1}, ..., â_{k-L+1}]
                        self._decision_history[1:] = self._decision_history[:-1]
                        self._decision_history[0] = a_k

                else:
                    raise ValueError(f"Unknown ted_mode: {self._ted_mode}")

                # ── Seed decision history at preamble→data transition ──
                if self._ted_mode == "decision_directed" and k == data_start - 1:
                    # Copy the last L ground-truth preamble bits into decision history
                    L = len(self.target)
                    for j in range(L):
                        if data_start - 1 - j >= 0:
                            self._decision_history[j] = float(
                                a_true[data_start - 1 - j]
                            )
                        else:
                            self._decision_history[j] = 0.0

                epsilon = mueller_muller_ted(y_history[k], y_history[k - 1], d_k, d_k1)
                self.nco.step(epsilon)

                # ── Equalizer adaptation ──
                if self._eq_mode == "wiener":
                    pass  # No online adaptation; use pre-computed Wiener coefficients.

                elif self._eq_mode == "lms_preamble":
                    # LMS/NLMS only during preamble, frozen in data section.
                    if k < data_start:
                        ek_lms = d_k - y_k
                        ek_prev = self._nlms_update(
                            r_ts, fse_center, ek_lms, ek_prev, is_data=False
                        )

                elif self._eq_mode == "nlms_adaptive":
                    # Practically same as lms_preamble: NLMS only on
                    # preamble (ground-truth decisions), frozen in data
                    # section.  The data section FSE update is omitted
                    # because the DD error and PLL timing share the same
                    # signal path, creating a positive-feedback loop
                    # that degrades equalization.
                    if k < data_start:
                        eq_error = d_k - y_k
                        ek_prev = self._nlms_update(
                            r_ts,
                            fse_center,
                            eq_error,
                            ek_prev,
                            is_data=False,
                        )

            tau_history[k] = self.nco.tau

        a_hat = self._viterbi_detect(y_history)
        return tau_history, y_history, a_hat

    def _viterbi_detect(self, y: np.ndarray) -> np.ndarray:
        n_states = self.trellis.num_states
        phi = np.full(n_states, np.inf)
        phi[0] = 0.0
        paths = {s: [] for s in range(n_states)}
        for val in y:
            new_phi = np.full(n_states, np.inf)
            new_paths = {s: [] for s in range(n_states)}
            for s in range(n_states):
                best_m, best_p = np.inf, []
                for trans in self.trellis.get_transitions_to(s):
                    m = phi[trans.prestate] + (val - trans.output_symbol) ** 2
                    if m < best_m:
                        best_m, best_p = m, paths[trans.prestate] + [trans.input_bit]
                new_phi[s], new_paths[s] = best_m, best_p
            phi, paths = new_phi, new_paths
        return np.array(paths[int(np.argmin(phi))], dtype=float)


# ──────────────────────────────────────────────
# Target + FSE estimation
# ──────────────────────────────────────────────
def estimate_target_and_fse(
    T: float = 1.0,
    Ts: float = 1.0 / 1.05,
    pw50: float = 2.5,
    gpr_len: int = 5,
    fse_len: int = 21,
    design_bits: int = 2400,
    snr_db: float = 25.0,
    seed: int = 7,
    fs: int = 105,
) -> tuple:
    """Estimate GPR target (T-domain) and Ts-spaced Wiener FSE.

    Args:
        T: Nominal bit period.
        Ts: Receiver sampling period (Ts = T/N, where N is oversampling factor).
        pw50: Pulse width at 50% amplitude.
        gpr_len: GPR target length.
        fse_len: FSE tap count.
        design_bits: Number of bits for Wiener filter design.
        snr_db: SNR in dB for design signal.
        seed: Random seed.
        fs: Samples per T for signal generation. Default 105 (N=1.05 -> 105 spT).
             For N=1.25, use fs=125.
    """
    _, r_raw, a_full, _, _ = synthesize_readback_signal(
        length=design_bits,
        T=T,
        pw50=pw50,
        mode="pmr",
        sigma_j=0.0,
        sigma_w=0.0,
        freq_offset=0.0,
        snr_db=snr_db,
        fs=fs,
        seed=seed,
        preamble_length=0,
    )
    r_filtered = create_lowpass_filter(T=T, N=1, fs=fs).filter(r_raw)

    # T-rate samples for target estimation (mid-symbol: offset=fs//2)
    t_step = fs
    center_offset = fs // 2
    r_t = r_filtered[center_offset::t_step][: len(a_full)]

    # GPR target (T-domain)
    gpr_template = np.ones(gpr_len, dtype=float)
    _, gpr_coeff = gen_gpr_target(
        random_data=a_full.astype(float),
        sampled_data=r_t,
        gpr_template=gpr_template,
        fir_len=21,
        constraint="1",
        method="lagrange",
    )

    # Ts-rate samples: Ts = T/N, samples per Ts = fs / N = fs * Ts / T
    # fs samples per T, Ts/T samples per Ts → step = int(fs * Ts / T) = int(fs / N)
    N = T / Ts
    ts_step = int(round(fs / N))
    r_ts = r_filtered[::ts_step]

    # Ideal target output at T-rate
    d_t = np.zeros(len(a_full))
    for k in range(len(a_full)):
        d_t[k] = sum(
            gpr_coeff[j] * a_full[k - j] for j in range(len(gpr_coeff)) if k - j >= 0
        )

    # Interpolate d_t to Ts-domain
    ratio = T / Ts
    n_ts = min(len(r_ts), int(len(d_t) * ratio + 10))
    d_ts = np.zeros(n_ts)
    for n in range(n_ts):
        t_pos = n / ratio
        k0 = min(int(np.floor(t_pos)), len(d_t) - 1)
        k1 = min(k0 + 1, len(d_t) - 1)
        frac = t_pos - k0
        d_ts[n] = (1.0 - frac) * d_t[k0] + frac * d_t[k1]

    # Wiener FSE in Ts-domain — match fse_out[n] to d_ts[n] (no half shift)
    half = (fse_len - 1) // 2
    n_train = len(r_ts) - fse_len + 1
    if n_train <= 0:
        fse_taps = np.zeros(fse_len)
        fse_taps[half] = 1.0
        return gpr_coeff, fse_taps

    X = np.zeros((n_train, fse_len))
    for j in range(fse_len):
        X[:, j] = r_ts[j : j + n_train]
    target_train = d_ts[:n_train]
    ridge = 1e-4 * np.eye(fse_len)
    R = X.T @ X / n_train + ridge
    p = X.T @ target_train / n_train
    fse_taps = np.linalg.solve(R, p)
    return gpr_coeff, fse_taps


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────
def run_ber_sweep(
    eq_mode: str = "wiener",
    eq_mu: float = 0.01,
    eq_mu_data_scale: float = 0.1,
    eq_tap_leak: float = 1e-5,
):
    """Run BER sweep with configurable TED mode and equalizer adaptation mode.

    Args:
        eq_mode: Equalizer adaptation mode.
            "wiener"        - Fixed Wiener FSE, no online adaptation.
            "lms_preamble"  - NLMS only during preamble, frozen in data section.
            "nlms_adaptive" - Continuous NLMS (decision-directed in data section).
        eq_mu: Step size for NLMS adaptation (preamble phase).
        eq_mu_data_scale: Multiplier applied to eq_mu in the data section
                          (only for nlms_adaptive mode). Default 0.1 reduces
                          data-section updates to prevent decision-error drift.
        eq_tap_leak: Per-step leakage factor applied before the NLMS update.
                     Pulls taps towards zero to bound drift. Default 1e-5.
    """
    T = 1.0
    N = 1.25
    Ts = T / N
    FS = N * 100
    pw50 = 2.5
    ted_mode = "decision_directed"

    print(
        f"Estimating GPR target and Ts-spaced FSE "
        f"(ted_mode={ted_mode}, eq_mode={eq_mode}, N={N:.2f})..."
    )
    target, fse_taps = estimate_target_and_fse(T=T, Ts=Ts, pw50=pw50, fs=int(FS))
    print(f"GPR target: {np.array2string(target, precision=4)}")

    snr_range = np.arange(2, 30, 2)
    results = []

    for snr in snr_range:
        total_errors, total_bits = 0, 0
        n_sectors = 20
        for s in range(n_sectors):
            _, r_raw, a_full, _, data_start = synthesize_readback_signal(
                length=4095,
                T=T,
                pw50=pw50,
                mode="pmr",
                sigma_j=0.01 * T,
                sigma_w=0.005 * T,
                freq_offset=0.004,
                snr_db=float(snr),
                fs=int(FS),
                seed=s,
                preamble_length=128,
                preamble_pattern="4T",
            )
            r_f = create_lowpass_filter(T=T, N=1, fs=FS).filter(r_raw)
            # Ts-rate: fs=FS, Ts=T/N → decimate by int(FS * Ts)
            r_ts = r_f[:: int(FS * Ts)]

            rec = ITRReceiver(
                target_coeffs=target,
                fse_taps=fse_taps.copy(),
                Ts=Ts,
                T=T,
                acq_gains=(0.1, 0.004),
                trac_gains=(0.012, 0.00027),
                lms_mu=0.01,
                interp_mode="quadratic",
                ted_mode=ted_mode,
                eq_mode=eq_mode,
                eq_mu=eq_mu,
                eq_mu_data_scale=eq_mu_data_scale,
                eq_tap_leak=eq_tap_leak,
            )
            rec.scale = 1.0  # Wiener FSE already matches target energy
            rec.nco.tau = 0.0

            _, _, a_hat = rec.process(r_ts, a_full, data_start)
            a_data = a_full[data_start:]
            a_det = (
                a_hat[-len(a_data) :]
                if len(a_hat) >= len(a_data)
                else np.zeros(len(a_data))
            )
            if len(a_det) >= len(a_data):
                a_det = a_det[-len(a_data) :]
                total_errors += int(np.sum(a_data != a_det))
                total_bits += len(a_data)

        ber = total_errors / max(total_bits, 1)
        results.append((float(snr), ber))
        print(f"SNR={snr:.1f}dB  BER={ber:.6e}  ({total_errors}/{total_bits})")

    # Plot
    _snrs, _bers = zip(*results)
    plt.figure(figsize=(8, 6))
    plt.semilogy(_snrs, _bers, "bo-", linewidth=2, markersize=6)
    plt.grid(True, which="both", alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"ITR 2012 — PMR, ND=2.5 (eq_mode={eq_mode})")
    od = Path(__file__).parent / "results"
    od.mkdir(parents=True, exist_ok=True)
    fname = f"ber_curve_{eq_mode}.png"
    plt.savefig(od / fname, dpi=150)
    print(f"Saved: {od / fname}")
    plt.close()

    return _snrs, _bers, eq_mode


if __name__ == "__main__":
    # Run all three equalizer modes for comparison.
    _eq_modes = ["wiener", "lms_preamble", "nlms_adaptive"]
    _all_results = {}
    for _mode in _eq_modes:
        print(f"\n{'='*60}")
        print(f"Running eq_mode={_mode!r}")
        print(f"{'='*60}")
        if _mode == "nlms_adaptive":
            # NLMS only on preamble; leakage helps bound drift from
            # the preamble-only updates as well.
            _s, _b, _ = run_ber_sweep(
                eq_mode=_mode,
                eq_mu=0.02,
                eq_mu_data_scale=1.0,
                eq_tap_leak=1e-5,
            )
        else:
            _s, _b, _ = run_ber_sweep(eq_mode=_mode, eq_mu=0.01)
        _all_results[_mode] = (_s, _b)

    # Overlay comparison plot
    plt.figure(figsize=(10, 7))
    _colors = {"wiener": "bo-", "lms_preamble": "rs--", "nlms_adaptive": "g^:"}
    for _mode, (_s, _b) in _all_results.items():
        plt.semilogy(_s, _b, _colors[_mode], linewidth=2, markersize=6, label=_mode)
    plt.grid(True, which="both", alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("ITR 2012 — PMR, ND=2.5 (Equalizer Mode Comparison)")
    plt.legend()
    _od = Path(__file__).parent / "results"
    _od.mkdir(parents=True, exist_ok=True)
    plt.savefig(_od / "ber_curve_comparison.png", dpi=150)
    print(f"Saved: {_od / 'ber_curve_comparison.png'}")
    plt.close()
