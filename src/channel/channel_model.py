import numpy as np
from scipy.special import erf
from .signal_gen import generate_binary_sequence, generate_transition_sequence
from .jitter import generate_media_jitter, generate_clock_jitter


def _normalize_channel_mode(mode: str) -> str:
    """Map public mode aliases to the underlying channel family."""
    normalized = mode.lower()
    if normalized in {"pmr", "perpendicular"}:
        return "pmr"
    if normalized in {"lmr", "longitudinal"}:
        return "lmr"
    raise ValueError(f"Unknown channel mode: {mode}")


def _build_fractional_kernel_bank(
    T: float,
    pw50: float,
    mode: str,
    fs: int,
    half_window_symbols: float,
    frac_bins: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Precompute h(t) kernels for quantized sub-sample delays.

    Returns:
        rel_time: Relative time axis around the kernel center.
        bank: Shape (frac_bins, kernel_len), bank[q] corresponds to
              fractional delay q/(frac_bins-1) of one sample.
        half_window_samples: Half window size in samples.
    """
    half_window_samples = int(np.ceil(half_window_symbols * fs))
    rel_samples = np.arange(-half_window_samples, half_window_samples + 1)
    rel_time = rel_samples / float(fs)

    if frac_bins < 2:
        frac_bins = 2

    bank = np.zeros((frac_bins, rel_time.size), dtype=float)
    frac_grid = np.linspace(0.0, 1.0, frac_bins)
    dt = 1.0 / float(fs)

    for q, frac in enumerate(frac_grid):
        # Shift by fractional sample delay within [0, 1] sample.
        bank[q] = h_response(rel_time - frac * dt, T, pw50, mode)

    return rel_time, bank, half_window_samples


def p_transition(t, pw50, mode="pmr"):
    """
    Transition response p(t) per Kovintavewat et al. (2003) Sec. 2.

    PMR: p(t) = erf(2 * t * sqrt(ln 2) / PW50)
    LMR: p(t) = 1 / (1 + (2t/PW50)^2)
    """
    normalized_mode = _normalize_channel_mode(mode)

    if normalized_mode == "pmr":
        A = 2.0 * np.sqrt(np.log(2.0)) / pw50
        return erf(A * t)

    A = 2.0 / pw50
    return 1.0 / (1.0 + (A * t) ** 2)


def h_response(t, T, pw50, mode="pmr"):
    """
    Dibit response: h(t) = p(t) - p(t - T)
    """
    return p_transition(t, pw50, mode) - p_transition(t - T, pw50, mode)


def generate_preamble(length: int, pattern: str = "4T") -> np.ndarray:
    """
    Generate a preamble sequence for timing acquisition.

    Args:
        length: Number of preamble bits.
        pattern: Preamble pattern type.
            '4T' -> alternating +1, -1 (period 4T pattern: 1,1,-1,-1,...)

    Returns:
        Preamble bit sequence in {+1, -1}.
    """
    if pattern == "4T":
        # 4T pattern: two +1 followed by two -1, repeated
        preamble = np.zeros(length, dtype=int)
        for i in range(length):
            preamble[i] = 1 if (i % 4) < 2 else -1
        return preamble
    else:
        raise ValueError(f"Unknown preamble pattern: {pattern}")


def synthesize_readback_signal(
    length=100,
    T=1.0,
    pw50=1.0,
    mode="pmr",
    sigma_j=0.03,
    sigma_w=0.005,
    freq_offset=0.0,
    snr_db=20.0,
    fs=100,
    seed=None,
    preamble_length=0,
    preamble_pattern="4T",
):
    """
    Synthesize the readback signal per paper Eq. (1):

    p(t) = sum_k a_k * h(t - k*T_eff - delta_t_k - tau_k) + n(t)

    where:
        T_eff = T * (1 + freq_offset)    (frequency offset)
        delta_t_k ~ N(0, |b_k/2| * sigma_j^2), truncated to T/2  (media jitter)
        tau_k = tau_{k-1} + N(0, sigma_w^2)                      (clock jitter, random walk)

    Paper parameters (Sec.4):
        ND = 2.5, sigma_j/T = 3%, sigma_w/T = 0.5%, freq_offset = 0.4%
        Packet: C-bit preamble (4T pattern) + 4096-bit data

    Args:
        length: Number of data bits (excluding preamble).
        T: Nominal bit period.
        pw50: Pulse width at 50% amplitude.
        mode: 'pmr'/'perpendicular' or 'lmr'/'longitudinal'.
        sigma_j: Media jitter std (as fraction of T).
        sigma_w: Clock jitter step std (as fraction of T).
        freq_offset: Normalized frequency offset (e.g., 0.004 = 0.4%).
        snr_db: Signal-to-noise ratio in dB.
        fs: Samples per nominal symbol period T.
        seed: Random seed for reproducibility.
        preamble_length: Number of preamble bits (0 = no preamble).
        preamble_pattern: Preamble pattern type ('4T').

    Returns:
        t: Time axis.
        p_total: Readback signal with noise.
        a: Full bit sequence (preamble + data).
        b: Transition sequence.
        data_start: Index where data bits begin in the sequence.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate sequences
    if preamble_length > 0:
        preamble = generate_preamble(preamble_length, preamble_pattern)
        data = generate_binary_sequence(length, seed=seed)
        a = np.concatenate([preamble, data])
        data_start = preamble_length
    else:
        a = generate_binary_sequence(length, seed=seed)
        data_start = 0

    b = generate_transition_sequence(a)
    total_len = len(a)

    # 2. Generate jitter sequences (paper Sec.2)
    delta_t = generate_media_jitter(b, sigma_j, T)
    tau_k = generate_clock_jitter(total_len, sigma_w)

    # 3. Effective bit period with frequency offset
    T_eff = T * (1.0 + freq_offset)

    # 4. Define time axis on the receiver nominal sampling grid.
    num_samples = int(total_len * T * fs)
    t = np.arange(num_samples, dtype=float) / float(fs)

    # 5. Synthesize signal using local-window accumulation with
    # a fractional-delay kernel bank.
    # This keeps the same channel model while avoiding O(total_len * num_samples)
    # full-length accumulation for each bit.
    half_window_symbols = 8.0
    frac_bins = 32
    _, kernel_bank, half_window_samples = _build_fractional_kernel_bank(
        T=T,
        pw50=pw50,
        mode=mode,
        fs=fs,
        half_window_symbols=half_window_symbols,
        frac_bins=frac_bins,
    )

    p_clean = np.zeros(num_samples, dtype=float)
    kernel_len = kernel_bank.shape[1]

    for k in range(total_len):
        total_jitter = float(delta_t[k] + tau_k[k])
        shift = k * T_eff + total_jitter
        shift_samples = shift * float(fs)

        center = int(np.floor(shift_samples))
        frac = shift_samples - center
        frac_idx = int(np.round(frac * (frac_bins - 1)))
        frac_idx = int(np.clip(frac_idx, 0, frac_bins - 1))

        kernel = a[k] * kernel_bank[frac_idx]

        i0 = center - half_window_samples
        i1 = center + half_window_samples + 1

        j0 = 0
        j1 = kernel_len
        if i0 < 0:
            j0 = -i0
            i0 = 0
        if i1 > num_samples:
            j1 -= i1 - num_samples
            i1 = num_samples

        if i0 < i1 and j0 < j1:
            p_clean[i0:i1] += kernel[j0:j1]

    # 6. Add AWGN noise
    vp = 1.0
    sigma_n = vp / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, sigma_n, len(t))
    p_total = p_clean + noise

    return t, p_total, a, b, data_start
