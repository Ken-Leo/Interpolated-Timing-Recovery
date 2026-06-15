import numpy as np

from src.channel.channel_model import (
    generate_preamble,
    synthesize_readback_signal,
)
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.metrics import compute_ber, compute_fse_metrics, compute_tau_convergence


class TestSystemSmoke:
    def setup_method(self):
        self.length = 200
        self.preamble_len = 100
        self.T = 1.0
        self.pw50 = 2.5
        self.mode = "pmr"
        self.snr_db = 25.0
        self.fs = 100
        self.sigma_j = 0.03
        self.sigma_w = 0.005
        self.freq_offset = 0.004
        self.alpha = 0.0107
        self.beta = 0.000309
        self.taps = [
            -0.04014,
            0.090561,
            -0.109755,
            0.094373,
            -0.183844,
            0.152992,
            0.35265,
            -0.755096,
            -0.657579,
            0.955242,
            0.45171,
            0.46566,
            0.799206,
            -0.638464,
            -0.757566,
            0.373477,
            0.07353,
            0.137979,
            -0.351151,
            0.216157,
            -0.060725,
        ]

    def test_channel_frontend_smoke(self):
        t, r_raw, a_full, _, data_start = synthesize_readback_signal(
            length=self.length,
            T=self.T,
            pw50=self.pw50,
            mode=self.mode,
            sigma_j=self.sigma_j,
            sigma_w=self.sigma_w,
            freq_offset=self.freq_offset,
            snr_db=self.snr_db,
            fs=self.fs,
            seed=42,
            preamble_length=self.preamble_len,
            preamble_pattern="4T",
        )

        expected_preamble = generate_preamble(self.preamble_len, "4T")
        assert len(t) == len(r_raw)
        assert len(a_full) == self.length + self.preamble_len
        assert data_start == self.preamble_len
        np.testing.assert_array_equal(a_full[: self.preamble_len], expected_preamble)

        lpf = create_lowpass_filter(T=self.T, N=2, fs=self.fs)
        r_filtered = lpf.filter(r_raw)

        assert len(r_filtered) == len(r_raw)
        assert np.all(np.isfinite(r_filtered))
        assert np.std(r_filtered) > 0.0

    def test_opsps_va_end_to_end_smoke(self):
        _, r_raw, a_full, _, data_start = synthesize_readback_signal(
            length=self.length,
            T=self.T,
            pw50=self.pw50,
            mode=self.mode,
            sigma_j=self.sigma_j,
            sigma_w=self.sigma_w,
            freq_offset=self.freq_offset,
            snr_db=self.snr_db,
            fs=self.fs,
            seed=42,
            preamble_length=self.preamble_len,
            preamble_pattern="4T",
        )
        a_data = a_full[data_start:]

        lpf = create_lowpass_filter(T=self.T, N=2, fs=self.fs)
        r_filtered = lpf.filter(r_raw)

        opsps_va = OPSPVA(
            taps=self.taps,
            T=self.T,
            alpha=self.alpha,
            beta=self.beta,
            samples_per_symbol=self.fs,
            mu_fse=1e-4,
        )
        a_hat, tau_hat, fse_mse = opsps_va.decode(r_filtered, ground_truth=None)

        # Paper OPSP-VA: output length = L + K/2 due to FSE delay tail.
        half_k = (len(self.taps) - 1) // 4
        assert len(a_hat) == len(a_full) + half_k
        assert len(tau_hat) == len(a_full) + half_k
        assert np.all(np.isfinite(a_hat))
        assert np.all(np.isfinite(tau_hat))
        assert np.all(np.isfinite(fse_mse))

        # Compare only the first L valid symbols (ignore FSE delay tail)
        ber, errors, shift = compute_ber(
            a_data, a_hat[data_start : data_start + len(a_data)]
        )
        tau_metrics = compute_tau_convergence(tau_hat)
        fse_metrics = compute_fse_metrics(fse_mse)

        assert errors >= 0
        assert shift == 0 or abs(shift) <= 5
        assert ber < 0.5  # smoke test only: verify detection is not inverted
        assert abs(tau_metrics["tau_final"]) < 5.0  # extended tail may accumulate drift
        assert np.isfinite(fse_metrics["mse_initial"])
        assert np.isfinite(fse_metrics["mse_final"])
        assert fse_metrics["mse_final"] >= 0.0


def test_channel_mode_aliases_match_pmr_waveform():
    kwargs = dict(
        length=32,
        T=1.0,
        pw50=2.5,
        sigma_j=0.0,
        sigma_w=0.0,
        freq_offset=0.0,
        snr_db=200.0,
        fs=100,
        seed=7,
        preamble_length=8,
        preamble_pattern="4T",
    )

    _, signal_pmr, a_pmr, b_pmr, data_start_pmr = synthesize_readback_signal(
        mode="pmr",
        **kwargs,
    )
    (
        _,
        signal_perpendicular,
        a_perpendicular,
        b_perpendicular,
        data_start_perpendicular,
    ) = synthesize_readback_signal(
        mode="perpendicular",
        **kwargs,
    )

    np.testing.assert_allclose(signal_pmr, signal_perpendicular)
    np.testing.assert_array_equal(a_pmr, a_perpendicular)
    np.testing.assert_array_equal(b_pmr, b_perpendicular)
    assert data_start_pmr == data_start_perpendicular
