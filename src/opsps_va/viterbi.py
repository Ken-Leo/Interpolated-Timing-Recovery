import numpy as np
from typing import List, Tuple
from .trellis import GPRTrellis
from .ted import early_late_ted, mueller_muller_ted
from .survivor_pll import SurvivorPLL, PLLState
from ..frontend.fse import FractionallySpacedEqualizer


class OPSPVA:
    """
    Oversampled Per-Survivor Processing Viterbi Algorithm (OPSP-VA).

    Architecture per Kovintavewat et al. (2003):
    1. Each survivor path maintains its own sampling phase (tau) and frequency (theta).
    2. The high-rate signal is interpolated to T/2-rate per path using the path's tau.
    3. The FSE operates on T/2-rate samples to produce T/2-rate equalized output.
    4. The TED uses T/2-rate equalized samples: epsilon = d_hat * (x_{2k+1} - x_{2k-1}).
    5. The PLL for each state is updated only when that path survives.
    6. FSE weights are updated using decision-directed LMS targeting PR-IV response.
    """

    def __init__(
        self,
        taps: List[float],
        T: float,
        alpha: float,
        beta: float,
        samples_per_symbol: int = 100,
        mu_fse: float = 0.001,
        ted_mode: str = "early_late",
        ted_data_clip: float | None = None,
        pll_theta_leak: float = 0.0,
        pll_theta_clip: float | None = None,
        target_response: np.ndarray | None = None,
        final_output_source: str = "traceback",
        traceback_depth: int | None = None,
        detector_input_scaling: str = "global_mapminmax",
        slicer_threshold: float | None = 0.0,
        slicer_mode: str = "binary_threshold",
        lookahead_threshold_lms_mu: float = 0.0,
        lookahead_threshold_offset_clip: float | None = 2.0,
        metric_scaling_debug: bool = False,
        metric_scaling_align_tol: float = 2e-2,
        fse_use_nlms: bool = False,
        fse_nlms_eps: float = 1e-8,
        fse_error_clip: float | None = None,
        fse_tap_leak: float = 0.0,
        training_mode: str = "auto",
        fse_oversampling_ratio: int = 2,
    ):
        if target_response is None:
            self.target_response = np.array([1.0, 0.0, -1.0], dtype=float)
        else:
            self.target_response = np.asarray(target_response, dtype=float)
            if self.target_response.ndim != 1 or self.target_response.size < 1:
                raise ValueError(
                    "target_response must be a 1-D array with at least one tap"
                )

        self.trellis = GPRTrellis(self.target_response)
        self.fse = FractionallySpacedEqualizer(
            taps,
            T,
            N=fse_oversampling_ratio,
            project_taps=self._should_project_fse_taps(),
            symmetry_strength=1.0,
            enforce_center_max=True,
            use_nlms=fse_use_nlms,
            nlms_eps=fse_nlms_eps,
            error_clip=fse_error_clip,
            tap_leak=fse_tap_leak,
        )
        self.pll = SurvivorPLL(
            alpha,
            beta,
            theta_leak=pll_theta_leak,
            theta_clip=pll_theta_clip,
        )
        self.T = T
        self.sps = samples_per_symbol  # samples per symbol in the input signal
        self.mu_fse = mu_fse

        if training_mode not in {"auto", "decision_directed"}:
            raise ValueError(
                "training_mode must be either 'auto' or 'decision_directed'"
            )
        self.training_mode = training_mode

        if ted_mode not in {"early_late", "mm"}:
            raise ValueError("ted_mode must be either 'early_late' or 'mm'")
        self.ted_mode = ted_mode
        if self.fse.N == 1 and self.ted_mode == "early_late":
            raise ValueError(
                "early_late TED requires N>=2 (T/2-rate or finer samples). "
                "For N=1 (symbol-rate sampling), use ted_mode='mm' (Mueller-Müller)."
            )
        if ted_data_clip is not None and ted_data_clip <= 0.0:
            raise ValueError("ted_data_clip must be positive or None")
        self.ted_data_clip = ted_data_clip
        if final_output_source == "detector":
            final_output_source = "slicer"
        if final_output_source not in {"traceback", "slicer"}:
            raise ValueError(
                "final_output_source must be either 'traceback' or 'slicer'"
            )
        if detector_input_scaling not in {"none", "global_mapminmax"}:
            raise ValueError(
                "detector_input_scaling must be one of 'none' or 'global_mapminmax'"
            )
        self.final_output_source = final_output_source
        self.traceback_depth = traceback_depth
        self.detector_input_scaling = detector_input_scaling
        if slicer_mode not in {
            "binary_threshold",
            "multilevel_nearest",
            "lookahead_dynamic",
        }:
            raise ValueError(
                "slicer_mode must be one of 'binary_threshold', "
                "'multilevel_nearest', or 'lookahead_dynamic'"
            )
        self.slicer_mode = slicer_mode
        if slicer_threshold is not None:
            slicer_threshold = float(slicer_threshold)
        self.slicer_threshold = slicer_threshold
        if lookahead_threshold_lms_mu < 0.0:
            raise ValueError("lookahead_threshold_lms_mu must be non-negative")
        self.lookahead_threshold_lms_mu = float(lookahead_threshold_lms_mu)
        self.metric_scaling_debug = bool(metric_scaling_debug)
        if metric_scaling_align_tol <= 0.0:
            raise ValueError("metric_scaling_align_tol must be positive")
        self.metric_scaling_align_tol = float(metric_scaling_align_tol)
        if lookahead_threshold_offset_clip is not None:
            lookahead_threshold_offset_clip = float(lookahead_threshold_offset_clip)
            if lookahead_threshold_offset_clip <= 0.0:
                raise ValueError(
                    "lookahead_threshold_offset_clip must be positive or None"
                )
        self.lookahead_threshold_offset_clip = lookahead_threshold_offset_clip

        # Mandatory slicer branch:
        # quantize equalized y_k to the nearest valid target-output level,
        # then map it to a bipolar hard decision for TED/FSE adaptation.
        # This is the per-symbol symbol-estimation step, not the final
        # sequence-level Viterbi traceback output.
        self._slicer_levels = self._build_slicer_levels()
        self._slicer_level_to_bit = self._build_slicer_level_to_bit_map()
        # Expose latest adaptation error sequence for debugging/visualization.
        self.last_w_k: np.ndarray | None = None
        self.last_d_hat_k: np.ndarray | None = None
        self.last_slicer_hat_a: np.ndarray | None = None
        self.last_traceback_a_hat: np.ndarray | None = None
        self.last_sliding_traceback_a_hat: np.ndarray | None = None
        self.last_stage_y_k: np.ndarray | None = None
        self.last_stage_branch_output: np.ndarray | None = None
        self.last_stage_branch_input: np.ndarray | None = None
        self.last_stage_metric_margin: np.ndarray | None = None
        self.last_detector_input_y_k: np.ndarray | None = None
        self.last_metric_probe_y_k_raw: np.ndarray | None = None
        self.last_metric_probe_y_k: np.ndarray | None = None
        self.last_mapminmax_debug_records: list[dict[str, float | int | str | bool]] = (
            []
        )
        self.last_detector_input_scale: float = 1.0
        self.last_detector_input_bias: float = 0.0
        self.last_slicer_threshold: float = 0.0
        self.last_lookahead_threshold_offset: float = 0.0

    def _get_traceback_depth(self) -> int:
        if self.traceback_depth is not None:
            return max(1, int(self.traceback_depth))
        return max(16, 5 * max(1, self.trellis.memory))

    def _traceback_window(
        self,
        prestate_hist: list[np.ndarray],
        input_hist: list[np.ndarray],
        metrics: np.ndarray,
    ) -> list[float]:
        if not prestate_hist:
            return []

        state = int(np.argmin(metrics))
        traced_inputs: list[float] = []
        for stage_idx in range(len(prestate_hist) - 1, -1, -1):
            traced_inputs.append(float(input_hist[stage_idx][state]))
            state = int(prestate_hist[stage_idx][state])

        traced_inputs.reverse()
        return traced_inputs

    @staticmethod
    def _fit_mapminmax(
        x: np.ndarray, out_min: float, out_max: float
    ) -> Tuple[float, float] | None:
        if x.size < 2:
            return None

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if abs(x_max - x_min) < 1e-9:
            return None

        scale = (out_max - out_min) / (x_max - x_min)
        bias = out_min - scale * x_min
        return float(scale), float(bias)

    def _should_project_fse_taps(self) -> bool:
        """Enable tap projection only when the active target is symmetric.

        The old center-symmetric, center-max projection is a useful heuristic for
        symmetric partial-response targets, but it should not be forced onto
        asymmetric PR/PMR/LMR targets.
        """
        return bool(np.allclose(self.target_response, self.target_response[::-1]))

    def _detector_target_range(self) -> Tuple[float, float]:
        """Return the theoretical detector-domain range of the active target.

        For a target response h[k], the ideal branch output is
            y = sum_k h[k] * a_{n-k},   a_i in {+1, -1}.
        Its extremal values are therefore bounded by ±sum(|h[k]|).
        This gives the real target-response range to use for detector-domain
        scaling, independent of the trellis branch enumeration.
        """
        span = float(np.sum(np.abs(self.target_response)))
        return -span, span

    def _build_slicer_levels(self) -> np.ndarray:
        """Collect all realizable branch-output levels from the trellis.

        The level set is derived from the active partial-response target via the
        trellis branch outputs, so it automatically matches PR-IV or any other
        user-supplied target response.
        """
        levels: set[float] = set()
        for s in range(self.trellis.num_states):
            for trans in self.trellis.get_transitions_to(s):
                levels.add(float(trans.output_symbol))

        # Keep deterministic order for reproducibility.
        return np.array(sorted(levels), dtype=float)

    @staticmethod
    def _slicer_level_key(level: float) -> float:
        return round(float(level), 12)

    def _build_slicer_level_to_bit_map(self) -> dict[float, float | None]:
        """Map each realizable slicer level to its corresponding input bit.

        For general GPR targets, the sliced output level can carry more
        information than its sign alone. When a level maps uniquely to one input
        bit across the trellis, return that bit. If the level is ambiguous (for
        example PR-IV level 0), store None and let the runtime fallback decide.
        """
        level_to_bits: dict[float, set[float]] = {}
        for state in range(self.trellis.num_states):
            for trans in self.trellis.get_transitions_to(state):
                key = self._slicer_level_key(trans.output_symbol)
                level_to_bits.setdefault(key, set()).add(float(trans.input_bit))

        return {
            key: next(iter(bits)) if len(bits) == 1 else None
            for key, bits in level_to_bits.items()
        }

    def _slice_symbol_level(self, y_value: float) -> float:
        """Nearest-neighbor symbol slicing on the realizable output levels."""
        idx = int(np.argmin(np.abs(self._slicer_levels - y_value)))
        return float(self._slicer_levels[idx])

    def _slice_bit(
        self, y_value: float, fallback: float = 1.0, threshold: float = 0.0
    ) -> float:
        """Estimate bipolar write-data bit from detector input y_k.

        This slicer intentionally performs pure sign quantization on the scaled
        detector input y_k, independent of target-response output levels.
        """
        if y_value > threshold:
            return 1.0
        if y_value < threshold:
            return -1.0
        return float(fallback)

    def _slice_bit_ddf(
        self,
        y_value: float,
        past_decisions: list[float],
        fallback: float = 1.0,
        threshold: float = 0.0,
    ) -> float:
        """DDF (Decision Feedback) slicer: cancel ISI from past decisions first.

        residual = y_k - Σ_{j=1}^{L-1} h_j · â_{k-j}
        â_k = sign(residual / h_0)

        This makes the slicer much more reliable in DD mode because the strong
        ISI from the GPR target is removed before the sign test.
        """
        isi = 0.0
        for j in range(1, len(self.target_response)):
            if j - 1 < len(past_decisions):
                isi += self.target_response[j] * float(past_decisions[j - 1])

        residual = y_value - isi
        h0 = self.target_response[0]

        if h0 > 0:
            return 1.0 if residual >= threshold else -1.0
        else:
            return 1.0 if residual <= threshold else -1.0

    def _slice_bit_from_level(self, level: float, fallback: float = 1.0) -> float:
        input_bit = self._slicer_level_to_bit.get(self._slicer_level_key(level))
        if input_bit is not None:
            return float(input_bit)
        return float(fallback)

    def _compute_lookahead_threshold(self, decision_path: list[float]) -> float:
        threshold = 0.0
        for tap_idx, coeff in enumerate(self.target_response[1:], start=1):
            bit_idx = len(decision_path) - tap_idx
            if bit_idx >= 0:
                threshold += coeff * float(decision_path[bit_idx])
        return float(threshold)

    @staticmethod
    def _estimate_slicer_threshold(
        detector_input: np.ndarray,
        ted_reference: np.ndarray | None,
    ) -> float | None:
        if detector_input.size == 0:
            return None

        if ted_reference is not None and ted_reference.size == detector_input.size:
            known = np.isfinite(ted_reference)
            if np.any(known):
                pos = detector_input[known & (ted_reference >= 0.0)]
                neg = detector_input[known & (ted_reference < 0.0)]
                if pos.size > 0 and neg.size > 0:
                    return float(0.5 * (np.median(pos) + np.median(neg)))

        return float(np.median(detector_input))

    def _ideal_target_output(self, slicer_hat_a: np.ndarray, symbol_idx: int) -> float:
        y_ideal = 0.0
        for tap_idx, coeff in enumerate(self.target_response):
            bit_idx = symbol_idx - tap_idx
            if bit_idx >= 0:
                y_ideal += coeff * slicer_hat_a[bit_idx]
        return float(y_ideal)

    def _extract_t2_sample(self, r_m: np.ndarray, t2_idx: int, tau: float) -> float:
        """
        Extract a single T/N-rate sample from the high-rate signal.

        T/N-rate sample index j corresponds to time j * T/N + tau * T.
        On the fs-rate grid: position = j * (sps/N) + tau * sps.

        Args:
            r_m: High-rate signal.
            t2_idx: Index on the T/N-rate grid (called T/2 legacy for N=2).
            tau: Timing offset in symbol periods.

        Returns:
            Interpolated sample value.
        """
        rate_step_sa = self.sps / float(self.fse.N)
        pos_float = t2_idx * rate_step_sa + tau * self.sps

        i = int(np.floor(pos_float))
        frac = pos_float - i

        def get_val(idx: int) -> float:
            if 0 <= idx < len(r_m):
                return r_m[idx]
            return 0.0

        # Cubic interpolation
        y0, y1, y2, y3 = get_val(i - 1), get_val(i), get_val(i + 1), get_val(i + 2)
        l0 = ((frac - 0) * (frac - 1) * (frac - 2)) / (-6.0)
        l1 = ((frac + 1) * (frac - 1) * (frac - 2)) / 2.0
        l2 = ((frac + 1) * frac * (frac - 2)) / (-2.0)
        l3 = ((frac + 1) * frac * (frac - 1)) / 6.0

        return l0 * y0 + l1 * y1 + l2 * y2 + l3 * y3

    def _equalize_at_t2(
        self, r_m: np.ndarray, t2_center: int, tau: float
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the equalized output at a specific T/2-rate position.

        The FSE has tap_len taps at T/2 spacing. The output at T/2-rate index
        t2_center is: y = sum(w[j] * r[t2_center - tap_len//2 + j])

        Args:
            r_m: High-rate signal.
            t2_center: T/2-rate index for the center of the equalizer window.
            tau: Timing offset.

        Returns:
            (y_k, window): Equalized output and the T/2-rate sample window.
        """
        tap_len = len(self.fse.taps)
        half_tap = tap_len // 2
        window = np.zeros(tap_len)

        for j in range(tap_len):
            t2_idx = t2_center - half_tap + j
            window[j] = self._extract_t2_sample(r_m, t2_idx, tau)

        y_k = np.dot(window, self.fse.taps)
        return y_k, window

    def equalize_at_t2(
        self, r_m: np.ndarray, t2_center: int, tau: float
    ) -> Tuple[float, np.ndarray]:
        """Public probe helper for equalized T/N-rate samples.

        This is intended for diagnostics and offline parameter calibration,
        where external scripts need access to the same interpolation and FSE
        windowing used by the decoder.

        The legacy name 'equalize_at_t2' is retained for backward compatibility;
        it works for any N (t2_center is the sample index on the T/N-rate grid).
        """
        return self._equalize_at_t2(r_m, t2_center, tau)

    def _decode_once(
        self,
        r_m: np.ndarray,
        ground_truth: np.ndarray | None = None,
        ted_reference: np.ndarray | None = None,
        initial_detector_input_scale: float = 1.0,
        initial_detector_input_bias: float = 0.0,
        initial_slicer_threshold: float = 0.0,
        initial_lookahead_threshold_offset: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_states = self.trellis.num_states
        # General delay in symbol periods: group delay of linear-phase FIR
        # with M taps at T/N spacing = (M-1)/(2*N) symbol periods.
        half_k = (len(self.fse.taps) - 1) // (2 * self.fse.N)
        total_symbols = len(r_m) // self.sps + max(0, half_k)
        phi = np.full(num_states, np.inf)
        phi[0] = 0.0
        traceback_depth = self._get_traceback_depth()
        prestate_hist: list[np.ndarray] = []
        input_hist: list[np.ndarray] = []
        sliding_traceback_output: list[float] = []
        detector_input_scale = float(initial_detector_input_scale)
        detector_input_bias = float(initial_detector_input_bias)
        slicer_threshold = float(initial_slicer_threshold)
        lookahead_threshold_offset = float(initial_lookahead_threshold_offset)
        survivor_info = {
            s: {
                "pll": PLLState(),
                "path": [],
                "decision_path": [],
                "slicer_path": [],
                "tau_hist": [],
                "last_window": None,
                "last_x_m_center": 0.0,
                "last_x_m_early": 0.0,
                "last_x_m_late": 0.0,
                "last_branch_output": 0.0,
                "last_branch_input": 1.0,
                "last_slicer_bit": 1.0,
                "last_slicer_threshold": 0.0,
                "last_ted_d_hat": 0.0,
            }
            for s in range(num_states)
        }

        slicer_hat_a = np.zeros(total_symbols)
        adaptation_hat_a = np.zeros(total_symbols)
        d_hat_k_hist = np.zeros(total_symbols)
        w_k_hist = np.zeros(total_symbols)
        fse_errors = np.zeros(total_symbols)
        stage_y_k = np.zeros(total_symbols)
        detector_input_y_k = np.zeros(total_symbols)
        metric_probe_y_k_raw: list[float] = []
        stage_branch_output = np.zeros(total_symbols)
        stage_branch_input = np.ones(total_symbols)
        stage_metric_margin = np.full(total_symbols, np.nan)

        for k in range(total_symbols):
            new_phi = np.full(num_states, np.inf)
            new_survivor_info = {}
            curr_prestate = np.zeros(num_states, dtype=int)
            curr_input = np.ones(num_states, dtype=float)

            # Cache repeated equalizer probes within this symbol stage.
            # Multiple outgoing branches can share the same prestate and T/2 index.
            eq_probe_cache: dict[tuple[int, int], tuple[float, np.ndarray]] = {}

            for s in range(num_states):
                transitions = self.trellis.get_transitions_to(s)
                best_metric = np.inf
                best_trans = None
                best_pll_state: PLLState | None = None
                best_window = None
                best_x_m_center = 0.0
                best_x_m_early = 0.0
                best_x_m_late = 0.0
                best_branch_output = 0.0
                best_branch_input = 1.0
                best_decision_path = []
                best_slicer_path = []
                best_slicer_bit = 1.0
                best_slicer_threshold = 0.0
                best_ted_d_hat = 0.0
                second_best_metric = np.inf

                for trans in transitions:
                    pre = trans.prestate
                    if phi[pre] == np.inf:
                        continue

                    # ── Branch pruning: only from preamble (ted_reference) ──
                    branch_known = None
                    if (
                        ted_reference is not None
                        and k < len(ted_reference)
                        and np.isfinite(ted_reference[k])
                    ):
                        branch_known = 1.0 if ted_reference[k] >= 0.0 else -1.0

                    if branch_known is not None and trans.input_bit != branch_known:
                        continue

                    # ── TED decision symbol: preamble → ground_truth → slicer ──
                    ted_known = None
                    if branch_known is not None:
                        ted_known = branch_known
                    elif (
                        self.training_mode != "decision_directed"
                        and ground_truth is not None
                        and k < len(ground_truth)
                        and np.isfinite(ground_truth[k])
                    ):
                        ted_known = 1.0 if ground_truth[k] >= 0.0 else -1.0

                    decimation_n = self.fse.N
                    x_m_center_idx = k * decimation_n
                    center_key = (pre, x_m_center_idx)
                    center_probe = eq_probe_cache.get(center_key)
                    if center_probe is None:
                        # Paper: tau_k = 0 if k < K/2 (initial phase unknown)
                        tau_current = (
                            0.0 if k < half_k else survivor_info[pre]["pll"].tau
                        )
                        center_probe = self._equalize_at_t2(
                            r_m, x_m_center_idx, tau_current
                        )
                        eq_probe_cache[center_key] = center_probe
                    x_m_center, window = center_probe
                    y_k_raw = x_m_center
                    y_k = detector_input_scale * y_k_raw + detector_input_bias
                    metric_probe_y_k_raw.append(float(y_k_raw))
                    if self.slicer_mode == "lookahead_dynamic":
                        slicer_threshold_k = (
                            self._compute_lookahead_threshold(
                                survivor_info[pre]["decision_path"]
                            )
                            + lookahead_threshold_offset
                        )
                        slicer_bit = self._slice_bit(
                            y_k,
                            fallback=survivor_info[pre]["last_slicer_bit"],
                            threshold=slicer_threshold_k,
                        )
                    elif self.slicer_mode == "multilevel_nearest":
                        slicer_threshold_k = 0.0
                        sliced_level = self._slice_symbol_level(y_k)
                        slicer_bit = self._slice_bit_from_level(
                            sliced_level,
                            fallback=survivor_info[pre]["last_slicer_bit"],
                        )
                    else:
                        slicer_threshold_k = slicer_threshold
                        # Use DDF slicing to cancel ISI from past decisions.
                        # This is critical for DD mode — without it, the strong
                        # GPR target ISI causes frequent slicer errors.
                        slicer_bit = self._slice_bit_ddf(
                            y_k,
                            past_decisions=survivor_info[pre]["decision_path"],
                            fallback=survivor_info[pre]["last_slicer_bit"],
                            threshold=slicer_threshold_k,
                        )

                    if ted_known is not None:
                        decision_symbol = ted_known
                    else:
                        decision_symbol = slicer_bit

                    candidate_decision_path = survivor_info[pre]["decision_path"] + [
                        decision_symbol
                    ]
                    ted_d_hat = self._ideal_target_output(
                        np.asarray(candidate_decision_path, dtype=float),
                        len(candidate_decision_path) - 1,
                    )

                    rho = (y_k - trans.output_symbol) ** 2
                    metric = phi[pre] + rho

                    if metric < best_metric:
                        second_best_metric = best_metric
                        best_metric = metric
                        best_trans = trans
                        best_window = window
                        best_branch_output = float(trans.output_symbol)
                        best_branch_input = float(trans.input_bit)

                        if self.ted_mode == "early_late":
                            early_key = (pre, x_m_center_idx + 1)
                            early_probe = eq_probe_cache.get(early_key)
                            if early_probe is None:
                                tau_current = survivor_info[pre]["pll"].tau
                                early_probe = self._equalize_at_t2(
                                    r_m, x_m_center_idx + 1, tau_current
                                )
                                eq_probe_cache[early_key] = early_probe
                            x_m_early, _ = early_probe

                            late_key = (pre, x_m_center_idx - 1)
                            late_probe = eq_probe_cache.get(late_key)
                            if late_probe is None:
                                # Paper Eq.(3): x(kT - T/2) uses tau_{k-1},
                                # not tau_k.  tau_hist[-2] is the previous
                                # survivor's phase estimate.
                                tau_hist = survivor_info[pre]["tau_hist"]
                                tau_late = tau_hist[-2] if len(tau_hist) >= 2 else 0.0
                                late_probe = self._equalize_at_t2(
                                    r_m, x_m_center_idx - 1, tau_late
                                )
                                eq_probe_cache[late_key] = late_probe
                            x_m_late, _ = late_probe
                            epsilon_k = early_late_ted(
                                x_m_early,
                                x_m_late,
                                ted_d_hat,
                            )
                        else:
                            x_m_early = 0.0
                            x_m_late = 0.0
                            d_hat_prev = survivor_info[pre]["last_ted_d_hat"]
                            x_m_prev = survivor_info[pre]["last_x_m_center"]
                            epsilon_k = mueller_muller_ted(
                                x_current=y_k,
                                x_previous=x_m_prev,
                                d_current=ted_d_hat,
                                d_previous=d_hat_prev,
                            )

                        if ted_known is None and self.ted_data_clip is not None:
                            epsilon_k = float(
                                np.clip(
                                    epsilon_k, -self.ted_data_clip, self.ted_data_clip
                                )
                            )

                        best_pll_state = self.pll.update(
                            survivor_info[pre]["pll"], epsilon_k
                        )
                        best_x_m_center = y_k_raw
                        best_x_m_early = x_m_early
                        best_x_m_late = x_m_late
                        best_decision_path = candidate_decision_path
                        best_slicer_path = survivor_info[pre]["slicer_path"] + [
                            slicer_bit
                        ]
                        best_slicer_bit = slicer_bit
                        best_slicer_threshold = float(slicer_threshold_k)
                        best_ted_d_hat = ted_d_hat
                    elif metric < second_best_metric:
                        second_best_metric = metric

                if best_trans is not None:
                    tau_estimate = (
                        best_pll_state.tau
                        if best_pll_state is not None
                        else survivor_info[best_trans.prestate]["pll"].tau
                    )
                    new_phi[s] = best_metric
                    curr_prestate[s] = int(best_trans.prestate)
                    curr_input[s] = float(best_trans.input_bit)
                    new_survivor_info[s] = {
                        "pll": best_pll_state,
                        "path": survivor_info[best_trans.prestate]["path"]
                        + [best_trans.input_bit],
                        "decision_path": best_decision_path,
                        "slicer_path": best_slicer_path,
                        "tau_hist": survivor_info[best_trans.prestate]["tau_hist"]
                        + [tau_estimate],
                        "last_window": best_window,
                        "last_x_m_center": best_x_m_center,
                        "last_x_m_early": best_x_m_early,
                        "last_x_m_late": best_x_m_late,
                        "last_branch_output": best_branch_output,
                        "last_branch_input": best_branch_input,
                        "last_slicer_bit": best_slicer_bit,
                        "last_slicer_threshold": best_slicer_threshold,
                        "last_ted_d_hat": best_ted_d_hat,
                        "metric_margin": second_best_metric - best_metric,
                    }
                else:
                    new_phi[s] = np.inf
                    new_survivor_info[s] = {
                        "pll": PLLState(),
                        "path": [],
                        "decision_path": [],
                        "slicer_path": [],
                        "tau_hist": [],
                        "last_window": None,
                        "last_x_m_center": 0.0,
                        "last_x_m_early": 0.0,
                        "last_x_m_late": 0.0,
                        "last_branch_output": 0.0,
                        "last_branch_input": 1.0,
                        "last_slicer_bit": 1.0,
                        "last_slicer_threshold": 0.0,
                        "last_ted_d_hat": 0.0,
                        "metric_margin": np.nan,
                    }

            phi = new_phi
            survivor_info = new_survivor_info
            prestate_hist.append(curr_prestate)
            input_hist.append(curr_input)

            if len(prestate_hist) >= traceback_depth:
                traced_inputs = self._traceback_window(prestate_hist, input_hist, phi)
                if traced_inputs:
                    sliding_traceback_output.append(traced_inputs[0])
                    prestate_hist.pop(0)
                    input_hist.pop(0)

            best_state_k = int(np.argmin(phi))
            if best_state_k in survivor_info and survivor_info[best_state_k]["path"]:
                stage_y_k[k] = survivor_info[best_state_k]["last_x_m_center"]
                detector_input_y_k[k] = (
                    detector_input_scale * stage_y_k[k] + detector_input_bias
                )
                stage_branch_output[k] = survivor_info[best_state_k][
                    "last_branch_output"
                ]
                stage_branch_input[k] = survivor_info[best_state_k]["last_branch_input"]
                stage_metric_margin[k] = survivor_info[best_state_k]["metric_margin"]

                slicer_hat_a[k] = survivor_info[best_state_k]["last_slicer_bit"]
                slicer_threshold = float(
                    survivor_info[best_state_k].get("last_slicer_threshold", 0.0)
                )

                if (
                    self.training_mode != "decision_directed"
                    and ground_truth is not None
                    and k < len(ground_truth)
                    and np.isfinite(ground_truth[k])
                ):
                    a_hat_k = ground_truth[k]
                elif (
                    ted_reference is not None
                    and k < len(ted_reference)
                    and np.isfinite(ted_reference[k])
                ):
                    a_hat_k = 1.0 if ted_reference[k] >= 0.0 else -1.0
                else:
                    a_hat_k = survivor_info[best_state_k]["decision_path"][-1]

                adaptation_hat_a[k] = a_hat_k
                d_hat_k = self._ideal_target_output(adaptation_hat_a, k)
                d_hat_k_detector = detector_input_scale * d_hat_k + detector_input_bias
                d_hat_k_hist[k] = d_hat_k

                if (
                    self.slicer_mode == "lookahead_dynamic"
                    and self.lookahead_threshold_lms_mu > 0.0
                ):
                    if (
                        ted_reference is None
                        or k >= len(ted_reference)
                        or not np.isfinite(ted_reference[k])
                    ):
                        e_k = detector_input_y_k[k] - d_hat_k_detector
                        lookahead_threshold_offset += (
                            self.lookahead_threshold_lms_mu * e_k
                        )
                        if self.lookahead_threshold_offset_clip is not None:
                            lookahead_threshold_offset = float(
                                np.clip(
                                    lookahead_threshold_offset,
                                    -self.lookahead_threshold_offset_clip,
                                    self.lookahead_threshold_offset_clip,
                                )
                            )

                window = survivor_info[best_state_k]["last_window"]
                if window is not None:
                    y_k_actual = np.dot(window, self.fse.taps)
                    # Keep LMS residual in the same amplitude domain as detector metrics.
                    y_k_actual_detector = (
                        detector_input_scale * y_k_actual + detector_input_bias
                    )
                    w_k_raw = d_hat_k_detector - y_k_actual_detector
                    if self.fse.error_clip is not None:
                        w_k = float(
                            np.clip(
                                w_k_raw,
                                -self.fse.error_clip,
                                self.fse.error_clip,
                            )
                        )
                    else:
                        w_k = float(w_k_raw)
                    self.fse.update_weights(w_k, window, self.mu_fse)
                    w_k_hist[k] = w_k
                    fse_errors[k] = w_k**2

        best_final_state = int(np.argmin(phi))
        a_hat_traceback = np.array(survivor_info[best_final_state]["path"])
        if prestate_hist:
            sliding_traceback_output.extend(
                self._traceback_window(prestate_hist, input_hist, phi)
            )
        tau_hat = np.array(survivor_info[best_final_state]["tau_hist"])
        self.last_w_k = w_k_hist
        self.last_d_hat_k = d_hat_k_hist
        self.last_slicer_hat_a = slicer_hat_a.copy()
        self.last_traceback_a_hat = a_hat_traceback
        self.last_sliding_traceback_a_hat = np.asarray(
            sliding_traceback_output, dtype=float
        )
        self.last_stage_y_k = stage_y_k
        self.last_detector_input_y_k = detector_input_y_k
        metric_probe_y_k_raw_arr = np.asarray(metric_probe_y_k_raw, dtype=float)
        self.last_metric_probe_y_k_raw = metric_probe_y_k_raw_arr
        self.last_metric_probe_y_k = (
            detector_input_scale * metric_probe_y_k_raw_arr + detector_input_bias
        )
        self.last_stage_branch_output = stage_branch_output
        self.last_stage_branch_input = stage_branch_input
        self.last_stage_metric_margin = stage_metric_margin
        self.last_detector_input_scale = float(detector_input_scale)
        self.last_detector_input_bias = float(detector_input_bias)
        self.last_slicer_threshold = float(slicer_threshold)
        self.last_lookahead_threshold_offset = float(lookahead_threshold_offset)

        if self.final_output_source == "traceback":
            a_hat = a_hat_traceback.copy()
        else:
            a_hat = slicer_hat_a.copy()

        return a_hat, tau_hat, fse_errors

    def decode(
        self,
        r_m: np.ndarray,
        ground_truth: np.ndarray | None = None,
        ted_reference: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform OPSP-VA decoding.

        Args:
            r_m: High-rate readback signal (sps samples per symbol).
            ground_truth: Optional ground truth bits for supervised FSE training.
                Use finite values where supervision is enabled and NaN where
                adaptation should switch to decision-directed mode.
            ted_reference: Optional TED decision reference sequence.
                Use finite values where known (e.g., preamble), and NaN for
                unknown symbols where slicer decisions should be used.

        Returns:
            a_hat: Detected binary sequence from the configured output source.
                'traceback' returns the Viterbi survivor path.
                'slicer' returns the slicer output stream used by adaptation.
            tau_hat: Estimated timing offsets.
            fse_errors: MSE history of the FSE.
        """
        slicer_threshold = (
            float(self.slicer_threshold) if self.slicer_threshold is not None else 0.0
        )
        self.last_mapminmax_debug_records = []

        if self.detector_input_scaling != "global_mapminmax":
            a_hat, tau_hat, fse_errors = self._decode_once(
                r_m,
                ground_truth,
                ted_reference,
                initial_slicer_threshold=slicer_threshold,
            )

            if (
                self.slicer_mode == "binary_threshold"
                and self.slicer_threshold is None
                and self.last_detector_input_y_k is not None
            ):
                estimated = self._estimate_slicer_threshold(
                    self.last_detector_input_y_k, ted_reference
                )
                if estimated is not None and abs(estimated - slicer_threshold) > 1e-6:
                    a_hat, tau_hat, fse_errors = self._decode_once(
                        r_m,
                        ground_truth,
                        ted_reference,
                        initial_slicer_threshold=estimated,
                    )
            return a_hat, tau_hat, fse_errors

        target_min, target_max = self._detector_target_range()

        probe_a_hat, probe_tau_hat, _ = self._decode_once(
            r_m,
            ground_truth,
            ted_reference,
            initial_detector_input_scale=1.0,
            initial_detector_input_bias=0.0,
            initial_slicer_threshold=slicer_threshold,
            initial_lookahead_threshold_offset=self.last_lookahead_threshold_offset,
        )

        if self.last_stage_y_k is None or self.last_metric_probe_y_k_raw is None:
            return probe_a_hat, probe_tau_hat, np.array([])

        probe_source = self.last_metric_probe_y_k_raw
        if probe_source is None or probe_source.size < 2:
            probe_source = self.last_stage_y_k

        fit = self._fit_mapminmax(probe_source, target_min, target_max)
        if fit is None:
            return probe_a_hat, probe_tau_hat, np.array([])

        scale, bias = fit
        self.last_detector_input_scale = float(scale)
        self.last_detector_input_bias = float(bias)

        if self.slicer_mode == "binary_threshold" and self.slicer_threshold is None:
            estimated = self._estimate_slicer_threshold(
                (
                    self.last_detector_input_y_k
                    if self.last_detector_input_y_k is not None
                    else self.last_stage_y_k
                ),
                ted_reference,
            )
            if estimated is not None:
                slicer_threshold = float(estimated)

        a_hat, tau_hat, fse_errors = self._decode_once(
            r_m,
            ground_truth,
            ted_reference,
            initial_detector_input_scale=scale,
            initial_detector_input_bias=bias,
            initial_slicer_threshold=slicer_threshold,
            initial_lookahead_threshold_offset=self.last_lookahead_threshold_offset,
        )

        if self.last_stage_y_k is not None:
            self.last_detector_input_y_k = scale * self.last_stage_y_k + bias
        if self.last_metric_probe_y_k_raw is not None:
            self.last_metric_probe_y_k = scale * self.last_metric_probe_y_k_raw + bias

        self.last_mapminmax_debug_records = [
            {
                "iter": 1,
                "scale_in": 1.0,
                "bias_in": 0.0,
                "scale_next": float(scale),
                "bias_next": float(bias),
                "raw_min": float(np.min(probe_source)),
                "raw_max": float(np.max(probe_source)),
                "mapped_min": (
                    float(np.min(self.last_detector_input_y_k))
                    if self.last_detector_input_y_k is not None
                    else float("nan")
                ),
                "mapped_max": (
                    float(np.max(self.last_detector_input_y_k))
                    if self.last_detector_input_y_k is not None
                    else float("nan")
                ),
                "target_min": float(target_min),
                "target_max": float(target_max),
                "fit_source": (
                    "probe_raw"
                    if probe_source is self.last_metric_probe_y_k_raw
                    else "stage_y_k"
                ),
            }
        ]

        return a_hat, tau_hat, fse_errors
