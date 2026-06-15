import pytest
import numpy as np

from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target


def _make_opsps(target_response):
    return OPSPVA(
        taps=[0.0, 1.0, 0.0],
        T=1.0,
        alpha=0.001,
        beta=0.0001,
        samples_per_symbol=2,
        mu_fse=0.0,
        target_response=target_response,
    )


@pytest.mark.parametrize("mode", ["pmr", "lmr"])
def test_slicer_uses_sign_quantization_for_gpr_target_levels(mode):
    opsps = _make_opsps(get_gpr_target(mode=mode, oversampled=True))

    for level in opsps._slicer_levels:
        bit = opsps._slice_bit(float(level), fallback=-1.0)
        if level > 0.0:
            assert bit == 1.0
        elif level < 0.0:
            assert bit == -1.0
        else:
            assert bit == -1.0


def test_pr4_zero_level_uses_fallback_bit():
    opsps = _make_opsps(get_gpr_target(mode="pr4", oversampled=True))

    assert opsps._slice_bit(0.0, fallback=1.0) == 1.0
    assert opsps._slice_bit(0.0, fallback=-1.0) == -1.0


def test_slicer_supports_nonzero_threshold():
    opsps = _make_opsps(get_gpr_target(mode="pmr", oversampled=True))

    assert opsps._slice_bit(0.10, fallback=1.0, threshold=0.20) == -1.0
    assert opsps._slice_bit(0.30, fallback=-1.0, threshold=0.20) == 1.0


def test_estimate_slicer_threshold_from_known_reference():
    opsps = _make_opsps(get_gpr_target(mode="pmr", oversampled=True))
    detector_input = np.array([-0.6, -0.4, 0.2, 0.4], dtype=float)
    ted_reference = np.array([-1.0, -1.0, 1.0, 1.0], dtype=float)

    threshold = opsps._estimate_slicer_threshold(detector_input, ted_reference)
    assert threshold is not None
    assert abs(float(threshold) - (-0.1)) < 1e-12


def test_multilevel_nearest_can_recover_unique_bit_mapping():
    opsps = OPSPVA(
        taps=[0.0, 1.0, 0.0],
        T=1.0,
        alpha=0.001,
        beta=0.0001,
        samples_per_symbol=2,
        mu_fse=0.0,
        target_response=get_gpr_target(mode="pmr", oversampled=True),
        slicer_mode="multilevel_nearest",
    )

    # Positive/negative realizable levels should map back to corresponding bits.
    pos_level = float(np.max(opsps._slicer_levels))
    neg_level = float(np.min(opsps._slicer_levels))
    assert opsps._slice_bit_from_level(pos_level, fallback=-1.0) == 1.0
    assert opsps._slice_bit_from_level(neg_level, fallback=1.0) == -1.0


def test_lookahead_threshold_matches_known_part_sum():
    target = np.array([1.0, 1.421, 1.076, 0.451, 0.097], dtype=float)
    opsps = OPSPVA(
        taps=[0.0, 1.0, 0.0],
        T=1.0,
        alpha=0.001,
        beta=0.0001,
        samples_per_symbol=2,
        mu_fse=0.0,
        target_response=target,
        slicer_mode="lookahead_dynamic",
    )

    decision_path = [1.0, -1.0, 1.0, -1.0]
    expected = target[1] * decision_path[-1]
    expected += target[2] * decision_path[-2]
    expected += target[3] * decision_path[-3]
    expected += target[4] * decision_path[-4]
    threshold = opsps._compute_lookahead_threshold(decision_path)
    assert abs(float(threshold) - float(expected)) < 1e-12
