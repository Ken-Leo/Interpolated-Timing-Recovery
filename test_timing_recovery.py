import numpy as np
from src.opsps_va.ted import get_oversampled_samples, early_late_ted

def test_interpolation_accuracy():
    """Verify cubic interpolation recovers known signal values."""
    print("Testing Cubic Interpolation Accuracy...")

    signal = np.zeros(200)
    peak_idx = 100
    for i in range(200):
        signal[i] = np.exp(-((i - peak_idx)**2) / 500.0)

    sps = 100

    # Test 1: At peak with tau=0, center should be ~1.0
    x_early, x_center, x_late = get_oversampled_samples(signal, 1, 0.0, sps)
    print(f"  tau=0.0: center={x_center:.4f} (expected ~1.0)")
    assert abs(x_center - 1.0) < 0.01, f"Center sample too far from peak: {x_center}"

    # Test 2: Symmetric pulse => early == late at tau=0
    print(f"  tau=0.0: early={x_early:.4f}, late={x_late:.4f}")
    assert abs(x_early - x_late) < 0.01, f"Early/late not symmetric: {x_early} vs {x_late}"

    # Test 3: TED error is zero at symmetric position
    eps = early_late_ted(x_early, x_late, d_hat=1.0)
    print(f"  TED error at tau=0: {eps:.6f}")
    assert abs(eps) < 0.01, f"TED error should be ~0 at symmetry: {eps}"

    # Test 4: Positive tau shifts center away from peak
    x_e2, x_c2, x_l2 = get_oversampled_samples(signal, 1, 0.1, sps)
    print(f"  tau=0.1: center={x_c2:.4f} (should be < 1.0)")
    assert x_c2 < x_center, f"Center should decrease with positive tau"

    print("SUCCESS: All interpolation tests passed.")

if __name__ == "__main__":
    test_interpolation_accuracy()
