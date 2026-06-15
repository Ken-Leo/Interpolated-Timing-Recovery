"""
Paper GPR target coefficients (from Kovintavewat 2003, Sec.4).

Includes longitudinal (LMR) and perpendicular (PMR) targets at ND=2.5,
designed at the SNR required to achieve BER = 10^-5.
"""

import numpy as np

# Longitudinal (LMR) targets from paper Sec.4.1
# Symbol-rate: H(D) = 1 + 0.613D - 0.478D^2 - 0.626D^3 - 0.291D^4
GPR_LMR_SYMBOL = np.array([1.0, 0.613, -0.478, -0.626, -0.291])

# Oversampled: H(D) = 1 + 0.419D - 0.441D^2 - 0.544D^3 - 0.268D^4
GPR_LMR_OVERSAMPLED = np.array([1.0, 0.419, -0.441, -0.544, -0.268])

# Symbol-rate PMR target (4-state trellis, length 5)
# H(D) = 1 + 1.429D + 1.097D^2 + 0.465D^3 + 0.099D^4
GPR_PMR_SYMBOL = np.array([1.0, 1.429, 1.097, 0.465, 0.099])

# Oversampled PMR target (4-state trellis, length 5)
# H(D) = 1 + 1.421D + 1.076D^2 + 0.451D^3 + 0.097D^4
GPR_PMR_OVERSAMPLED = np.array([1.0, 1.421, 1.076, 0.451, 0.097])

# PR-IV target (used in current implementation)
# H(D) = 1 - D^2 => output = a_k - a_{k-2}
# This corresponds to a 3-tap target [1, 0, -1]
PR_IV = np.array([1.0, 0.0, -1.0])


def get_gpr_target(mode="pmr", oversampled=True) -> np.ndarray:
    """Return the GPR target coefficients for the given configuration."""
    if mode in {"pmr", "perpendicular"}:
        if oversampled:
            return GPR_PMR_OVERSAMPLED.copy()
        else:
            return GPR_PMR_SYMBOL.copy()
    elif mode in {"lmr", "longitudinal"}:
        if oversampled:
            return GPR_LMR_OVERSAMPLED.copy()
        else:
            return GPR_LMR_SYMBOL.copy()
    elif mode == "pr4":
        return PR_IV.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")
