import numpy as np
from typing import Tuple


def gen_gpr_target(
    random_data: np.ndarray,
    sampled_data: np.ndarray,
    gpr_template: np.ndarray,
    fir_len: int = 21,
    constraint: str = 'centre',
    method: str = 'lagrange',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jointly optimize FIR equalizer taps and GPR target coefficients via MMSE.

    Minimizes MSE = E[(w^T y_k - g^T a_k)^2] subject to a linear constraint
    on g (one coefficient fixed to 1 to avoid trivial solution g=0).

    Based on: Kovintavewat et al., "Generalized partial-response targets for
    perpendicular recording with jitter noise," IEEE Trans. Magnetics, 2002.

    Args:
        random_data: Transmitted bit sequence a_k in {+1, -1}.
        sampled_data: Readback signal after LPF (at symbol rate or T/2 rate).
        gpr_template: Template target shape (e.g. [1, 1, 0, -1, -1] for PR-IV-like).
                      Only the length matters; actual coefficients are optimized.
        fir_len: Length of FIR equalizer (must be odd).
        constraint: Which GPR coefficient to fix to 1.
            '1'      -> first coefficient
            'centre' -> center coefficient
            '2'      -> second coefficient
        method: Optimization method.
            'lagrange' -> Lagrange multiplier (recommended)
            'eigen'    -> Minimum eigenvalue decomposition

    Returns:
        (fir_coeff, gpr_coeff): Optimized FIR taps and GPR target coefficients.
    """
    gpr_len = len(gpr_template)
    data_len = len(random_data)
    K = (fir_len - 1) // 2   # FIR half-length
    G = (gpr_len - 1) // 2   # GPR half-length

    # ---- Constraint vector I ----
    I_matrix = np.zeros((gpr_len, 1))
    if constraint == '1':
        I_matrix[0, 0] = 1
    elif constraint == 'centre':
        I_matrix[gpr_len // 2, 0] = 1
    elif constraint == '2':
        I_matrix[1, 0] = 1
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    # ---- R matrix: autocorrelation of sampled data ----
    autocorr_sampled = np.correlate(sampled_data, sampled_data, mode='full')
    R_matrix = np.zeros((fir_len, fir_len))
    for i in range(-K, K + 1):
        idx = data_len + np.arange(-K, K + 1) - i - 1
        R_matrix[i + K, :] = autocorr_sampled[idx]

    # ---- A matrix: autocorrelation of input data ----
    autocorr_data = np.correlate(random_data, random_data, mode='full')
    A_matrix = np.zeros((gpr_len, gpr_len))
    for i in range(-G, G + 1):
        idx = data_len + np.arange(-G, G + 1) - i - 1
        A_matrix[i + G, :] = autocorr_data[idx]

    # ---- T matrix: cross-correlation between sampled data and input data ----
    corr_ran_sample = np.correlate(sampled_data, random_data, mode='full')
    T_matrix = np.zeros((fir_len, gpr_len))
    for i in range(-K, K + 1):
        idx = data_len + np.arange(-G, G + 1) - i - 1
        T_matrix[i + K, :] = corr_ran_sample[idx]

    # ---- Regularize R ----
    R_matrix += np.eye(fir_len) * 1e-10

    # ---- Schur complement ----
    inv_R = np.linalg.inv(R_matrix)
    M = A_matrix - T_matrix.T @ inv_R @ T_matrix

    # ---- Solve for GPR coefficients ----
    if method == 'lagrange':
        inv_M = np.linalg.inv(M)
        denom = I_matrix.T @ inv_M @ I_matrix
        lagrange = 1.0 / denom[0, 0]
        gpr_coeff = (lagrange * inv_M @ I_matrix).flatten()
    elif method == 'eigen':
        evals, evecs = np.linalg.eig(M)
        idx = np.argmin(np.abs(evals))
        gpr_coeff = evecs[:, idx]
        # Normalize to satisfy constraint
        norm = I_matrix.T @ gpr_coeff.reshape(-1, 1)
        gpr_coeff = gpr_coeff / norm[0, 0]
    else:
        raise ValueError(f"Unknown method: {method}")

    # ---- FIR taps from GPR target ----
    fir_coeff = (inv_R @ T_matrix @ gpr_coeff.reshape(-1, 1)).flatten()

    return fir_coeff, gpr_coeff
