import sys
import numpy as np
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA
from src.utils.gpr_coefficients import get_gpr_target
from src.utils.metrics import compute_ber, compute_fse_metrics

# Parameters from compare_fse_multiseed.py
T = 1.0
FS = 100
PW50 = 2.5
MODE = "perpendicular"
SNR_DB = 25.0
SIGMA_J = 0.03
SIGMA_W = 0.005
FREQ_OFFSET = 0.004
ALPHA = 0.0107
BETA = 0.000309
LENGTH = 16384
PREAMBLE_LEN = 100
FIXED_SEED = 42
TAPS = [
    -0.04014, 0.090561, -0.109755, 0.094373, -0.183844, 0.152992, 0.35265, -0.755096, 
    -0.657579, 0.955242, 0.45171, 0.46566, 0.799206, -0.638464, -0.757566, 0.373477, 
    0.07353, 0.137979, -0.351151, 0.216157, -0.060725,
]

MU_VALUES = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]

def run_simulation(mu_fse):
    np.random.seed(FIXED_SEED)
    a_full = np.random.choice([-1, 1], size=LENGTH)
    
    r_analog, _ = synthesize_readback_signal(
        a_full, T, FS, PW50, SNR_DB, SIGMA_J, SIGMA_W, FREQ_OFFSET, MODE, ALPHA, BETA
    )
    
    lp_filter = create_lowpass_filter(2 * FS, FS / 2, numtaps=64)
    r_filtered = np.convolve(r_analog, lp_filter, mode="same")
    
    gpr_target = get_gpr_target(PW50, T)
    
    # Initialize OPSPVA
    # Assuming standard OPSPVA parameters from scripts
    va = OPSPVA(
        target=gpr_target,
        mu_tau=0.01,
        mu_phi=0.001,
        mu_fse=mu_fse,
        fse_taps=np.array(TAPS),
    )
    
    data_start = PREAMBLE_LEN * FS
    res = va.run(r_filtered[data_start:], a_tx=a_full[PREAMBLE_LEN:])
    
    a_hat = res["a_hat"]
    tau_hat = res["tau_hat"]
    mse_history = res["mse_history"]
    
    ber, _, _ = compute_ber(a_full[PREAMBLE_LEN:], a_hat)
    fse_mets = compute_fse_metrics(mse_history)
    
    # Detrended tail metrics
    idx = np.arange(len(tau_hat), dtype=float)
    slope, intercept = np.polyfit(idx, tau_hat, deg=1)
    tau_detrended = tau_hat - (slope * idx + intercept)
    
    tail_len = 2000
    tail = tau_detrended[-tail_len:]
    t_idx = np.arange(len(tail), dtype=float)
    tail_slope, _ = np.polyfit(t_idx, tail, deg=1)
    tail_mean = np.mean(tail)
    tail_std = np.std(tail)
    
    converged = abs(tail_mean) < 0.05 and tail_std < 0.10 and abs(tail_slope) < 1e-4
    
    return {
        "mu": mu_fse,
        "ber": ber,
        "mse": fse_mets["mse_final"],
        "tau_slope": slope,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_slope": tail_slope,
        "converged": converged
    }

print(f"{'mu':<7} | {'BER':<6} | {'FinalMSE':<8} | {'TauSlope':<9} | {'TailMean':<8} | {'TailStd':<7} | {'TailSlp':<8} | {'Conv'}")
print("-" * 85)

for mu in MU_VALUES:
    row = run_simulation(mu)
    print(f"{row['mu']:<7g} | {row['ber']:.4f} | {row['mse']:.6f} | {row['tau_slope']:.2e} | {row['tail_mean']:.4f} | {row['tail_std']:.4f} | {row['tail_slope']:.2e} | {row['converged']}")

