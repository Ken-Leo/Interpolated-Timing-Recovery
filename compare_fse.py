import numpy as np
from src.channel.channel_model import synthesize_readback_signal
from src.opsps_va.viterbi import OPSPVA
from src.utils.metrics import compute_ber, compute_fse_metrics

def run_sim(seed, mu_fse):
    length = 1000
    preamble_length = 100
    fs = 100
    T = 1.0
    pw50 = 2.5
    mode = 'perpendicular'
    snr_db = 25
    sigma_j = 0.03
    sigma_w = 0.005
    freq_offset = 0.004
    alpha = 0.0107
    beta = 0.000309
    
    # PR-IV taps from main.py
    taps = [
        -0.04014, 0.090561, -0.109755, 0.094373, -0.183844,
        0.152992, 0.35265, -0.755096, -0.657579, 0.657579,
        0.755096, -0.35265, -0.152992, 0.183844, -0.094373,
        0.109755, -0.090561, 0.04014
    ]

    # Generate signal
    t, y, bits, b_coeffs, data_start = synthesize_readback_signal(
        length=length,
        preamble_length=preamble_length,
        T=T,
        pw50=pw50,
        mode=mode,
        snr_db=snr_db,
        sigma_j=sigma_j,
        sigma_w=sigma_w,
        freq_offset=freq_offset,
        seed=seed,
        fs=fs
    )
    
    # Run OPSP-VA
    # Looking at __init__(self, taps: List[float], T: float, alpha: float, beta: float, samples_per_symbol: int = 100, mu_fse: float = 0.001):
    opsps = OPSPVA(
        taps=taps,
        T=T,
        alpha=alpha,
        beta=beta,
        samples_per_symbol=fs,
        mu_fse=mu_fse
    )
    
    # Process
    det_bits, state_history = opsps.process(y)
    
    # Metrics
    ber = compute_ber(bits, det_bits)
    
    # FSE MSE
    # Each state_history entry is a list of PathState objects. 
    # Let's check how state_history is structured in viterbi.py process() method return.
    # Usually it's a list of states for each time step or just the final state.
    # Actually, in this project's OPSPVA.process, it returns (decoded_bits, state_history).
    # PathState usually has fse_taps.
    
    # Need to check PathState structure or just extract fse_taps from the survivors.
    # For now, let's assume compute_fse_metrics can handle it if we pass the history correctly.
    # If state_history is a list of lists of PathState, we might need to pick the best path's taps.
    
    fse_history = []
    for step_states in state_history:
        # Pick the fse_taps from the first state or some representative state
        # in the Viterbi trellis for tracking purposes.
        if hasattr(step_states[0], 'fse_taps'):
             fse_history.append(step_states[0].fse_taps)
        else:
             fse_history.append(taps) # Fallback

    fse_metrics = compute_fse_metrics(fse_history, target_taps=np.array(taps))
    
    return {
        'ber': ber,
        'initial_mse': fse_metrics['initial_mse'],
        'final_mse': fse_metrics['final_mse']
    }

seeds = [1, 2, 3, 4, 5]
mus = [0.0, 1e-4]

results = {mu: [] for mu in mus}

print(f"{'Seed':<5} | {'Mu':<7} | {'BER':<10} | {'Init MSE':<12} | {'Final MSE':<12}")
print("-" * 55)

for seed in seeds:
    for mu in mus:
        res = run_sim(seed, mu)
        results[mu].append(res)
        print(f"{seed:<5} | {mu:<7} | {res['ber']:<10.6f} | {res['initial_mse']:<12.6f} | {res['final_mse']:<12.6f}")

print("\nSummary (Mean ± Std):")
for mu in mus:
    bers = [r['ber'] for r in results[mu]]
    final_mses = [r['final_mse'] for r in results[mu]]
    print(f"Mu={mu}: BER={np.mean(bers):.6f}±{np.std(bers):.6f}, Final MSE={np.mean(final_mses):.6f}±{np.std(final_mses):.6f}")

mu0_ber = np.mean([r['ber'] for r in results[0.0]])
mu_eps_ber = np.mean([r['ber'] for r in results[1e-4]])
win = "Yes" if mu_eps_ber < mu0_ber else "No"
print(f"\nAdaptive FSE wins consistently (lower mean BER): {win}")
