import numpy as np
from src.channel.channel_model import synthesize_readback_signal
from src.frontend.filters import create_lowpass_filter
from src.opsps_va.viterbi import OPSPVA

T=1.0; FS=100; PW50=2.5; MODE='perpendicular'; SNR_DB=25.0
SIGMA_J=0.03; SIGMA_W=0.005; ALPHA=0.0107; BETA=0.000309
LENGTH=4096; PREAMBLE_LEN=100; SEED=42
TAPS=[-0.04014,0.090561,-0.109755,0.094373,-0.183844,0.152992,0.35265,-0.755096,-0.657579,0.955242,0.45171,0.46566,0.799206,-0.638464,-0.757566,0.373477,0.07353,0.137979,-0.351151,0.216157,-0.060725]

for foff in [0.004,0.0]:
    _, r_raw, a_full, _, _ = synthesize_readback_signal(
        length=LENGTH, T=T, pw50=PW50, mode=MODE,
        sigma_j=SIGMA_J, sigma_w=SIGMA_W, freq_offset=foff,
        snr_db=SNR_DB, fs=FS, seed=SEED,
        preamble_length=PREAMBLE_LEN, preamble_pattern='4T')
    r_filtered=create_lowpass_filter(T=T,N=2,fs=FS).filter(r_raw)
    dec=OPSPVA(taps=TAPS,T=T,alpha=ALPHA,beta=BETA,samples_per_symbol=FS,mu_fse=1e-4)
    _, tau_hat, _ = dec.decode(r_filtered, ground_truth=None)
    idx=np.arange(len(tau_hat),dtype=float)
    slope,_=np.polyfit(idx,tau_hat,1)
    tau_wrapped=((tau_hat+0.5)%1.0)-0.5
    print(f'freq_offset={foff:.4f}: tau_final={tau_hat[-1]:.6f}, tau_slope={slope:.6e}, wrapped_std_last200={np.std(tau_wrapped[-200:]):.6f}, wrapped_mean_last200={np.mean(tau_wrapped[-200:]):.6f}')
