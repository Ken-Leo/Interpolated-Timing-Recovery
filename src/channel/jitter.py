import numpy as np

def generate_media_jitter(b, sigma_j, T):
    """
    Media jitter noise Delta t_k.
    Distributed as N(0, |b_k/2| * sigma_j^2), truncated to T/2.

    Args:
        b: Transition sequence (array)
        sigma_j: Media jitter standard deviation
        T: Bit period
    Returns:
        delta_t: Random shifts for each transition
    """
    # Calculate variance for each transition: |b_k/2| * sigma_j^2
    variances = np.abs(b / 2.0) * (sigma_j**2)
    stds = np.sqrt(variances)

    # Generate Gaussian noise
    delta_t = np.random.normal(0, stds)

    # Truncate to T/2
    delta_t = np.clip(delta_t, -T/2, T/2)

    return delta_t

def generate_clock_jitter(length, sigma_w):
    """
    Clock jitter noise tau_k.
    Modeled as a random walk: tau_{k+1} = tau_k + N(0, sigma_w^2).

    Args:
        length: Number of symbols
        sigma_w: Clock jitter step standard deviation
    Returns:
        tau: Cumulative clock jitter sequence
    """
    steps = np.random.normal(0, sigma_w, length)
    tau = np.cumsum(steps)
    return tau
