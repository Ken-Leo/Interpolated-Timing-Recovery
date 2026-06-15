import numpy as np

def generate_binary_sequence(length, seed=None):
    """
    Generate a random binary sequence a_k in {+1, -1}.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([1, -1], size=length)

def generate_transition_sequence(a):
    """
    Convert binary sequence a_k to transition sequence b_k.
    b_k = a_k * (1 - D) = a_k - a_{k-1}
    b_k in {-2, 0, 2}
    """
    # Shift a to get a_{k-1}
    a_prev = np.roll(a, 1)
    # To strictly follow (1-D) where a_{-1} is assumed 0 or a fixed value,
    # we set the first element of a_prev to 0.
    a_prev[0] = 0

    b = a - a_prev
    return b
