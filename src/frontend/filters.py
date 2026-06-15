import numpy as np
from scipy import signal
from typing import Tuple

class ButterworthFilter:
    """
    Implementation of a 7th-order Butterworth low-pass filter
    as described in the OPSP-VA system model.
    """
    def __init__(self, cutoff_freq: float, fs: float):
        """
        Initialize the Butterworth filter.

        Args:
            cutoff_freq: Cut-off frequency in Hz.
            fs: Sampling frequency in Hz.
        """
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = 7

        # Design the filter
        self.b, self.a = signal.butter(self.order, cutoff_freq, fs=fs, btype='low', analog=False)

        # Initialize filter state for continuous filtering
        self.zi = signal.lfilter_zi(self.b, self.a)

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Filter the input data.

        Args:
            data: Input signal array.
        Returns:
            Filtered signal array.
        """
        # Use lfilter with zi to maintain state between calls if needed
        filtered, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
        return filtered

def create_lowpass_filter(T: float, N: int = 2, fs: float = 100.0) -> ButterworthFilter:
    """
    Helper function to create a filter based on the system parameters.
    Cutoff frequency is typically N / (2 * T).
    """
    cutoff_freq = N / (2.0 * T)
    return ButterworthFilter(cutoff_freq, fs)
