import numpy as np

class FarrowInterpolator:
    """
    Farrow-structure interpolator based on the implementation in E8_11_gardner.m.
    Provides quadratic interpolation of a signal.
    """
    def __init__(self):
        pass

    def interpolate(self, signal: np.ndarray, pos_float: float) -> float:
        """
        Interpolate signal at pos_float.
        
        Args:
            signal: The sampled signal.
            pos_float: The floating point index.
            
        Returns:
            Interpolated value.
        """
        i = int(np.floor(pos_float))
        u = pos_float - i
        
        # We need samples i-1, i, i+1, i+2
        # Guard against boundaries
        def get_val(idx):
            if 0 <= idx < len(signal):
                return signal[idx]
            return 0.0
            
        a_prev = get_val(i - 1)
        a_curr = get_val(i)
        a_next = get_val(i + 1)
        a_next2 = get_val(i + 2)
        
        # Farrow coefficients from E8_11_gardner.m
        # FI1 = 0.5*aI(i+2) - 0.5*aI(i+1) - 0.5*aI(i) + 0.5*aI(i-1)
        # FI2 = 1.5*aI(i+1) - 0.5*aI(i+2) - 0.5*aI(i) - 0.5*aI(i-1)
        # FI3 = aI(i)
        
        fi1 = 0.5 * a_next2 - 0.5 * a_next - 0.5 * a_curr + 0.5 * a_prev
        fi2 = 1.5 * a_next - 0.5 * a_next2 - 0.5 * a_curr - 0.5 * a_prev
        fi3 = a_curr
        
        # y = (FI1 * u + FI2) * u + FI3
        return (fi1 * u + fi2) * u + fi3
