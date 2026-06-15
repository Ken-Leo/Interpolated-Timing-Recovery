import numpy as np
from scipy.special import erf

def longitudinal_pulse(t, pw50):
    """
    Longitudinal recording transition response:
    g(t) = 1 / (1 + (2t/PW50)^2)
    """
    return 1.0 / (1.0 + (2.0 * t / pw50)**2)

def perpendicular_pulse(t, pw50):
    """
    Perpendicular recording transition response:
    g(t) = erf(2 * t * sqrt(ln 2) / PW50)
    """
    return erf(2.0 * t * np.sqrt(np.log(2)) / pw50)

def get_pulse_shape(mode='longitudinal', pw50=1.0):
    """
    Return the pulse shape function based on the mode.
    """
    if mode == 'longitudinal':
        return longitudinal_pulse
    elif mode == 'perpendicular':
        return perpendicular_pulse
    else:
        raise ValueError("Invalid mode. Choose 'longitudinal' or 'perpendicular'.")
