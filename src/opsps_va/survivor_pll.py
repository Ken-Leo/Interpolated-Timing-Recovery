from dataclasses import dataclass


@dataclass
class PLLState:
    """
    State of a second-order Phase-Locked Loop (PLL)
    associated with a specific survivor path.
    """

    tau: float = 0.0  # Sampling phase offset
    theta: float = 0.0  # Frequency error estimate


class SurvivorPLL:
    """
    Implementation of the second-order PLL used in OPSP-VA
    to track the sampling phase and frequency.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        theta_leak: float = 0.0,
        theta_clip: float | None = None,
    ):
        """
        Initialize the PLL with gain parameters.

        Args:
            alpha: Phase loop gain (controls convergence rate).
            beta: Frequency loop gain.
            theta_leak: Leakage factor in [0, 1). When >0, applies
                new_theta = (1 - theta_leak) * old_theta + beta * epsilon.
            theta_clip: Optional symmetric limit for theta magnitude.
        """
        self.alpha = alpha
        self.beta = beta
        if not (0.0 <= theta_leak < 1.0):
            raise ValueError("theta_leak must satisfy 0 <= theta_leak < 1")
        if theta_clip is not None and theta_clip <= 0.0:
            raise ValueError("theta_clip must be positive or None")
        self.theta_leak = theta_leak
        self.theta_clip = theta_clip

    @staticmethod
    def _wrap_tau(tau: float) -> float:
        """Pass-through: cubic interpolation handles any tau value.
        Wrapping was causing PLL instability with DC-biased TED."""
        return tau

    def update(self, current_state: PLLState, epsilon: float) -> PLLState:
        """
        Update the PLL state based on the timing error epsilon.

        Paper equations (4) and (5):
            theta_k = theta_{k-1} + beta * epsilon_k
            tau_{k+1} = tau_k + alpha * epsilon_k + theta_k

        The TED S-curve has negative slope (epsilon > 0 when tau < 0),
        so positive feedback in the PLL creates overall negative feedback.
        """
        new_theta = (1.0 - self.theta_leak) * current_state.theta + self.beta * epsilon
        if self.theta_clip is not None:
            new_theta = max(-self.theta_clip, min(self.theta_clip, new_theta))
        new_tau = current_state.tau + self.alpha * epsilon + new_theta
        new_tau = self._wrap_tau(new_tau)

        return PLLState(tau=new_tau, theta=new_theta)
