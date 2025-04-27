import numpy as np
import pydrake.symbolic as sym
from typing import Tuple


def system_dynamics() -> Tuple[np.ndarray, np.ndarray]:
    """
    This function defines the system dyanmics of a single integrator.
    The system dynamics is:
    ̇x₀ = u₀ ; ̇x₁ = u₁
    we can see that the state variables are not present in dynamics
    for the single integrator.
    """
    f = np.array([sym.Polynomial(), sym.Polynomial()])
    g = np.array(
        [
            [sym.Polynomial(1), sym.Polynomial()],
            [sym.Polynomial(), sym.Polynomial(1)],
        ]
    )
    return f, g


def system_dynamics_forward() -> Tuple[np.ndarray, np.ndarray]:
    """
    We can use this function when doing real time simulations
    """
    f = np.array([0, 0])
    g = np.array([[1, 0], [0, 1]])
    return f, g


def control_limits() -> Tuple[np.ndarray, np.ndarray]:
    """
    This function defines the control limits of the system
    u₁ ∈ [-1, 1],
    u₂ ∈ [-1, 1]
    """
    u1_min = -20
    u1_max = 20
    u2_min = -20
    u2_max = 20
    Au = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    bu = np.array([u1_max, -u1_min, u2_max, -u2_min])
    return Au, bu
