import numpy as np
import pydrake.symbolic as sym
from typing import Tuple


def system_dynamics(x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function defines the system dyanmics of a unicycle model 
    with only agular velocity control and a constant v. 
    The system dynamics is:
    ̇x₀ = cos(x₂) * v
    ̇x₁ = sin(x₂) * v
    ̇x₂ = u₀
    to polynomuialize the dynamics, we define x0 as the horizontal position,
    x1 as the vertical position, and x2 as the sin(θ), where θ is the heading angle,
    and we define x3 as the cos(θ)-1, so that cos(θ) = x3 + 1.
    The system dynamics is then:
    ̇x₀ = (x3 + 1) * v
    ̇x₁ = x2 * v
    ̇x₂ = u₀(x3 + 1)
    ̇x₃ = -u₀ * x2
    """
    assert x.shape == (4,)
    v = 0.5  # constant forward velocity
    f = np.array([
        sym.Polynomial(v*x[3] + v),
        sym.Polynomial(v*x[2]),
        sym.Polynomial(0),
        sym.Polynomial(0)
    ])
    g = np.array([
        [sym.Polynomial(0)],
        [sym.Polynomial(0)],
        [sym.Polynomial(x[3] + 1)],
        [sym.Polynomial(-x[2])]
    ])
    return f, g

def equation_constraint(x:np.ndarray) -> sym.Polynomial:
    """
    This function defines the equation constraint for the unicycle model.
    The equation constraint is:
    x2^2 + (x3 + 1)^2 = 1
    """
    assert x.shape == (4,)
    return sym.Polynomial(x[2]**2 + (x[3] + 1)**2 - 1)

def control_input_limits() -> Tuple[np.ndarray, np.ndarray]:
    """
    This function defines the control limits of the system
    u₀ ∈ [-1, 1]
    """
    u0_min = -1
    u0_max = 1
    Au = np.array([
        [sym.Polynomial(1)], 
        [sym.Polynomial(-1)]
    ])
    bu = np.array([
        sym.Polynomial(u0_max), 
        sym.Polynomial(-u0_min)
    ])
    return Au, bu
