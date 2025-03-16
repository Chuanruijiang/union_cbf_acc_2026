"""
This file defines the dynamics of the non-linear toy
example.

The dynamics are given by:
̇θ  = u0,
̇γ  = -sin(θ) - u1,

This is a nonlinear dynamics with trigonometric functions. in order
to polynomialize it. We define the following change of variables:
x0 = γ , x1 = sin(θ), x2 = cos(θ)-1
The new dyanmics would be:
̇x0 = -x1 - u1,
x1 = (x2 + 1) * u0,
̇x2 = -x1 * u0
with a state equation constraint: x1^2 + (x2 + 1)^2 = 1.
"""
import numpy as np
import pydrake.symbolic as sym
from typing import Tuple

def system_dynamics(x:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """
    This function defines the system dyanmics of the turtle bot.
    """
    f = np.array([
        sym.Polynomial(-x[1]),
        sym.Polynomial(),
        sym.Polynomial()
        ])
    g = np.array([
        [sym.Polynomial(),       sym.Polynomial(-1)],
        [sym.Polynomial(x[2]+1), sym.Polynomial()  ],
        [sym.Polynomial(-x[1]),  sym.Polynomial()  ]
        ])
    return f, g


def system_dynamics_forward(x:np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    This function defines the system dyanmics of the turtle bot.
    """
    f = np.array([-x[1], 0, 0])
    g = np.array([
        [0,         -1         ],
        [x[2]+1    , 0         ],
        [-x[1]     , 0         ]
    ])
    return f + g @ u


def state_equation_constraint(x:np.ndarray)->np.ndarray:
    """
    This function defines the state equation constraint of the turtle bot.
    """
    return np.array([x[1]**2 + x[2]**2 + 2*x[2]])