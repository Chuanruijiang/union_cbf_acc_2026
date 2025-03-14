"""
This file defines the dynamics of the turtle bot.
The origianl dynamics is the following:
Define px as the position of the turtle bot in the x direction.
Define py as the position of the turtle bot in the y direction.
Define θ as the orientation of the turtle bot.
We have vx = v * cos(θ) and vy = v * sin(θ).
where v is the true velocity of the turtle bot.
The control command are the velocity and the angular velocity.

Define x0 = px, x1 = py, x2 = θ, u0 = v, u1 = ω.
The dynamics are given by:
̇x0 = u0 * cos(x2),
̇x1 = u0 * sin(x2),
̇x2 = u1.

This is a nonlinear dynamics with trigonometric functions. in order
to polynomialize it. We define the following change of variables:
x0 = px, x1 = py, x2 = sin(θ), x3 = cos(θ)-1, u0 = v, u1 = ω.
The new dyanmics would be:
̇x0 = u0 * (x3 + 1),
̇x1 = u0 * x2,
̇x2 = (x3 + 1)*u1,
̇x3 = -x2 * u1.
with a state equation constraint: x3^2 + (x4 + 1)^2 = 1.
"""
import numpy as np
import pydrake.symbolic as sym
from typing import Tuple

def system_dynamics(x:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """
    This function defines the system dyanmics of the turtle bot.
    """
    f = np.array([
        sym.Polynomial(),
        sym.Polynomial(),
        sym.Polynomial(),
        sym.Polynomial()
        ])
    g = np.array([
        [sym.Polynomial(x[3] + 1), sym.Polynomial()],
        [sym.Polynomial(x[2]), sym.Polynomial()],
        [sym.Polynomial(), sym.Polynomial(x[3] + 1)],
        [sym.Polynomial(), sym.Polynomial(-x[2])]
        ])
    return f, g


def system_dynamics_forward(x:np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    This function defines the system dyanmics of the turtle bot.
    """
    f = np.array([0, 0, 0, 0])
    g = np.array([
        [(x[3] + 1), 0         ],
        [x[2]      , 0         ],
        [0         , (x[3] + 1)],
        [0         , -x[2]     ]
    ])
    return f + g @ u


def state_equation_constraint(x:np.ndarray)->np.ndarray:
    """
    This function defines the state equation constraint of the turtle bot.
    """
    return np.array([x[2]**2 + x[3]**2 + 2*x[3]])