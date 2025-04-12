"""
This file defines the dynamics of the 2D quadrotor reach-avoid example.

Here are the variables for the 2D quadrotor:
x0 = Px meaning the x position of the quadrotor
x1 = Py meaning the y position of the quadrotor
x2 = Vx meaning the x velocity of the quadrotor
x3 = Vy meaning the y velocity of the quadrotor
x4 = θ  meaning the rolling angle of the quadrotor
x5 = ω meaning the rolling rate of the quadrotor
The dynamics are given by:
Ṗx  = Vx,
Ṗy  = Vy,
V̇x  = 1/m * sinθ * (u_0 + u_1)
V̇y  = 1/m * cosθ * (u_0 + u_1) - g
̇θ   = ω
̇ω   = Lr/In * (u_0 - u_1)

since the dynamics has trigonometric functions, we need to polynomialize it.
define the following change of variables:
x0 = Px, x1 = Py, x2 = Vx, x3 = Vy, x4 = sin(θ), x5 = ω, x6 = cos(θ) - 1
The new dyanmics would be:
̇x0 = x2,
̇x1 = x3,
̇x2 = 1/m * (x4 * (u_0 + u_1)),
̇x3 = 1/m * (x6+1) * (u_0 + u_1) - g,
̇x4 = x5(x6 + 1)
̇x5 = Lr/In * (u_0 - u_1)
̇x6 = -x4 * x5
with a state equation constraint: x4^2 + (x6 + 1)^2 = 1.
"""

import numpy as np
import pydrake.symbolic as sym
from typing import Tuple

# define the parameters
m = 0.486
g = 9.81
Lr = 0.25
In = 0.004


def system_dynamics(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function defines the system dyanmics of the quadrotor.
    """
    f = np.array(
        [
            sym.Polynomial(x[2]),
            sym.Polynomial(x[3]),
            sym.Polynomial(),
            sym.Polynomial(-g),
            sym.Polynomial(x[5] * (x[6] + 1)),
            sym.Polynomial(),
            sym.Polynomial(-x[4] * x[5]),
        ]
    )
    G = np.array(
        [
            [sym.Polynomial(), sym.Polynomial()],
            [sym.Polynomial(), sym.Polynomial()],
            [sym.Polynomial(x[4] / m), sym.Polynomial(x[4] / m)],
            [sym.Polynomial((x[6] + 1) / m), sym.Polynomial((x[6] + 1)) / m],
            [sym.Polynomial(), sym.Polynomial()],
            [sym.Polynomial(Lr / In), sym.Polynomial(-Lr / In)],
            [sym.Polynomial(), sym.Polynomial()],
        ]
    )
    return f, G


def system_dynamics_forward(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    This function defines the system dyanmics of the quadrotor.
    """
    assert x.shape[0] == 7, "x should be of shape (7,)"
    assert u.shape[0] == 2, "u should be of shape (2,)"
    f = np.array([
        x[2],
        x[3],
        0,
        -g,
        x[5] * (x[6] + 1),
        0,
        -x[4] * x[5]
        ])
    G = np.array(
        [
            [0, 0],
            [0, 0],
            [x[4] / m, x[4] / m],
            [(x[6] + 1) / m, (x[6] + 1) / m],
            [0, 0],
            [Lr / In, -Lr / In],
            [0, 0],
        ]
    )
    return f + G @ u


def state_equation_constraint(x: np.ndarray) -> np.ndarray:
    """
    This function defines the state equation constraint of the turtle bot.
    """
    return np.array([sym.Polynomial(x[4] ** 2 + (x[6] + 1) ** 2 - 1)])


def original_to_extended_state_space(x: np.ndarray) -> np.ndarray:
    """
    This function transforms the original state space to the extended state space.
    """
    assert x.shape[0] == 6, "x should be of shape (6,)"
    z = np.empty(x.shape[0] + 1)
    z[0] = x[0]
    z[1] = x[1]
    z[2] = x[2]
    z[3] = x[3]
    z[4] = np.sin(x[4])
    z[5] = x[5]
    z[6] = np.cos(x[4]) - 1
    return z
