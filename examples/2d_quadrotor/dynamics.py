"""
This file defines the dynamics of the 2D quadrotor reach-avoid example.

Here are the variables for the 2D quadrotor:
x0 = Px meaning the x position of the quadrotor
x1 = Pz meaning the y position of the quadrotor
x2 = θ  meaning the rolling angle of the quadrotor
x3 = Vx meaning the x velocity of the quadrotor
x4 = Vz meaning the y velocity of the quadrotor
x5 = ω meaning the rolling rate of the quadrotor
The dynamics are given by:
Ṗx  = Vx,
Ṗy  = Vz,
̇θ   = ω,
V̇x  = -1/m * sinθ * (u_0 + u_1)
V̇y  = 1/m * cosθ * (u_0 + u_1) - g
̇ω   = Lr/In * (u_0 - u_1)

since the dynamics has trigonometric functions, we need to polynomialize it.
define the following change of variables:
x0 = Px, x1 = Pz, x2 = sin(θ), x3 = cos(θ)-1, x4 = Vx, x5 = Vz, x6 = ω
The new dyanmics would be:
̇x0 = x4,
̇x1 = x5,
̇x2 = (x3 + 1) * x6,
̇x3 = -x2 * x6,
̇x4 = -1/m * (x2 * (u_0 + u_1)),
̇x5 = 1/m * (x3 + 1) * (u_0 + u_1) - g,
̇x6 = Lr/In * (u_0 - u_1)
with a state equation constraint: x4^2 + (x6 + 1)^2 = 1.

Another easier way to handle the sin and cos terms in the dynamics is to use
the taylor exampsion. If the rolling angle of the quadrotor is within [-pi/9, pi/9],
then sin can be just replaced by sinx = x - x^3/(3*2) and cos can be replaced by
cosx = 1 - x^2/2. This will make the dynamics polynomial.
The new dyanmics would be:
̇x0 = x3,
̇x1 = x4,
̇x2 = x5,
̇x3 = -1/m * (sin_theta * (u_0 + u_1)),
̇x4 = 1/m * (cos_theta * (u_0 + u_1)) - g,
̇x5 = Lr/In * (u_0 - u_1)
sin_theta = x2 - x2^3/(3*2)
cos_theta = 1 - x3/2
"""

from typing import Tuple, Union

import numpy as np
import jax.numpy as jnp
import pydrake.symbolic as sym


class Quadrotor2dPlant:
    """
    The state is x = (pos_x, pos_z, theta, pos_x_dot, pos_z_dot, theta_dot)
    """

    m: float
    l: float
    I: float
    g: float

    def __init__(self):
        self.m = 0.486
        self.l = 0.25
        self.I = 0.00383
        self.g = 9.81

    def dynamics(
        self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        if isinstance(x, np.ndarray):
            numpy_type = np
        elif isinstance(x, jnp.ndarray):
            numpy_type = jnp
        s_theta = numpy_type.sin(x[2])
        c_theta = numpy_type.cos(x[2])
        return numpy_type.stack(
            (
                x[3],
                x[4],
                x[5],
                -s_theta / self.m * (u[0] + u[1]),
                c_theta / self.m * (u[0] + u[1]) - self.g,
                self.l / self.I * (u[0] - u[1]),
            )
        )

    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Write the control-affine dynamics xdot = f(x) + g(x) * u. Notice that
        f and g are not polynomials of x.
        """
        f = np.empty((6,), dtype=object)
        g = np.empty((6, 2), dtype=object)
        f[0] = 1 * x[3]
        f[1] = 1 * x[4]
        f[2] = 1 * x[5]
        f[3] = sym.Expression(0)
        f[4] = sym.Expression(-self.g)
        f[5] = sym.Expression(0)
        for i in range(3):
            for j in range(2):
                g[i, j] = sym.Expression()
        g[3, 0] = -np.sin(x[2]) / self.m
        g[3, 1] = -np.sin(x[2]) / self.m
        g[4, 0] = np.cos(x[2]) / self.m
        g[4, 1] = np.cos(x[2]) / self.m
        g[5, 0] = sym.Expression(self.l / self.I)
        g[5, 1] = sym.Expression(-self.l / self.I)
        return f, g

    def taylor_affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use taylor expansion around theta = 0 on the cos and sin, and write the
        control affine dynamics as xdot = f(x) + g(x)*u
        """
        f = np.empty((6,), dtype=object)
        g = np.empty((6, 2), dtype=object)
        f[0] = sym.Polynomial(x[3])
        f[1] = sym.Polynomial(x[4])
        f[2] = sym.Polynomial(x[5])
        f[3] = sym.Polynomial()
        f[4] = sym.Polynomial(-self.g)
        f[5] = sym.Polynomial()
        for i in range(3):
            for j in range(2):
                g[i, j] = sym.Polynomial()
        g[3, 0] = sym.Polynomial(-(x[2] - 1 / 6.0 * x[2] ** 3) / self.m)
        g[3, 1] = sym.Polynomial(-(x[2] - 1 / 6.0 * x[2] ** 3) / self.m)
        g[4, 0] = sym.Polynomial((1 - x[2] ** 2 / 2) / self.m)
        g[4, 1] = sym.Polynomial((1 - x[2] ** 2 / 2) / self.m)
        g[5, 0] = sym.Polynomial(self.l / self.I)
        g[5, 1] = sym.Polynomial(-self.l / self.I)
        return f, g

    def linearize_dynamics(
        self, x_des: np.ndarray, u_des: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        s_theta = np.sin(x_des[2])
        c_theta = np.cos(x_des[2])
        A[:3, 3:] = np.eye(3)
        A[3, 2] = -c_theta / self.m * (u_des[0] + u_des[1])
        A[4, 2] = -s_theta / self.m * (u_des[0] + u_des[1])
        B[3, 0] = -s_theta / self.m
        B[3, 1] = -s_theta / self.m
        B[4, 0] = c_theta / self.m
        B[4, 1] = c_theta / self.m
        B[5, 0] = self.l / self.I
        B[5, 1] = -self.l / self.I
        return A, B


class Quadrotor2dTrigPlant:
    """
    The state is
    x = (pos_x, pos_z, sin(theta), cos(theta)-1, pos_x_dot, pos_z_dot, theta_dot))
    Note that the equilibrium is at x=0.
    The dynamics is a polynomial function of the state/action.
    """

    m: float
    l: float
    I: float
    g: float

    def __init__(self):
        self.m = 0.486
        self.l = 0.25
        self.I = 0.00383
        self.g = 9.81

    def dynamics(
        self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        if isinstance(x, np.ndarray):
            numpy_type = np
        elif isinstance(x, jnp.ndarray):
            numpy_type = jnp

        c_theta = x[3] + 1
        xdot = numpy_type.stack(
            (
                x[4],
                x[5],
                c_theta * x[6],
                -x[2] * x[6],
                -x[2] / self.m * (u[0] + u[1]),
                c_theta / self.m * (u[0] + u[1]) - self.g,
                self.l / self.I * (u[0] - u[1]),
            )
        )
        return xdot

    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The control-affine dynamics xdot = f(x) + g(x) * u.
        Returns:
          f, g: The affine dynamics term where each entry is a polynomial of x.
        """
        assert x.shape == (7,)
        f = np.empty((7,), dtype=object)
        g = np.empty((7, 2), dtype=object)
        f[0] = sym.Polynomial(x[4])
        f[1] = sym.Polynomial(x[5])
        f[2] = sym.Polynomial((x[3] + 1) * x[6])
        f[3] = sym.Polynomial(-x[2] * x[6])
        f[4] = sym.Polynomial()
        f[5] = sym.Polynomial(-self.g)
        f[6] = sym.Polynomial()
        for i in range(4):
            for j in range(2):
                g[i, j] = sym.Polynomial()
        g[4, 0] = sym.Polynomial(-x[2] / self.m)
        g[4, 1] = sym.Polynomial(-x[2] / self.m)
        g[5, 0] = sym.Polynomial((x[3] + 1) / self.m)
        g[5, 1] = sym.Polynomial((x[3] + 1) / self.m)
        g[6, 0] = sym.Polynomial(self.l / self.I)
        g[6, 1] = sym.Polynomial(-self.l / self.I)
        return (f, g)

    def linearize_dynamics(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics xdot =f(x, u)
        A = ∂f/∂x, B = ∂f/∂u
        """
        c_theta = x[3] + 1
        A = np.zeros((7, 7))
        A[0, 4] = 1
        A[1, 5] = 1
        A[2, 3] = x[6]
        A[2, 6] = c_theta
        A[3, 2] = -x[6]
        A[3, 6] = -x[2]
        A[4, 2] = -1 / self.m * (u[0] + u[1])
        A[5, 3] = 1 / self.m * (u[0] + u[1])

        B = np.zeros((7, 2))
        B[4, 0] = -x[2] / self.m
        B[4, 1] = -x[2] / self.m
        B[5, 0] = c_theta / self.m
        B[5, 1] = c_theta / self.m
        B[6, 0] = self.l / self.I
        B[6, 1] = -self.l / self.I
        return A, B

    def equality_constraint(self, x: np.ndarray) -> np.ndarray:
        """
        The left-hand side of the equality constraint x[2]^2 + (x[3]+1)^2 - 1 = 0
        """
        return np.array([sym.Polynomial(x[2] ** 2 + x[3] ** 2 + 2 * x[3])])

    def to_trig_state(self, x_non_trig: np.ndarray) -> np.ndarray:
        """
        From the non-trigonometric state (pos, theta, pos_dot, theta_dot) to the
        trigonometric state (pos, sin(theta), cos(theta)-1, pos_dot, theta_dot)
        """
        if x_non_trig.shape == (6,):
            return np.stack(
                (
                    x_non_trig[0],
                    x_non_trig[1],
                    np.sin(x_non_trig[2]),
                    np.cos(x_non_trig[2]) - 1,
                    x_non_trig[3],
                    x_non_trig[4],
                    x_non_trig[5],
                )
            )
        elif len(x_non_trig.shape) == 2:
            return np.concatenate(
                (
                    x_non_trig[:, :2],
                    np.sin(x_non_trig[:, 2:3]),
                    np.cos(x_non_trig[:, 2:3]) - 1,
                    x_non_trig[:, 3:],
                ),
                axis=1,
            )
