"""
This file defines the dynamics of the numerical example. We modify the high
dementional differential equation from the following paper:
"FOSSIL: a software tool for the formal synthesis of lyapunov functions
and barrier certificates using neural networks" by Alessandro Abate, est al.
We add control input variables to the dynamics and provides system dynamics
with 2,3,4,5,6,7,8 dimensions to record the compuation time of verification
and synthesis.
dim2:
[̇x0] = [0,        1][x0] + [1, 0][u0]
[̇x1]   [-576, -2400][x1]   [0, 1][u1]
dim3:
[̇x0] = [   0,     1,     0][x0] + [1, 0, 0][u0]
[̇x1]   [   0,     0,     1][x1]   [0, 1, 0][u1]
[̇x2]   [-576, -2400, -4180][x2]   [0, 0, 1][u2]
dim4:
[̇x0] = [   0,     1,     0,     0][x0] + [1, 0, 0, 0][u0]
[̇x1]   [   0,     0,     1,     0][x1]   [0, 1, 0, 0][u1]
[̇x2]   [   0,     0,     0,     1][x2]   [0, 0, 1, 0][u2]
[̇x3]   [-576, -2400, -4180, -3980][x3]   [0, 0, 0, 1][u3]
dim5:
[̇x0] = [   0,     1,     0,     0,     0][x0] + [1, 0, 0, 0, 0][u0]
[̇x1]   [   0,     0,     1,     0,     0][x1]   [0, 1, 0, 0, 0][u1]
[̇x2]   [   0,     0,     0,     1,     0][x2]   [0, 0, 1, 0, 0][u2]
[̇x3]   [   0,     0,     0,     0,     1][x3]   [0, 0, 0, 1, 0][u3]
[̇x4]   [-576, -2400, -4180, -3980, -2273][x4]   [0, 0, 0, 0, 1][u4]
dim6:
[̇x0] = [   0,     1,     0,     0,     0,    0][x0] + [1, 0, 0, 0, 0, 0][u0]
[̇x1]   [   0,     0,     1,     0,     0,    0][x1]   [0, 1, 0, 0, 0, 0][u1]
[̇x2]   [   0,     0,     0,     1,     0,    0][x2]   [0, 0, 1, 0, 0, 0][u2]
[̇x3]   [   0,     0,     0,     0,     1,    0][x3]   [0, 0, 0, 1, 0, 0][u3]
[̇x4]   [   0,     0,     0,     0,     0,    1][x4]   [0, 0, 0, 0, 1, 0][u4]
[̇x5]   [-576, -2400, -4180, -3980, -2273, -800][x5]   [0, 0, 0, 0, 0, 1][u5]
dim7:
[̇x0] = [   0,     1,     0,     0,     0,    0,    0][x0] + [1, 0, 0, 0, 0, 0, 0][u0]
[̇x1]   [   0,     0,     1,     0,     0,    0,    0][x1]   [0, 1, 0, 0, 0, 0, 0][u1]
[̇x2]   [   0,     0,     0,     1,     0,    0,    0][x2]   [0, 0, 1, 0, 0, 0, 0][u2]
[̇x3]   [   0,     0,     0,     0,     1,    0,    0][x3]   [0, 0, 0, 1, 0, 0, 0][u3]
[̇x4]   [   0,     0,     0,     0,     0,    1,    0][x4]   [0, 0, 0, 0, 1, 0, 0][u4]
[̇x5]   [   0,     0,     0,     0,     0,    0,    1][x5]   [0, 0, 0, 0, 0, 1, 0][u5]
[̇x6]   [-576, -2400, -4180, -3980, -2273, -800, -170][x6]   [0, 0, 0, 0, 0, 0, 1][u6]
dim8:
[̇x0] = [   0,     1,     0,     0,     0,    0,    0,   0][x0] + [1, 0, 0, 0, 0, 0, 0, 0][u0]
[̇x1]   [   0,     0,     1,     0,     0,    0,    0,   0][x1]   [0, 1, 0, 0, 0, 0, 0, 0][u1]
[̇x2]   [   0,     0,     0,     1,     0,    0,    0,   0][x2]   [0, 0, 1, 0, 0, 0, 0, 0][u2]
[̇x3]   [   0,     0,     0,     0,     1,    0,    0,   0][x3]   [0, 0, 0, 1, 0, 0, 0, 0][u3]
[̇x4]   [   0,     0,     0,     0,     0,    1,    0,   0][x4]   [0, 0, 0, 0, 1, 0, 0, 0][u4]
[̇x5]   [   0,     0,     0,     0,     0,    0,    1,   0][x5]   [0, 0, 0, 0, 0, 1, 0, 0][u5]
[̇x6]   [   0,     0,     0,     0,     0,    0,    0,   1][x6]   [0, 0, 0, 0, 0, 0, 1, 0][u6]
[̇x7]   [-576, -2400, -4180, -3980, -2273, -800, -170, -20][x7]   [0, 0, 0, 0, 0, 0, 0, 1][u7]
"""

from typing import Tuple
import numpy as np
import pydrake.symbolic as sym


class HighDementionalDyanmics:
    def __init__(self, dim: int):
        assert dim >= 2, "The dimension of the system should be greater than 2"
        assert dim <= 8, "The dimension of the system should be less than 8"
        self.dim = dim
        self.paramters = np.array([-576, -2400, -4180, -3980, -2273, -800, -170, -20])
        last_row = self.paramters[0:dim]
        A = np.zeros((dim, dim))
        A[0 : dim - 1, 1:dim] = np.eye(dim - 1)
        A[dim - 1, 0:dim] = last_row
        self.A = A
        self.B = np.eye(dim)

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.dim
        assert u.shape[0] == self.dim
        return np.dot(self.A, x) + np.dot(self.B, u)

    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Write the control-affine dynamics xdot = f(x) + g(x) * u. Notice that
        f and g are polynomials of x.
        """
        assert x.shape[0] == self.dim
        f = np.array([sym.Polynomial(self.A[i, :].dot(x)) for i in range(self.dim)])
        g = self.B
        return f, g
