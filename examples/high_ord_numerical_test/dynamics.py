"""
We use the following example to show the scalability and the upper 
performance limit of our verification approach for union of HOCBFs.

The experiemnt set up is the following:
We have a sistem consisting of two parts, each part is an high-order
integrator system defined as:
y(i) = u
where y(i) is the i-th order derivative of the output y, and u is 
the control input.

We have two such systems, and the let's define the first output as
y0, and the second output is y1, the control for the first system
is u0, and the control for the second system is u1. Now, we wanted
to keep the whole system stay in the union of the following two
sets:
X1 = {y|h(y) = y0 + y1 >=0}
X2 = {y|h(y) = y0 - y1 >=0}
"""

from dataclasses import dataclass
import numpy as np
import pydrake.symbolic as sym
from typing import Tuple, List
import pydrake.systems.framework as drake_sys_frame

@ dataclass
class HighOrdSystem:
    # we only use this system to show verification scalability
    # and record verification time so we do not need this class
    # to be a DrakeLeafSystem
    order: int
    def single_f_component(self, x:np.ndarray) -> Tuple[np.ndarray]:
        assert x.shape == (self.order,)
        f_mat = np.zeros((self.order, self.order))
        f_mat[:-1, 1:] = np.eye(self.order-1)
        f = f_mat @ x[:self.order]
        f_mat_poly = f.reshape(-1,)
        for i in range(self.order):
            f_mat_poly[i] = sym.Polynomial(f_mat_poly[i])
        return f_mat_poly

    def affine_dynamics(self, x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == (2*self.order,)
        f0 = self.single_f_component(x[:self.order])
        f1 = self.single_f_component(x[self.order:])
        f = np.concatenate((f0, f1), axis=0)
        g = np.zeros((2*self.order, 2))
        g[self.order-1, 0] = 1
        g[-1, 1] = 1
        return f, g
    
    def control_limits(self, output_polys = True) -> Tuple[np.ndarray, np.ndarray]:
        u_min = -10
        u_max = 10
        if not output_polys:
            A = np.array([
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
            ])
            c = np.array([
                u_max,
                -u_min,
                u_max,
                -u_min,
            ])
        else:
            A = np.array([
                [sym.Polynomial(1), sym.Polynomial(0)],
                [sym.Polynomial(-1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(1)],
                [sym.Polynomial(0), sym.Polynomial(-1)],
            ])
            c = np.array([
                sym.Polynomial(u_max),
                sym.Polynomial(-u_min),
                sym.Polynomial(u_max),
                sym.Polynomial(-u_min),
            ])
        return A, c


