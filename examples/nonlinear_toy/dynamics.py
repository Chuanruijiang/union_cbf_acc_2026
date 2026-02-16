# """
# This file defines the dynamics of a non-linear toy system 
# that we are going to use for verification time comparison
# between the verif-I and verif-II methods. The dynamics are
# given by:

# x0_dot = x1 + (0.2x0^2+0.2 x1 + 1)u0,
# x1_dot = (x2 + x0^3/3 + x1) + (-0.2x1^2 + 0.2x0 + 4)u1,

# This dynamics is given by the paper:

# H. Wang, K. Margellos, and A. Papachristodoulou, 
# “Safe and stable filter design using a relaxed 
# compatibitlity control barrier–lyapunov condition,” 
# arXiv preprint arXiv:2407.00414, 2024.

# Since we won't use this system for any real-time simulations
# then we simpliy define this system as an independent class
# (not an inherited class from the Drake's system class).
# """

import numpy as np
import pydrake.symbolic as sym
from typing import Tuple

class NonlinearToyPlant():
    def __init__(self):
        pass
    
    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        definet the polynomial system dynamics:
        ̇x0 = x1 + (0.2x0^2+0.2 x1 + 1)u0,
        x1 = (x2 + x0^3/3 + x1) + (-0.2x1^2 + 0.2x0 + 4)u1,
        """
        assert x.shape == (2,)
        f = np.array([
            sym.Polynomial(x[1]),
            sym.Polynomial(x[0] + x[0]**3/3 + x[1])
        ])
        g = np.array([
            [sym.Polynomial(0.2*x[0]**2 + 0.2*x[1] + 1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(-0.2*x[1]**2 + 0.2*x[0] + 4)]
        ])
        return f, g

    def control_limits(
        self, 
        output_poly = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the control limits of the system
        """
        u1_min = -5
        u1_max = 5
        u2_min = -5
        u2_max = 5
        if output_poly:
            Au = np.array([
                [sym.Polynomial(1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(1)],
                [sym.Polynomial(-1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(-1)]
            ])
            bu = np.array([
                sym.Polynomial(u1_max),
                sym.Polynomial(u2_max),
                sym.Polynomial(-u1_min),
                sym.Polynomial(-u2_min)
            ])
        else:
            Au = np.array([
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1]
            ])
            bu = np.array([
                u1_max,
                u2_max,
                -u1_min,
                -u2_min
            ])
        return Au, bu




