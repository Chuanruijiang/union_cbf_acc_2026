
import numpy as np
import pydrake.symbolic as sym
from typing import Tuple

class NonlinearToyPlant():
    def __init__(self):
        pass
    def affine_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the system dyanmics of a single integrator.
        The system dynamics is:
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

    def control_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the control limits of the system
        """
        u1_min = -5
        u1_max = 5
        u2_min = -5
        u2_max = 5
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
