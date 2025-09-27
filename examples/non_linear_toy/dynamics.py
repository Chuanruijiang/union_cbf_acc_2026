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
import pydrake.systems.framework as drake_sys_frame
from typing import Tuple

class NonlinearToyPlant(drake_sys_frame.LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort(name="u", size=2)
        state_index = self.DeclareContinuousState(num_state_variables=2)
        self.DeclareStateOutputPort(name="x", state_index=state_index)

    def DoCalcTimeDerivatives(
        self,
        context: drake_sys_frame.Context,
        derivatives: drake_sys_frame.ContinuousState,
    ):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.EvalVectorInput(context, 0).CopyToVector()
        xdot = self.system_dynamics(x, u)
        derivatives.SetFromVector(xdot)

    def system_dynamics(
        self,
        th_gam:np.ndarray,
        u:np.ndarray
    ) -> np.ndarray:
        """
        We use thm_gam to denote the state vector [theta, gamma]
        """
        assert th_gam.shape == (2,)
        assert u.shape == (2,)
        xdot = np.array([
            u[0],
            -sym.sin(th_gam[0]) - u[1]
        ])
        return xdot
    
    def affine_dynamics(
        self,
        x:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the system dyanmics of a single integrator.
        The system dynamics is:
        ̇x0 = -x1 - u1,
        x1 = (x2 + 1) * u0,
        ̇x2 = -x1 * u0
        we can see that the state variables are not present in dynamics
        for the single integrator.
        """
        assert x.shape == (3,)
        f = np.array([
            sym.Polynomial(-x[1]),
            sym.Polynomial(0),
            sym.Polynomial(0)
        ])
        g = np.array([
            [sym.Polynomial(0), sym.Polynomial(-1)],
            [sym.Polynomial(x[2] + 1), sym.Polynomial(0)],
            [sym.Polynomial(-x[1]), sym.Polynomial(0)]
        ])
        return f, g

    def approximate_dynamics(
            self,
            x: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This dynamics repalce the sin as the third order Taylor expansion
        theta = x[0], gamma = x[1]
        ̇x0 = u0,
        ̇x1 = -x0 + x0^3/6 - u1
        """
        assert x.shape == (2,)
        f = np.array([
            sym.Polynomial(0),
            sym.Polynomial(-x[0] + (x[0]**3)/6)
        ])
        g = np.array([
            [sym.Polynomial(1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(-1)]
        ])
        return f, g

    def state_eq_constraint(
        self,
        x: np.array
    ) -> sym.Polynomial:
        return np.array([sym.Polynomial(x[1]**2 + x[2]**2 + 2*x[2])])

    def control_limits(
        self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the control limits of the system
        u₁ ∈ [-0.5, 0.5],
        u₂ ∈ [-5, 5]
        """
        u1_min = -0.5
        u1_max = 0.5
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

    def original_to_extended_state_space(
        self,
        input_points: np.ndarray
    ) -> np.ndarray:
        """
        This function maps the original state space to the extended
        state space.
        input_points: shape (N, 2), where N is the number of points.
        output_points: shape (N, 3), where N is the number of points.
        for the input point [gamma, theta], the output point is
        [gamma, sin(theta), cos(theta)-1]
        """
        assert input_points.shape[1] == 2
        N = input_points.shape[0]
        output_points = np.zeros((N, 3))
        for i in range(N):
            output_points[i, 0] = input_points[i, 0]
            output_points[i, 1] = np.sin(input_points[i, 1])
            output_points[i, 2] = np.cos(input_points[i, 1]) - 1
        return output_points

