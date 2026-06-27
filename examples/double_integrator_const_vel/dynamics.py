"""
Besidest the defined dyanmics above, we also interested in
a setup such that a double integrator with 2D position domain
always has a constant velocity along x direction and we can
only control its acceleration along y direction. The dynamics
in this case would be:
x0 = px, x1 = py, x2 = vy
x0_dot = vx_const
x1_dot = x2
x2_dot = u0
This kind of example can be used to show that the verification
under switching policy I can verifies some types of systems 
while the verification under swithcing policy II cannot.
"""

import numpy as np
import pydrake.symbolic as sym
from typing import Tuple
import pydrake.systems.framework as drake_sys_frame
import pydrake.systems.controllers as controllers

class DoubleIntegratorPlantConstantVel(drake_sys_frame.LeafSystem):
    def __init__(self, vx_const: float):
            super().__init__()
            self.vx_const = vx_const
            num_control_input = 1
            num_system_states = 3
            self.DeclareVectorInputPort(
                name="u", size=num_control_input
                )
            state_index = self.DeclareContinuousState(
                num_state_variables=num_system_states
                )
            self.DeclareStateOutputPort(
                name="x", state_index=state_index
                )
    
    def DoCalcTimeDerivatives(
        self,
        context: drake_sys_frame.Context,
        derivatives: drake_sys_frame.ContinuousState,
    ):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.EvalVectorInput(context, 0).CopyToVector()
        xdot = self.forward_dynamics(x, u)
        derivatives.SetFromVector(xdot)
    
    def forward_dynamics(
        self,
        x:np.ndarray,
        u:np.ndarray
    ) -> np.ndarray:
        assert x.shape == (3,)
        assert u.shape == (1,)
        xdot = np.array([
            self.vx_const,
            x[2],
            u[0]
        ])
        return xdot

    def affine_dynamics(
        self, 
        x:np.ndarray
     ) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == (3,)
        f = np.array([
            sym.Polynomial(self.vx_const),
            sym.Polynomial(x[2]),
            sym.Polynomial(0)
        ])
        g = np.array([
            [sym.Polynomial(0)],
            [sym.Polynomial(0)],
            [sym.Polynomial(1)]
        ])
        return f, g

    def control_input_limits(
        self,
        output_poly=True
    )-> Tuple[np.ndarray, np.ndarray]:
        u_min = -1
        u_max = 1
        if output_poly:
            Au = np.array([
            [sym.Polynomial(1)], 
            [sym.Polynomial(-1)]
            ])
            bu = np.array([
                sym.Polynomial(u_max), 
                sym.Polynomial(-u_min)
            ])
            return Au, bu
        else:
            Au = np.array([
            [1], 
            [-1]
            ])
            bu = np.array([
                u_max, 
                -u_min
            ])
            return Au, bu


# Define the Nomial controller for the 2D double integrator system with
# constant velocity along x direction.
# we use this nominal controller to kee the system follows the x1= 0
# axis.
class LQRNominalController1D(drake_sys_frame.LeafSystem):
    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        super().__init__()
        self.Q = Q
        self.R = R
        self.nx = 3
        self.nu = 1
        
        # compute LQR control gain.
        # the double integrator dynamics matrices:
        A = np.array([
            [0, 1],
            [0, 0]
        ])
        B = np.array([
            [0], 
            [1]
        ])
        K_lqr, _ = controllers.LinearQuadraticRegulator(A, B, self.Q, self.R)
        self.K_lqr = K_lqr

        # Declare input and output ports
        self.DeclareVectorInputPort("state", self.nx)
        self.action_output_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()
        
    def action_output_port(self):
        return self.get_output_port(self.action_output_index)

    def calc_action(
        self,
        context: drake_sys_frame.Context,
        output
    ):
        # we don't need waypoints for LQR since LQR is to stabilize
        # the system to origin in the state space
        state_val = self.get_input_port(0).Eval(context)
        controlled_states = state_val[1:3]
        print("Current state:", state_val)

        u_d = -self.K_lqr @ controlled_states
        output.set_value(u_d)

# we don't need to define state converter since the double integrator
# dynamics has no trigonometric terms in its dynamics so we don't have
# extended states.

