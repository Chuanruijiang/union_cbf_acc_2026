"""
This script defines the dynamics of a double intergrator:
if the integrator's position domian is 1D, the dynamics
would be:
̇x0 = x1;
̇x1 = u
We only use this example to check the correctness of coding,
we don't actually use this 1D double integrator dynamics
in any experiments.

If the integrator is 2D position domian, the dynamics
would be: 
̇x0 = x2
̇x1 = x3
̇x2 = u0
̇x3 = u1
This is the main dynamics we use in the experiments.
"""

import numpy as np
import pydrake.symbolic as sym
from typing import Tuple
import pydrake.systems.framework as drake_sys_frame
import pydrake.systems.controllers as controllers

class DoubleIntegratorPlant2D(drake_sys_frame.LeafSystem):
    def __init__(self):
        super().__init__()
        num_control_input = 2
        num_system_states = 4
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
        assert x.shape == (4,)
        assert u.shape == (2,)
        xdot = np.array([
            x[2],
            x[3],
            u[0],
            u[1]
        ])
        return xdot
    
    def affine_dynamics(self, x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == (4,)
        f = np.array([
            sym.Polynomial(x[2]),
            sym.Polynomial(x[3]),
            sym.Polynomial(0),
            sym.Polynomial(0)
        ])
        g = np.array([
            [sym.Polynomial(0), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(0)],
            [sym.Polynomial(1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(1)],
        ])
        return f, g
    
    def control_input_limits(self, output_poly=True)->Tuple[np.ndarray, np.ndarray]:
        u_min = -5
        u_max = 5
        if output_poly:
            Au = np.array([
                [sym.Polynomial(1), sym.Polynomial(0)],
                [sym.Polynomial(-1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(1)],
                [sym.Polynomial(0), sym.Polynomial(-1)],
            ])
            bu = np.array([
                sym.Polynomial(u_max),
                sym.Polynomial(-u_min),
                sym.Polynomial(u_max),
                sym.Polynomial(-u_min),
            ])
        else:
            Au = np.array([
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
            ])
            bu = np.array([
                u_max,
                -u_min,
                u_max,
                -u_min,
            ])
        return Au, bu

class DoubleIntegratorPlant1D(drake_sys_frame.LeafSystem):
    def __init__(self):
        super().__init__()
        num_control_input = 1
        num_system_states = 2
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
        assert x.shape == (2,)
        assert u.shape == (1,)
        xdot = np.array([
            x[1],
            u[0]
        ])
        return xdot
    
    def affine_dynamics(self, x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == (2,)
        f = np.array([
            sym.Polynomial(x[1]),
            sym.Polynomial(0)
        ])
        g = np.array([
            [sym.Polynomial(0)],
            [sym.Polynomial(1)]
        ])
        return f, g

    def control_input_limits(self)-> Tuple[np.ndarray, np.ndarray]:
        u_min = -1
        u_max = 1
        Au = np.array([
        [sym.Polynomial(1)], 
        [sym.Polynomial(-1)]
        ])
        bu = np.array([
            sym.Polynomial(u_max), 
            sym.Polynomial(-u_min)
        ])
        return Au, bu
    
# Define the Nomial controller for the 2D double integrator system.
# since the double integrator dynamics is linear, we can just use LQR
# as the nominal controller,
class LQRNominalController2D(drake_sys_frame.LeafSystem):
    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        super().__init__()
        self.Q = Q
        self.R = R
        self.nx = 4
        self.nu = 2
        
        # compute LQR control gain.
        # the double integrator dynamics matrices:
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
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
        position_val = state_val[0:2]
        print("Current state:", state_val)

        u_d = -self.K_lqr @ state_val
        output.set_value(u_d)

# we don't need to define state converter since the double integrator
# dynamics has no trigonometric terms in its dynamics so we don't have
# extended states.

