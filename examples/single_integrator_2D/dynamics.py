# """
# This script defines a simple 2D single integrator system,
# which is a common example in control theory. The dynamics
# of the system are given by:
# ẋ = u

# where x is the state of the system and u is the input.
# Since the integrator is 2D, then we have both x, u ∈ ℝ².
# """

import numpy as np
from typing import Tuple

import pydrake.symbolic as sym
import pydrake.systems.framework as drake_sys_frame

# this class defines the dynamics of the plant:
class SingleIntegrator2D(drake_sys_frame.LeafSystem):
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
        xdot = self.forward_dynamics(x, u)
        derivatives.SetFromVector(xdot)

    def forward_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray
    )->np.ndarray:
        assert x.shape == (2,)
        assert u.shape == (2,)
        f = np.zeros(2)
        g = np.eye(2)
        return f + g @ u

    def affine_dynamics(self, x: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        f = np.array([
            sym.Polynomial(),
            sym.Polynomial()
            ])
        g = np.array([
            [sym.Polynomial(1), sym.Polynomial()],
            [sym.Polynomial(), sym.Polynomial(1)],
            ])
        return f, g

    def control_limits(self, output_poly = True)->Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the control limits of the system
        u₁ ∈ [-1, 1],
        u₂ ∈ [-1, 1]
        """
        u1_min = -1
        u1_max = 1
        u2_min = -1
        u2_max = 1
        if output_poly:
            Au = np.array([
                [sym.Polynomial(1), sym.Polynomial()], 
                [sym.Polynomial(-1), sym.Polynomial()], 
                [sym.Polynomial(), sym.Polynomial(1)], 
                [sym.Polynomial(), sym.Polynomial(-1)]
                ])
            bu = np.array([
                sym.Polynomial(u1_max), 
                sym.Polynomial(-u1_min), 
                sym.Polynomial(u2_max), 
                sym.Polynomial(-u2_min)
                ])
        else:
            Au = np.array([
                [1, 0], 
                [-1, 0], 
                [0, 1], 
                [0, -1]
                ])
            bu = np.array([
                u1_max, 
                -u1_min, 
                u2_max, 
                -u2_min
                ])
        return Au, bu

# this class defines a nomial controller for the 
# plant to reach a goal point.
class NominalController(drake_sys_frame.LeafSystem):
    def __init__(self, waypoints: np.ndarray, gain: float=1.0):
        super().__init__()
        self.nx = 2
        self.nu = 2
        assert waypoints.shape[1] == 2
        self.waypoints = waypoints
        self.gain = gain
        # declare controller's ports:
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
        state_val = self.get_input_port(0).Eval(context)
        # print(f"Current state: {state_val}")
        dist_to_current_goal = np.linalg.norm(state_val - self.waypoints[0])
        if dist_to_current_goal > 1e-6:
            u_d = self.gain * (self.waypoints[0] - state_val)
        elif dist_to_current_goal <= 1e-6 and self.waypoints.shape[0] > 1:
            self.waypoints = self.waypoints[1:]
            u_d = self.gain * (self.waypoints[0] - state_val)
        elif dist_to_current_goal <= 1e-6 and self.waypoints.shape[0] == 1:
            u_d = np.zeros(self.nu)

        output.set_value(u_d)



