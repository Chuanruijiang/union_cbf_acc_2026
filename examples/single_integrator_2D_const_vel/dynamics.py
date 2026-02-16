# """
# In this file, we define a type of single integrator system in 2D 
# with constant horizontal velocity. This example will be used to 
# illustrate the case where verif-I could verify but verif-II fails.
# The dynamics of the system are given by:
# x0_dot = v_const
# x1_dot = u
# """

import numpy as np
from typing import Tuple

import pydrake.symbolic as sym
import pydrake.systems.framework as drake_sys_frame

# this class defines the dynamics of the plant:
class SingleIntegrator(drake_sys_frame.LeafSystem):
    def __init__(self, v_const: float = 1.0):
        super().__init__()
        self.v_const = v_const
        self.DeclareVectorInputPort(name="u", size=1)
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
        assert u.shape == (1,)
        f = np.array([self.v_const, 0.0])
        g = np.array([[0.0], [1.0]])
        return f + g @ u

    def affine_dynamics(self, x: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        assert x.shape == (2,)
        f = np.array([
            sym.Polynomial(self.v_const),
            sym.Polynomial()
            ])
        g = np.array([
            [sym.Polynomial(0)],
            [sym.Polynomial(1)],
            ])
        return f, g

    def control_limits(self, output_poly = True)->Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the control limits of the system
        u ∈ [-1.5, 1.5]
        """
        u_min = -1.5
        u_max = 1.5
        if output_poly:
            Au = np.array([
                [sym.Polynomial(1)], 
                [sym.Polynomial(-1)],
                ])
            bu = np.array([
                sym.Polynomial(u_max), 
                sym.Polynomial(-u_min)
                ])
        else:
            Au = np.array([
                [1], 
                [-1],
                ])
            bu = np.array([
                u_max, 
                -u_min
                ])
        return Au, bu

# this class defines a nomial controller for the 
# plant to reach a goal point.
class NominalController(drake_sys_frame.LeafSystem):
    def __init__(self, target_height: float, gain: float=1.0):
        super().__init__()
        self.nx = 2
        self.nu = 1
        self.target_height = target_height
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
        u_d = np.array([self.gain * (self.target_height - state_val[1])])

        output.set_value(u_d)



