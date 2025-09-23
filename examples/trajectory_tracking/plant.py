import numpy as np
import pydrake.symbolic as sym
from typing import Tuple
import pydrake.systems.framework as drake_sys_frame

class SingleIntegratorPlant(drake_sys_frame.LeafSystem):
    """
    This class defines the single integrator control system. It is inherited
    from LeafSystem, meaning that we will use it for both the union CBF
    verfication, and the simulation of the switching CBF-QP as well.
    The initialization function defines the input, state and output ports.
    The function DoCalcTimeDerivatives and system_dynamics
    defines the dynamics of the system. 
    These functions are used for simulation.
    For verification, we use the function "affine_dynamics".
    """
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
        x:np.ndarray,
        u:np.ndarray
    ) -> np.ndarray:
        assert x.shape == (2,)
        assert u.shape == (2,)
        xdot = np.array([u[0], u[1]])
        return xdot
    
    def affine_dynamics(
        self,
        x:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the system dyanmics of a single integrator.
        The system dynamics is:
        ̇x₀ = u₀ ; ̇x₁ = u₁
        we can see that the state variables are not present in dynamics
        for the single integrator.
        """
        assert x.shape == (2,)
        f = np.array([sym.Polynomial(), sym.Polynomial()])
        g = np.array(
            [
                [sym.Polynomial(1), sym.Polynomial()],
                [sym.Polynomial(), sym.Polynomial(1)],
            ]
        )
        return f, g

    def control_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the control limits of the system
        u₁ ∈ [-0.4, 0.4],
        u₂ ∈ [-0.4, 0.4]
        """
        u1_min = -0.5
        u1_max = 0.5
        u2_min = -0.5
        u2_max = 0.5
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
