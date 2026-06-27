"""
This script defines the dynamics for a 2D quadrotor model.
The original model has the following state and input definitions:
states: x = [py, pz, theta, vy, vz, omega]
inputs: u = [t1, t2]
py is the position in y-axis, right direction is positive
pz is the position in z-axis, up direction is positive
theta is the rolling angle, counter-clockwise is positive
vy is the velocity in y-axis
vz is the velocity in z-axis
omega is the angular velocity
t1 is the thrust from the left propeller
t2 is the thrust from the right propeller

The original system dynamics is:
x0_dot = x3
x1_dot = x4
x2_dot = x5
x3_dot = -sin(x2) * (u0 + u1) / m
x4_dot = cos(x2) * (u0 + u1) / m - g
x5_dot = (u1 - u0) * l / I
where m is the mass, g is the gravitational acceleration, 
l is half the distance between the two propellers,
and I is the moment of inertia.

To polynomialize the system, we define the following states:
x0 = py
x1 = pz
x2 = sin(theta)
x3 = cos(theta) - 1
x4 = vy
x5 = vz
x6 = omega
Then the system dynamics becomes the following trig-poly affine dynamics:
x0_dot = x4
x1_dot = x5
x2_dot = x6 * (x3 + 1)
x3_dot = - x6 * x2
x4_dot = - x2 * (u0 + u1) / m
x5_dot = (x3 + 1) * (u0 + u1) / m - g
x6_dot = (u1 - u0) * l / I
with the equation constraint:
(x2)^2 + (x3 + 1)^2 = 1
"""
import numpy as np
import pydrake.symbolic as sym
from typing import Tuple, List
import pydrake.systems.framework as drake_sys_frame
import pydrake.systems.controllers as controllers

class QuadrotorDynamics2D(drake_sys_frame.LeafSystem):
    def __init__(self):
        super().__init__()
        num_control_input = 2
        num_system_states = 6
        self.DeclareVectorInputPort(
            name="u", size=num_control_input
            )
        state_index = self.DeclareContinuousState(
            num_state_variables=num_system_states
            )
        self.DeclareStateOutputPort(
            name="x", state_index=state_index
            )
        
        # Physical parameters
        self.m = 0.486  # mass
        self.g = 9.81  # gravitational acceleration
        self.l = 0.25  # half distance between propellers
        self.I = 0.00383  # moment of inertia

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
        # this function uses the ORIGINAL dynamics to compute the
        # state derivatives, which is used for simulation.
        xdot = np.zeros(6)
        xdot[0] = x[3]
        xdot[1] = x[4]
        xdot[2] = x[5]
        xdot[3] = -np.sin(x[2]) * (u[0] + u[1]) / self.m
        xdot[4] = np.cos(x[2]) * (u[0] + u[1]) / self.m - self.g
        xdot[5] = (u[1] - u[0]) * self.l / self.I
        
        
        return xdot
    
    def linearized_dyn_matrices(self) -> Tuple[
        np.ndarray,# the A matrix of the linearized system
        np.ndarray # the B matrix of the linearized system
        ]:
        # in order to apply the nomial LQR controller for simulation,
        # we need to linearize the forward dynamics at a hovering equilibrium
        # point: x = [0, 0, 0, 0, 0, 0], u = [m*g/2, m*g/2]
        # it should be noted that any hovering point with theta = 0
        # can be used for linearization. But the LQR controller
        # will only stabilize the system at the origin.
        # It should be noted that the linearized system state and control 
        # variables are not the original states and controls, but the offset
        # from the equilibrium point.
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -self.g, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1/self.m, 1/self.m],
            [-self.l/self.I, self.l/self.I]
        ])
        return A, B
        
    def trig_poly_affine_dynamics(
        self,
        x:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == (7,)
        f = np.zeros(7)
        g = np.zeros((7, 2))
        f = np.array([
            sym.Polynomial(x[4]),
            sym.Polynomial(x[5]),
            sym.Polynomial(x[6] * (x[3] + 1)),
            sym.Polynomial(- x[6] * x[2]),
            sym.Polynomial(0),
            sym.Polynomial(-self.g),
            sym.Polynomial(0)
        ])
        g = np.array([
            [sym.Polynomial(0), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(0)],
            [sym.Polynomial(-x[2]/self.m), sym.Polynomial(-x[2]/self.m)],
            [sym.Polynomial((x[3] + 1)/self.m), sym.Polynomial((x[3] + 1)/self.m)],
            [sym.Polynomial(-self.l/self.I), sym.Polynomial(self.l/self.I)]
        ])
        return f, g

    def equation_constraint(
        self,
        x:np.ndarray
    ) -> np.ndarray:
        """
        This function defines the equation constraint for the 2D quadrotor model.
        The equation constraint is:
        (x2)^2 + (x3 + 1)^2 = 1
        """
        assert x.shape == (7,)
        return np.array([sym.Polynomial(x[2]**2 + (x[3] + 1)**2 - 1)])

    def control_limits(self, output_polys = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function defines the control limits of the system
        u0 ∈ [0, m*g]
        u1 ∈ [0, m*g]
        """
        u0_min = 0.0
        u0_max = 1.5 * self.m * self.g
        u1_min = 0.0
        u1_max = 1.5 * self.m * self.g
        if output_polys:
            A = np.array([
                [sym.Polynomial(1), sym.Polynomial(0)],
                [sym.Polynomial(-1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(1)],
                [sym.Polynomial(0), sym.Polynomial(-1)]
            ])
            b = np.array([
                sym.Polynomial(u0_max),
                sym.Polynomial(-u0_min),
                sym.Polynomial(u1_max),
                sym.Polynomial(-u1_min)
            ])
            return A, b
        else:
            A = np.array([
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1]
            ])
            b = np.array([
                u0_max,
                -u0_min,
                u1_max,
                -u1_min
            ])
            return A, b


# for simulation, we define the nominal controller as an LQR for the 
# linearized system model.
class NominalController(drake_sys_frame.LeafSystem):
    def __init__(self, Q:np.ndarray, R:np.ndarray):
        super().__init__()
        self.nx = 6
        self.nu = 2
        self.Q = Q
        self.R = R
        assert Q.shape == (self.nx, self.nx)
        assert R.shape == (self.nu, self.nu)

        self.quadrotor = QuadrotorDynamics2D()
        A, B = self.quadrotor.linearized_dyn_matrices()
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

        # it should be noted that LQR takes the offset state and 
        # computes the offset control. But since the equilibrium 
        # state is the origin, then in this setup our linear state
        # are also the orignal states. But the control output for
        # the original system should be the offset control plus the
        # equilibrium control, which is [m*g/2, m*g/2]
        u_d += np.array([
            self.quadrotor.m * self.quadrotor.g / 2,
            self.quadrotor.m * self.quadrotor.g / 2
            ])
        output.set_value(u_d)


# state convertion:
class StateConverter(drake_sys_frame.LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("x_original", 6)
        self.DeclareVectorOutputPort("x_converted", 7, self.calc_converted_state)
    
    def calc_converted_state(
        self,
        context: drake_sys_frame.Context,
        output
    ):
        x_original = self.get_input_port(0).Eval(context)
        py = x_original[0]
        pz = x_original[1]
        theta = x_original[2]
        vy = x_original[3]
        vz = x_original[4]
        omega = x_original[5]
        if theta > np.pi:
            theta -= 2*np.pi
        if theta < -np.pi:
            theta += 2*np.pi
        x_extended = np.array([
            py,
            pz,
            np.sin(theta),
            np.cos(theta) - 1,
            vy,
            vz,
            omega
        ])
        output.set_value(x_extended)



