import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

"""
This file includes the bilinear alternation of a CLF synthesis. 
This is the problem formulation:
Given a set of points x1,...,xn, find a CLF V(x) such that:
1. V(x) is an SOS polynomial that goes through the origin, V(x=0)=0.
2. The sublevel set {x | V(x) <= 1} contains all the points x1,...,xn.
3. For all x in the sublevel set, there exists a u such that 
        dVdx * f(x) + dVdx * g(x) * u <= -kappa * V.
4. The superlevel set {x | V(x) >= 1} should include a ball with radius r.

From the SOS perspective, we have the following SOS constraints:
1. The sublevel set contains the ball:
    
"""


class ClfConstraint:
    """
    Add the linear constraint dVdx * f(x) + dVdx * g(x) * u <= -kappa * V on u.
    """

    def __init__(
        self,
        V: sym.Polynomial,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        kappa: float,
    ):
        self.V = V
        dVdx = V.Jacobian(x)
        dVdx_times_f = dVdx.dot(f)
        dVdx_times_g = dVdx @ g
        self.rhs = -kappa * V - dVdx_times_f
        self.lhs_coeff = dVdx_times_g
        self.x = x

    def add_to_prog(
        self, prog: solvers.MathematicalProgram, x_val: np.ndarray, u: np.ndarray
    ) -> solvers.Binding[solvers.LinearConstraint]:
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        constraint = prog.AddLinearConstraint(lhs_coeff, -np.inf, rhs, u)
        return constraint
