from typing import Optional

import numpy as np
from typing import List, Optional

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.framework

import compatible_clf_union_cbf.utils as utils


class ClfCbfController(pydrake.systems.framework.LeafSystem):
    def __init__(
        self,
        f: np.ndarray,
        g: np.ndarray,
        V: sym.Polynomial,
        h: np.ndarray,
        x: np.ndarray,
        kappa_V: float,
        kappa_h: np.ndarray,
        Qu: np.ndarray,
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
        solver_id: Optional[solvers.SolverId],
        solver_options: Optional[solvers.SolverOptions],
    ):
        super().__init__()
        self.nu = g.shape[1]
        self.DeclareVectorInputPort("state", x.size)
        self.action_output_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()
        self.V = V
        self.h = h
        self.x = x
        self.V_output_index = self.DeclareVectorOutputPort(
            "V", 1, self.calc_V
        ).get_index()
        self.h_output_index = self.DeclareVectorOutputPort(
            "h", h.size, self.calc_h
        ).get_index()
        self.clf_constraint = clf.ClfConstraint(V, f, g, x, kappa_V)
        self.cbf_constraint = [
            cbf.CbfConstraint(h[i], f, g, x, kappa_h[i])
            for i in range(h.size)
        ]
        self.Qu = Qu
        self.Au = Au
        self.bu = bu
        self.solver_id = solver_id
        self.solver_options = solver_options

    def action_output_port(self):
        return self.get_output_port(self.action_output_index)

    def clf_output_port(self):
        return self.get_output_port(self.V_output_index)

    def cbf_output_port(self):
        return self.get_output_port(self.h_output_index)

    def calc_action(self, context: pydrake.systems.framework.Context, output):
        x_val: np.ndarray = self.get_input_port(0).Eval(context)
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        prog.AddQuadraticCost(self.Qu, np.zeros((self.nu,)), u, is_convex=True)
        self.clf_constraint.add_to_prog(prog, x_val, u)
        for cbf_cnstr in self.cbf_constraint:
            cbf_cnstr.add_to_prog(prog, x_val, u)
        if self.Au is not None:
            assert self.bu is not None
            prog.AddLinearConstraint(
                self.Au, np.full_like(self.bu, -np.inf), self.bu, u
            )
        result = utils.solve_with_id(
            prog, self.solver_id, self.solver_options
        )
        assert result.is_success()
        u_val = result.GetSolution(u)
        output.set_value(u_val)

    def calc_V(self, context: pydrake.systems.framework.Context, output):
        x_val: np.ndarray = self.get_input_port(0).Eval(context)
        env = {self.x[i]: x_val[i] for i in range(self.x.size)}
        V_val = self.V.Evaluate(env)
        print(f"time={context.get_time()}, V_val={V_val}")
        output.set_value(np.array([V_val]))

    def calc_h(self, context: pydrake.systems.framework.Context, output):
        x_val: np.ndarray = self.get_input_port(0).Eval(context)
        env = {self.x[i]: x_val[i] for i in range(self.x.size)}
        h_val = np.array([h_i.Evaluate(env) for h_i in self.h])
        print(f"time={context.get_time()}, h_val={h_val}")
        output.set_value(h_val)


def cbf_clf_QP(
        x_value: np.ndarray,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        V: sym.Polynomial,
        h: np.ndarray,
        kappa_V: float,
        kappa_h: List[List[float]],
        relative_degrees: List[int],
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
        Q: np.ndarray,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None
        ) -> np.ndarray:
    """
    Online CBF-CLF QP controller, the outout is the control input u.
    """
    # initialize the clf and cbf constraint objects:
    clf_constraint = clf.ClfConstraint(V, f, g, x, kappa_V)
    cbf_constraints = [
        cbf.CbfConstraint(h[i], f, g, x, kappa_h[i], relative_degrees[i]) 
        for i in range(h.shape[0])
        ]
    # create the QP:
    prog = solvers.MathematicalProgram()
    nu = g.shape[1]
    u = prog.NewContinuousVariables(nu, "u")
    # (1)add the cost function u^T*Q*u:
    prog.AddQuadraticCost(Q, np.zeros((nu,)), u, is_convex=True)
    # (2)add the clf constraint:
    clf_constraint.add_to_prog(prog, x_value, u)
    # (3)add the cbf constraints:
    for each_cbf_constraint in cbf_constraints:
        each_cbf_constraint.add_to_prog(prog, x_value, u)
    # (4)add the input limit constraints:
    if Au is not None:
        assert bu is not None
        prog.AddLinearConstraint(Au, np.full_like(bu, -np.inf), bu, u)
    # solve the QP:
    result = utils.solve_with_id(prog, solver_id, solver_options)
    assert result.is_success()
    u_val = result.GetSolution(u)

    return u_val