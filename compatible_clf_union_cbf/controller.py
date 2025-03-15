import numpy as np
from typing import List, Optional

import pydrake.solvers as solvers
import pydrake.symbolic as sym

import compatible_clf_union_cbf.clf as clf
import compatible_clf_union_cbf.union_cbf as cbf
import compatible_clf_union_cbf.utils as utils


def cbf_clf_qp(
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
    solver_options: Optional[solvers.SolverOptions] = None,
) -> np.ndarray:
    """
    Online CBF-CLF QP controller, the outout is the control input u.
    """
    # TODO: for union of CBFs, this function needs to be rewritten

    # initialize the clf and cbf constraint objects:
    clf_constraint = clf.ClfConstraint(V, f, g, x, kappa_V)
    cbf_constraints = [
        cbf.CbfConstraint(
            h=h[i],
            f=f,
            g=g,
            x=x,
            kappa=kappa_h[i],
            relative_degree=(
                relative_degrees[i]
                if relative_degrees is not None
                else None)
            )
        for i in range(h.shape[0])
    ]

    # the input CBFs are assumed to be in the order of [h0, h1, h2, ...], 
    activated_idx = 0
    for i in range(h.shape[0]):
        if h[i].EvaluateIndeterminates(x, x_value) >= 0:
            activated_idx = i
            break

    # create the QP:
    prog = solvers.MathematicalProgram()
    nu = g.shape[1]
    u = prog.NewContinuousVariables(nu, "u")
    # (1)add the cost function u^T*Q*u:
    prog.AddQuadraticCost(Q, np.zeros((nu,)), u, is_convex=True)
    # (2)add the clf constraint:
    clf_constraint.add_to_prog(prog, x_value, u)
    # (3)add the cbf constraints:
    cbf_constraints[activated_idx].add_to_prog(prog, x_value, u)
    # (4)add the input limit constraints:
    if Au is not None:
        assert bu is not None
        prog.AddLinearConstraint(Au, np.full_like(bu, -np.inf), bu, u)
    # solve the QP:
    result = utils.solve_with_id(prog, solver_id, solver_options)
    assert result.is_success()
    u_val = result.GetSolution(u)

    return u_val
