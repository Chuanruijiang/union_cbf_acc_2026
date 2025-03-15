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

def compared_cbf_qp(
    x_value: np.ndarray,
    x: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    h_polys: np.ndarray,
    kappa_h: float,
    K_desired_control: float,
    smooth_k: float,
    buffer_b: float
) -> np.ndarray:
    """
    This function simulates the cbf rule proposed in the "union of sets" 
    section of the following paper:
    "Composing Control Barrier Functions for Complex Safety Specifications" 
    by Tamas Molnar et al.
    """
    assert x_value.shape[0] == x.shape[0]
    assert f.shape[0] == x.shape[0]
    assert g.shape[0] == f.shape[0]
    num_u = g.shape[1]
    # compute each h_i(x) in h_polys at x_value:
    h_i_values = np.array([
        h_poly.EvaluateIndeterminates(x, x_value)[0]
        for h_poly in h_polys
    ])
    # compute the union cbf h(x):
    h_value = (1/smooth_k) * np.log(
        np.sum(np.exp(smooth_k * h_i_values))
    ) - buffer_b/smooth_k
    # compute λ(x):
    lambda_i_values = np.exp(smooth_k * h_i_values - smooth_k * (h_value + buffer_b))
    # compute Lfh_i(x):
    num_h_i = h_polys.shape[0]
    Lfh_i_values = np.zeros((num_h_i,))
    for i in range(num_h_i):
        Lfh_i_values[i] = utils.lie_derivative(
            poly=h_polys[i],
            vector_feild=f,
            variables=x,
            pow=1
        ).EvaluateIndeterminates(x, x_value)[0]
    # compute Lgh_i(x):
    Lgh_i_values = np.zeros((num_h_i, num_u))
    for i in range(num_h_i):
        Lgh_i = utils.lie_derivative(
            poly=h_polys[i],
            vector_feild=g,
            variables=x,
            pow=1
        )
        for j in range(num_u):
            Lgh_i_values[i, j] = (
                Lgh_i[j].EvaluateIndeterminates(x, x_value)
                )
    # compute Lfh(x) and Lgh(x) at x_value:
    Lf_h_value = np.dot(lambda_i_values, Lfh_i_values)
    Lg_h_value = np.dot(lambda_i_values, Lgh_i_values)
    # compute desired control input:
    x_goal = np.array([0, 0])
    Kp = K_desired_control
    u_d = Kp * (x_goal - x_value)
    
    # For the CBF-QP:
    prog = solvers.MathematicalProgram()
    u = prog.NewContinuousVariables(2, "u")
    cost = (u - u_d).dot(u - u_d)
    prog.AddQuadraticCost(cost)
    prog.AddLinearConstraint(
        a=Lg_h_value,
        lb=-Lf_h_value - kappa_h*h_value,
        ub=np.inf,
        vars=u,
    )

    result = solvers.Solve(prog)
    assert result.is_success()
    u_val = result.GetSolution(u)
    return u_val
    
    
    

