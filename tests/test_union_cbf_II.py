import numpy as np
import pydrake.symbolic as sym

from union_cbf_base.utils import(
    Degree,
    check_polynomial_arrays_equal
)
from union_cbf_base.union_cbf_II import(
    CbfFeasibilityLagrangianDegrees,
    UnionCbfII
)

def test_compute_xi_lambda():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([sym.Polynomial(0), sym.Polynomial(0)])
    g = np.array(
        [[sym.Polynomial(1), sym.Polynomial(0)], [sym.Polynomial(0), sym.Polynomial(1)]]
    )
    A = np.array(
        [
            [sym.Polynomial(1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(1)],
            [sym.Polynomial(-1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(-1)],
        ]
    )
    c = np.array(
        [sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1)]
    )
    cbfs = np.array(
        [
            sym.Polynomial((0.6) ** 2 - (x[0] - 0.5) ** 2 - (x[1] - 0) ** 2),
            sym.Polynomial((0.6) ** 2 - (x[0] + 0.5) ** 2 - (x[1] - 0) ** 2),
        ]
    )
    alpha = 0.1
    eta = 0.1
    epsilon = 0.01
    test_obj = UnionCbfII(x=x, f=f, g=g, alpha=alpha, control_limits=(A, c))
    # test case 1: first cbf
    expected_lambda = np.array([
        [sym.Polynomial(2*(x[0]-0.5)), sym.Polynomial(2*(x[1]-0))],
        [sym.Polynomial(1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(1)],
        [sym.Polynomial(-1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(-1)],
    ])
    expected_xi = np.array([
        sym.Polynomial(-2*(x[0]-0.5)*f[0] - 2*(x[1]-0)*f[1] 
                       + alpha*((0.6)**2 - (x[0]-0.5)**2 - (x[1]-0)**2) 
                       - eta),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
    ])
    computed_lambda, computed_xi = test_obj._xi_lambda(
        alpha=alpha,
        cbf=cbfs[0],
        eta=eta,
        eps=epsilon,
    )
    check_polynomial_arrays_equal(computed_lambda, expected_lambda, tol=1e-8)
    check_polynomial_arrays_equal(computed_xi, expected_xi, tol=1e-8)

    # test case 2: second cbf
    expected_lambda = np.array([
        [sym.Polynomial(2*(x[0]+0.5)), sym.Polynomial(2*(x[1]-0))],
        [sym.Polynomial(1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(1)],
        [sym.Polynomial(-1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(-1)],
    ])
    expected_xi = np.array([
        sym.Polynomial(-2*(x[0]+0.5)*f[0] - 2*(x[1]-0)*f[1] 
                       + alpha*((0.6)**2 - (x[0]+0.5)**2 - (x[1]-0)**2) 
                       - eta),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
    ])
    computed_lambda, computed_xi = test_obj._xi_lambda(
        alpha=alpha,
        cbf=cbfs[1],
        eta=eta,
        eps=epsilon,
    )
    check_polynomial_arrays_equal(computed_lambda, expected_lambda, tol=1e-8)
    check_polynomial_arrays_equal(computed_xi, expected_xi, tol=1e-8)

def test_construct_x_y_sets():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([sym.Polynomial(0), sym.Polynomial(0)])
    g = np.array(
        [[sym.Polynomial(1), sym.Polynomial(0)], [sym.Polynomial(0), sym.Polynomial(1)]]
    )
    A = np.array(
        [
            [sym.Polynomial(1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(1)],
            [sym.Polynomial(-1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(-1)],
        ]
    )
    c = np.array(
        [sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1)]
    )
    test_obj = UnionCbfII(x=x, f=f, g=g, alpha=0.1, control_limits=(A, c))
    (
        x_set, y_set, xy_set, y_squared_polys
    ) = test_obj._construct_x_y_sets()
    assert x_set.size() == 2
    assert y_set.size() == 5
    assert xy_set.size() == 7
    assert y_squared_polys.shape[0] == 5

def test_check_single_cbf_feasibility():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([sym.Polynomial(0), sym.Polynomial(0)])
    g = np.array(
        [[sym.Polynomial(1), sym.Polynomial(0)], [sym.Polynomial(0), sym.Polynomial(1)]]
    )
    A = np.array(
        [
            [sym.Polynomial(1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(1)],
            [sym.Polynomial(-1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(-1)],
        ]
    )
    c = np.array(
        [sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1)]
    )
    cbfs = np.array(
        [
            sym.Polynomial((0.6) ** 2 - (x[0] - 0.5) ** 2 - (x[1] - 0) ** 2),
            sym.Polynomial((0.6) ** 2 - (x[0] + 0.5) ** 2 - (x[1] - 0) ** 2),
        ]
    )
    alpha = 0.1
    eta = 1e-3
    epsilon = 1e-3
    test_obj = UnionCbfII(x=x, f=f, g=g, alpha=alpha, control_limits=(A, c))
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        cbf=Degree(x=2, y=2, c=0),
        lambda_y=[
            Degree(x=2, y=0, c=0),
            Degree(x=2, y=0, c=0),
        ],
        xi_y=Degree(x=2, y=0, c=0),
    )
    (feasible,_) = test_obj.check_cbf_feasibility(
        cbf=cbfs[0],
        lagrangian_degrees=lagrangian_degrees,
        eta=eta,
        eps=epsilon
    )
    assert feasible is True

