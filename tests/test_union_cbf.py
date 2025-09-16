from union_cbf_base.union_cbf import (
    UnionCbf,
    SubsetFeasibilityLagrangian,
    SubsetFeasibilityLagrangianDegrees
)
from union_cbf_base.non_empty_subset import(
    Subset
)
from union_cbf_base.utils import(
    Degree,
    check_polynomial_arrays_equal
)
import numpy as np

import pydrake.symbolic as sym


def test_get_non_empty_subsets():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([sym.Polynomial(), sym.Polynomial()])
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
            sym.Polynomial((0.4) ** 2 - (x[0] - 0.5) ** 2 - (x[1] - 0) ** 2),
            sym.Polynomial((0.4) ** 2 - (x[0] + 0.5) ** 2 - (x[1] - 0) ** 2),
        ]
    )
    test_obj = UnionCbf(x=x, f=f, g=g, cbfs=cbfs, alpha=0.1, control_limits=(A, c))
    activation_masks = test_obj.all_possible_subsets()
    non_empty_subsets = test_obj.get_non_empty_subsets(
        all_subsets_mask=activation_masks
    )
    assert (non_empty_subsets[0].activation_index == np.array([0, 1])).all()
    assert (non_empty_subsets[1].activation_index == np.array([1, 0])).all()

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
    test_obj = UnionCbf(x=x, f=f, g=g, cbfs=cbfs, alpha=alpha, control_limits=(A, c))
    subset = Subset(x=x, cbfs=cbfs, activation_index=np.array([1, 1]))
    (lambda_list, xi_list) = test_obj._lambda_xi(
        subset=subset,
        eta=eta,
        epsilon=epsilon,
    )

    expected_lambda_list = [
        np.array(
            [
                [sym.Polynomial(2 * x[0] - 1), sym.Polynomial(2 * x[1])],
                [sym.Polynomial(1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(1)],
                [sym.Polynomial(-1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(-1)],
            ]
        ),
        np.array(
            [
                [sym.Polynomial(2 * x[0] + 1), sym.Polynomial(2 * x[1])],
                [sym.Polynomial(1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(1)],
                [sym.Polynomial(-1), sym.Polynomial(0)],
                [sym.Polynomial(0), sym.Polynomial(-1)],
            ]
        ),
    ]
    expected_xi_list = [
        np.array(
            [
                test_obj.alpha * cbfs[0] - eta,
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
            ]
        ),
        np.array(
            [
                test_obj.alpha * cbfs[1] - eta,
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
            ]
        ),
    ]
    assert len(lambda_list) == len(expected_lambda_list)
    assert len(xi_list) == len(expected_xi_list)
    assert len(lambda_list) == len(xi_list)
    for i in range(len(lambda_list)):
        check_polynomial_arrays_equal(
            lambda_list[i], expected_lambda_list[i], tol=1e-8
        )
        check_polynomial_arrays_equal(xi_list[i], expected_xi_list[i], tol=1e-8)

def test_construct_lagrangian_degrees():
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
    test_obj = UnionCbf(x=x, f=f, g=g, cbfs=cbfs, alpha=alpha, control_limits=(A, c))
    test_subset = Subset(x=x, cbfs=cbfs, activation_index=np.array([1, 1]))
    test_subset_lagrangian_degree = test_obj._construct_lagrangian_degrees(
        subset=test_subset,
        cbf_lagrangian_x_degree=1,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=3,
        lambda_y_lagrangian_y_degree=4,
        xi_y_lagrangian_x_degree=5,
        xi_y_lagrangian_y_degree=6
    )
    assert isinstance(test_subset_lagrangian_degree, SubsetFeasibilityLagrangianDegrees)
    assert len(test_subset_lagrangian_degree.s0) == cbfs.shape[0]
    assert len(test_subset_lagrangian_degree.s_lambda_y) == 2
    assert len(test_subset_lagrangian_degree.q_xi_y) == 2
    for each_s0 in test_subset_lagrangian_degree.s0:
        assert isinstance(each_s0, Degree)
        assert each_s0.x == 1
        assert each_s0.y == 2
        assert each_s0.c == 0
    for each_s_lambda_y in test_subset_lagrangian_degree.s_lambda_y:
        assert isinstance(each_s_lambda_y, list)
        assert len(each_s_lambda_y) == test_obj.n_u
        for each_element in each_s_lambda_y:
            assert isinstance(each_element, Degree)
            assert each_element.x == 3
            assert each_element.y == 4
            assert each_element.c == 0
    for each_q_xi_y in test_subset_lagrangian_degree.q_xi_y:
        assert isinstance(each_q_xi_y, Degree)
        assert each_q_xi_y.x == 5
        assert each_q_xi_y.y == 6
        assert each_q_xi_y.c == 0

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
    cbfs = np.array(
        [
            sym.Polynomial((0.6) ** 2 - (x[0] - 0.5) ** 2 - (x[1] - 0) ** 2),
            sym.Polynomial((0.6) ** 2 - (x[0] + 0.5) ** 2 - (x[1] - 0) ** 2),
            sym.Polynomial((0.6) ** 2 - (x[0] - 0) ** 2 - (x[1] - 0.5) ** 2),
        ]
    )
    alpha = 0.1
    test_obj = UnionCbf(x=x, f=f, g=g, cbfs=cbfs, alpha=alpha, control_limits=(A, c))
    test_subset = Subset(x=x, cbfs=cbfs, activation_index=np.array([1, 1, 1]))
    num_activated = 3
    test_xy_sets = test_obj._construct_x_y_sets(subset=test_subset)
    assert len(test_xy_sets) == 4

    assert isinstance(test_xy_sets[0], sym.Variables)
    assert len(test_xy_sets[0]) == test_obj.n_x + (
        (test_obj.control_limits[0].shape[0])+1) * num_activated
    
    assert isinstance(test_xy_sets[1], sym.Variables)
    assert len(test_xy_sets[1]) == test_obj.n_x

    assert isinstance(test_xy_sets[2], list)
    assert len(test_xy_sets[2]) == num_activated + 1
    for i in range(num_activated):
        assert isinstance(test_xy_sets[2][i], sym.Variables)
        assert len(test_xy_sets[2][i]) == (
            (test_obj.control_limits[0].shape[0])+1
            ) * (num_activated - 1)
    assert isinstance(test_xy_sets[2][-1], sym.Variables)
    assert len(test_xy_sets[2][-1]) == (
        (test_obj.control_limits[0].shape[0])+1
        ) * num_activated
    
    assert isinstance(test_xy_sets[3], list)
    assert len(test_xy_sets[3]) == num_activated
    for each_y_squared_poly in test_xy_sets[3]:
        assert isinstance(each_y_squared_poly, np.ndarray)
        assert each_y_squared_poly.shape == ((test_obj.control_limits[0].shape[0])+1,)
        for each_element in each_y_squared_poly:
            assert isinstance(each_element, sym.Polynomial)

    test_subset = Subset(x=x, cbfs=cbfs, activation_index=np.array([1, 0, 0]))
    num_activated = 1
    test_xy_sets = test_obj._construct_x_y_sets(subset=test_subset)
    assert isinstance(test_xy_sets[2], list)
    assert len(test_xy_sets[2]) == num_activated + 1
    assert test_xy_sets[2][0] is None
    assert isinstance(test_xy_sets[2][-1], sym.Variables)
    assert len(test_xy_sets[2][-1]) == (
        (test_obj.control_limits[0].shape[0])+1
        ) * num_activated

def test_check_feasibility_in_subset():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(0), sym.Polynomial(0)
    ])
    g = np.array([
        [sym.Polynomial(1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(1)]
    ])
    A = np.array([
        [sym.Polynomial(1), sym.Polynomial(0)], 
        [sym.Polynomial(0), sym.Polynomial(1)], 
        [sym.Polynomial(-1), sym.Polynomial(0)], 
        [sym.Polynomial(0), sym.Polynomial(-1)]
    ])
    c = np.array([
        sym.Polynomial(1), 
        sym.Polynomial(1), 
        sym.Polynomial(1), 
        sym.Polynomial(1)
    ])
    cbfs = np.array([
        sym.Polynomial(2**2 - (x[0] + 3)**2 - (x[1] - 0)**2),
        sym.Polynomial(2**2 - (x[0] - 0)**2 - (x[1] - 0)**2),
        sym.Polynomial(2**2 - (x[0] - 3)**2 - (x[1] - 0)**2),
        sym.Polynomial(2**2 - (x[0] - 6)**2 - (x[1] - 0)**2)
    ])
    alpha = 0.1
    eta = 1e-2
    epsilon = 0.01
    test_obj = UnionCbf(
        x = x,
        f = f,
        g = g,
        cbfs = cbfs,
        alpha=alpha,
        control_limits=(A, c)
    )
    subset = Subset(
        x=x,
        cbfs=cbfs,
        activation_index=np.array([1, 0, 0, 0])
    )
    test_success = test_obj.check_feasibility_in_subset(
        subset=subset,
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=eta,
        epsilon=epsilon,
    )

    assert test_success

def test_check_simplified_feasibility():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(0), sym.Polynomial(0)
    ])
    g = np.array([
        [sym.Polynomial(1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(1)]
    ])
    A = np.array([
        [sym.Polynomial(1), sym.Polynomial(0)], 
        [sym.Polynomial(0), sym.Polynomial(1)], 
        [sym.Polynomial(-1), sym.Polynomial(0)], 
        [sym.Polynomial(0), sym.Polynomial(-1)]
    ])
    c = np.array([
        sym.Polynomial(1), 
        sym.Polynomial(1), 
        sym.Polynomial(1), 
        sym.Polynomial(1)
    ])
    cbfs = np.array([
        sym.Polynomial((0.6)**2 - (x[0]-0.5)**2 - (x[1] - 0)**2),
        sym.Polynomial((0.6)**2 - (x[0]+0.5)**2 - (x[1] - 0)**2)
    ])
    alpha = 0.1
    eta = 1e-2
    epsilon = 0.01
    test_obj = UnionCbf(
        x = x,
        f = f,
        g = g,
        cbfs = cbfs,
        alpha=alpha,
        control_limits=(A, c)
    )
    test_success = test_obj.check_simplified_feasibility(
        cbf_index=0,
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=2,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=2,
        eta=eta,
        epsilon=epsilon,
    )

    assert test_success

