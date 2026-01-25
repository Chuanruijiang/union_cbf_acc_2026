import numpy as np
import pydrake.symbolic as sym

from union_cbf_base.union_cbf_I import (
    UnionCbfI,
    SubsetGeneralLagrangianDegrees,
    SubsetFeasibilityLagrangianDegrees
)
from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.utils import(
    Degree,
    check_polynomial_arrays_equal
)


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
    test_obj = UnionCbfI(x=x, f=f, g=g, alpha=0.1, control_limits=(A, c))
    all_subsets = test_obj.all_possible_subsets(cbfs=cbfs)
    non_empty_subsets = test_obj.get_non_empty_subsets(
        all_subsets=all_subsets
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
    test_obj = UnionCbfI(x=x, f=f, g=g, alpha=alpha, control_limits=(A, c))
    subset = Subset(x=x, all_polys=cbfs, activation_index=np.array([1, 1]))
    (lambda_list, xi_list) = test_obj._lambda_xi(
        subset=subset,
        eta=eta,
        eps=epsilon,
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
    test_obj = UnionCbfI(x=x, f=f, g=g, alpha=alpha, control_limits=(A, c))
    test_subset = Subset(x=x, all_polys=cbfs, activation_index=np.array([1, 1]))
    subset_general_lagrangian_degree = SubsetGeneralLagrangianDegrees(
        num_control_inputs=test_obj.n_u,
        cbfs_lagrangian_x_degree=1,
        cbfs_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=3,
        lambda_lagrangian_y_degree=4,
        xi_lagrangian_x_degree=5,
        xi_lagrangian_y_degree=6
    )
    (
        test_subset_lagrangian_degree 
    )= subset_general_lagrangian_degree.construct_lagrangian_degrees(
        subset=test_subset
    )
    
    assert isinstance(test_subset_lagrangian_degree, SubsetFeasibilityLagrangianDegrees)
    assert len(test_subset_lagrangian_degree.cbfs) == cbfs.shape[0]
    assert len(test_subset_lagrangian_degree.lambda_y) == 2
    assert len(test_subset_lagrangian_degree.xi_y) == 2
    for cbf_lagrangian_degree in test_subset_lagrangian_degree.cbfs:
        assert isinstance(cbf_lagrangian_degree, Degree)
        assert cbf_lagrangian_degree.x == 1
        assert cbf_lagrangian_degree.y == 2
        assert cbf_lagrangian_degree.c == 0
    for lambda_y_degree in test_subset_lagrangian_degree.lambda_y:
        assert isinstance(lambda_y_degree, list)
        assert len(lambda_y_degree) == test_obj.n_u
        for each_element in lambda_y_degree:
            assert isinstance(each_element, Degree)
            assert each_element.x == 3
            assert each_element.y == 4
            assert each_element.c == 0
    for xi_y_degree in test_subset_lagrangian_degree.xi_y:
        assert isinstance(xi_y_degree, Degree)
        assert xi_y_degree.x == 5
        assert xi_y_degree.y == 6
        assert xi_y_degree.c == 0

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
    test_obj = UnionCbfI(x=x, f=f, g=g, alpha=alpha, control_limits=(A, c))
    test_subset = Subset(x=x, all_polys=cbfs, activation_index=np.array([1, 1, 1]))
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

    test_subset = Subset(x=x, all_polys=cbfs, activation_index=np.array([1, 0, 0]))
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
    test_obj = UnionCbfI(
        x = x,
        f = f,
        g = g,
        alpha=alpha,
        control_limits=(A, c)
    )
    subset = Subset(
        x=x,
        all_polys=cbfs,
        activation_index=np.array([1, 0, 0, 0])
    )
    general_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=test_obj.n_u,
        cbfs_lagrangian_x_degree=2,
        cbfs_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=2,
        lambda_lagrangian_y_degree=0,
        xi_lagrangian_x_degree=2,
        xi_lagrangian_y_degree=0,
    )
    test_success = test_obj.check_feasibility_in_subset(
        subset=subset,
        lagrangian_degrees=general_degrees,
        eta=eta,
        eps=epsilon,
    )

    assert test_success

