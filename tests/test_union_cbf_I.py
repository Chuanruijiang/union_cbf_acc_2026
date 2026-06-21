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

def test_all_phi_functions_for_all_cbfs():
    # create a 2D double integrator system
    x = sym.MakeVectorContinuousVariable(4, "x")
    f = np.array([
        sym.Polynomial(x[2]),
        sym.Polynomial(x[3]),
        sym.Polynomial(0),
        sym.Polynomial(0)
    ])
    g = np.array([
        [sym.Polynomial(0), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(0)],
        [sym.Polynomial(1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(1)]
    ])
    # HOCBFs:
    switching_cbfs = np.array([
        sym.Polynomial(-x[0] + 1),
        sym.Polynomial( x[1] - 1),
    ])
    static_cbfs = np.array([
        sym.Polynomial(x[0] + x[1] - 1)
    ])
    relative_degree = [2, 2]
    alphas = [[0.01, 0.1], [0.01, 0.1]]
    # test object
    test_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        control_limits=None,
        relative_degree=relative_degree,
        alpha=alphas
    )
    (all_psi_switching, all_phi_static
     ) = test_obj._all_phi_polys_for_all_cbfs(
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs
    )
    expected_psi_switching = np.array([
        [switching_cbfs[0], sym.Polynomial(-1*x[2] + alphas[0][0] * switching_cbfs[0])],
        [switching_cbfs[1], sym.Polynomial( 1*x[3] + alphas[0][0] * switching_cbfs[1])]
    ])
    expected_phi_normal = [
        np.array([
            static_cbfs[0], 
            sym.Polynomial(x[2] + x[3] + alphas[1][0] * static_cbfs[0])
            ])
    ]
    check_polynomial_arrays_equal(
        p = all_psi_switching,
        q = expected_psi_switching,
        tol = 1e-8
    )
    for i in range(len(all_phi_static)):
        check_polynomial_arrays_equal(
            p = all_phi_static[i],
            q = expected_phi_normal[i],
            tol = 1e-8
        )

def test_all_possible_subsets():
    from itertools import product

    x = sym.MakeVectorContinuousVariable(3, "x")
    f = np.array(
        [
            sym.Polynomial(x[0] - x[1]**2),
            sym.Polynomial(x[1] + x[2]),
            sym.Polynomial(0)
        ]
    )
    g = np.array(
        [
            [sym.Polynomial(0)],
            [sym.Polynomial(0)],
            [sym.Polynomial(1)]
        ]
    )
    A = np.array(
        [
            [sym.Polynomial(1)],
            [sym.Polynomial(-1)]
        ]
    )
    c = np.array(
        [sym.Polynomial(2), sym.Polynomial(2)]
    )

    # test case 1:
    switching_cbfs = np.array(
        [
            sym.Polynomial(x[0] + 2),
            sym.Polynomial(-x[0] + 4)
        ]
    )
    static_cbfs = np.array(
        [
            sym.Polynomial(x[0] + x[1] + 3),
        ]
    )
    alpha = [
        [0.27, 0.71, 1.19],
        [0.43, 0.88],
    ]
    relative_degree = [3, 2]
    test_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        relative_degree=relative_degree,
        control_limits=(A, c),
    )

    all_subsets = test_obj.all_possible_subsets(
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
    )
    expected_switching_psi_polys = [
        np.array(
            [
                switching_cbfs[0],
                sym.Polynomial(
                    x[0] - x[1] ** 2 + alpha[0][0] * switching_cbfs[0]
                ),
                sym.Polynomial(
                    x[0]
                    - 3 * x[1] ** 2
                    - 2 * x[1] * x[2]
                    + (alpha[0][0] + alpha[0][1]) * (x[0] - x[1] ** 2)
                    + alpha[0][0] * alpha[0][1] * switching_cbfs[0]
                ),
            ],
            dtype=sym.Polynomial,
        ),
        np.array(
            [
                switching_cbfs[1],
                sym.Polynomial(
                    -x[0] + x[1] ** 2 + alpha[0][0] * switching_cbfs[1]
                ),
                sym.Polynomial(
                    -x[0]
                    + 3 * x[1] ** 2
                    + 2 * x[1] * x[2]
                    + (alpha[0][0] + alpha[0][1]) * (-x[0] + x[1] ** 2)
                    + alpha[0][0] * alpha[0][1] * switching_cbfs[1]
                ),
            ],
            dtype=sym.Polynomial,
        ),
    ]
    expected_static_phi_polys = [
        np.array(
            [
                static_cbfs[0],
                sym.Polynomial(
                    x[0]
                    - x[1] ** 2
                    + x[1]
                    + x[2]
                    + alpha[1][0] * static_cbfs[0]
                ),
            ],
            dtype=sym.Polynomial,
        ),
    ]

    assert len(all_subsets) == 3

    expected_masks = [
        np.array([1, 1]),
        np.array([1, 0]),
        np.array([0, 1]),
    ]
    expected_num_components = [1, 3, 3]
    for i in range(len(all_subsets)):
        assert isinstance(all_subsets[i], np.ndarray)
        assert all_subsets[i].shape == (expected_num_components[i],)

    subset_component = all_subsets[0][0]
    assert isinstance(subset_component, Subset)
    assert subset_component.num_avaliable_switching_cbfs == 2
    assert np.array_equal(
        subset_component.mask_avaliable_switching_cbfs,
        expected_masks[0],
    )
    assert len(subset_component.activated_poly_groups) == 3
    check_polynomial_arrays_equal(
        subset_component.activated_poly_groups[0],
        expected_switching_psi_polys[0],
        tol=1e-8,
    )
    check_polynomial_arrays_equal(
        subset_component.activated_poly_groups[1],
        expected_switching_psi_polys[1],
        tol=1e-8,
    )
    check_polynomial_arrays_equal(
        subset_component.activated_poly_groups[2],
        expected_static_phi_polys[0],
        tol=1e-8,
    )
    assert subset_component.deactivated_polys is None
    assert subset_component.subset_component_indicator is None

    for i in range(3):
        subset_component = all_subsets[1][i]
        assert isinstance(subset_component, Subset)
        assert subset_component.num_avaliable_switching_cbfs == 1
        assert np.array_equal(
            subset_component.mask_avaliable_switching_cbfs,
            expected_masks[1],
        )
        assert len(subset_component.activated_poly_groups) == 2
        check_polynomial_arrays_equal(
            subset_component.activated_poly_groups[0],
            expected_switching_psi_polys[0],
            tol=1e-8,
        )
        check_polynomial_arrays_equal(
            subset_component.activated_poly_groups[1],
            expected_static_phi_polys[0],
            tol=1e-8,
        )
        check_polynomial_arrays_equal(
            subset_component.deactivated_polys,
            np.array([expected_switching_psi_polys[1][i]], dtype=sym.Polynomial),
            tol=1e-8,
        )
        assert np.array_equal(subset_component.subset_component_indicator, np.array([i]))

    for i in range(3):
        subset_component = all_subsets[2][i]
        assert isinstance(subset_component, Subset)
        assert subset_component.num_avaliable_switching_cbfs == 1
        assert np.array_equal(
            subset_component.mask_avaliable_switching_cbfs,
            expected_masks[2],
        )
        assert len(subset_component.activated_poly_groups) == 2
        check_polynomial_arrays_equal(
            subset_component.activated_poly_groups[0],
            expected_switching_psi_polys[1],
            tol=1e-8,
        )
        check_polynomial_arrays_equal(
            subset_component.activated_poly_groups[1],
            expected_static_phi_polys[0],
            tol=1e-8,
        )
        check_polynomial_arrays_equal(
            subset_component.deactivated_polys,
            np.array([expected_switching_psi_polys[0][i]], dtype=sym.Polynomial),
            tol=1e-8,
        )
        assert np.array_equal(subset_component.subset_component_indicator, np.array([i]))

    # test case 2:
    switching_cbfs = np.array(
        [
            sym.Polynomial(x[0] + 2),
            sym.Polynomial(-x[0] + 4),
            sym.Polynomial(x[0] + 9)
        ]
    )
    static_cbfs = np.array(
        [
            sym.Polynomial(x[0] + x[1] + 3),
        ]
    )
    alpha = [
        [0.27, 0.71, 1.19],
        [0.43, 0.88],
    ]
    relative_degree = [3, 2]
    test_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        relative_degree=relative_degree,
        control_limits=(A, c),
    )
    all_subsets = test_obj.all_possible_subsets(
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
    )

    # check whether the amount of X_N is identical to total
    # possible number of N.
    assert len(all_subsets) == 7

    # For each X_N, check whether it has rb^|N| number of
    # X_N^(bar_i) subsets.
    for i in range(len(all_subsets)):
        X_N = all_subsets[i]
        num_deact = 3 - X_N[0].num_avaliable_switching_cbfs
        assert X_N.shape[0] == 3**(num_deact)

def test_get_non_empty_subsets():
    # case 1: switching CBF only
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
    test_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=[[0.1]],
        relative_degree=[1],
        control_limits=(A, c),
    )
    all_subsets = test_obj.all_possible_subsets(
        switching_cbfs=cbfs,
        static_cbfs=None,
    )
    non_empty_subsets = test_obj.get_non_empty_subsets(
        all_subsets=all_subsets
    )
    non_empty_masks = [
        subset.mask_avaliable_switching_cbfs for subset in non_empty_subsets
    ]
    assert len(non_empty_masks) == 2
    assert any(np.array_equal(mask, np.array([1, 0])) for mask in non_empty_masks)
    assert any(np.array_equal(mask, np.array([0, 1])) for mask in non_empty_masks)

    # case 2: with static CBFs
    # using the same single integrator and the control input limits.
    switching_cbfs = cbfs
    static_cbfs = np.array([
        sym.Polynomial(9 - x[0]**2 - x[1]**2)
    ])
    test_obj = UnionCbfI(
        x=x, f=f, g=g,
        control_limits=(A, c),
        relative_degree=[1, 1],
        alpha=[[0.1], [0.1]]
        )
    all_subsets = test_obj.all_possible_subsets(
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs
    )
    non_empty_subset_components = test_obj.get_non_empty_subsets(
        all_subsets=all_subsets
    )
    assert len(non_empty_subset_components) == 2
    subset_components_by_mask = {
        tuple(subset.mask_avaliable_switching_cbfs.tolist()): subset
        for subset in non_empty_subset_components
    }
    subset_01 = subset_components_by_mask[(0, 1)]
    subset_10 = subset_components_by_mask[(1, 0)]

    assert subset_01.num_avaliable_switching_cbfs == 1
    assert len(subset_01.activated_poly_groups) == 2
    assert subset_01.deactivated_polys.shape[0] == 1
    check_polynomial_arrays_equal(
        p=subset_01.activated_poly_groups[0],
        q=np.array([switching_cbfs[1]]),
        tol=1e-8
    )
    check_polynomial_arrays_equal(
        p=subset_01.activated_poly_groups[1],
        q=static_cbfs,
        tol=1e-8
    )
    assert subset_10.num_avaliable_switching_cbfs == 1
    assert len(subset_10.activated_poly_groups) == 2
    assert subset_10.deactivated_polys.shape[0] == 1
    check_polynomial_arrays_equal(
        p=subset_10.activated_poly_groups[0],
        q=np.array([switching_cbfs[0]]),
        tol=1e-8
    )
    check_polynomial_arrays_equal(
        p=subset_10.activated_poly_groups[1],
        q=static_cbfs,
        tol=1e-8
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
    test_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=[[alpha]],
        relative_degree=[1],
        control_limits=(A, c),
    )
    subset = Subset(
        x=x,
        activated_poly_groups=[
            np.array([cbfs[0]], dtype=sym.Polynomial),
            np.array([cbfs[1]], dtype=sym.Polynomial),
        ],
        deactivated_polys=None,
        equation_constraints=None,
        num_avaliable_switching_cbfs=2,
        mask_avaliable_switching_cbfs=np.array([1, 1]),
        subset_component_indicator=None,
    )
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
                alpha * cbfs[0] - eta,
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
                sym.Polynomial(1 - epsilon),
            ]
        ),
        np.array(
            [
                alpha * cbfs[1] - eta,
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
    test_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=[[alpha]],
        relative_degree=[1],
        control_limits=(A, c),
    )
    test_subset = Subset(
        x=x,
        activated_poly_groups=[
            np.array([cbfs[0]], dtype=sym.Polynomial),
            np.array([cbfs[1]], dtype=sym.Polynomial),
        ],
        deactivated_polys=None,
        equation_constraints=None,
        num_avaliable_switching_cbfs=2,
        mask_avaliable_switching_cbfs=np.array([1, 1]),
        subset_component_indicator=None,
    )
    subset_general_lagrangian_degree = SubsetGeneralLagrangianDegrees(
        num_control_inputs=test_obj.n_u,
        activated_lagrangian_x_degree=1,
        activated_lagrangian_y_degree=2,
        deactivated_lagrangian_x_degree=1,
        deactivated_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=3,
        lambda_lagrangian_y_degree=4,
        xi_lagrangian_x_degree=5,
        xi_lagrangian_y_degree=6,
        state_eq_lagrangian_x_degree=None,
        state_eq_lagrangian_y_degree=None,
    )
    (
        test_subset_lagrangian_degree 
    )= subset_general_lagrangian_degree.construct_lagrangian_degrees(
        subset=test_subset
    )
    
    assert isinstance(test_subset_lagrangian_degree, SubsetFeasibilityLagrangianDegrees)
    assert len(test_subset_lagrangian_degree.activated_poly_groups_degree) == cbfs.shape[0]
    assert test_subset_lagrangian_degree.deactivated_polys_degree is None
    assert len(test_subset_lagrangian_degree.lambda_y_degree) == 2
    assert len(test_subset_lagrangian_degree.xi_y_degree) == 2
    assert test_subset_lagrangian_degree.state_eq_degree is None
    for activated_group_degree in test_subset_lagrangian_degree.activated_poly_groups_degree:
        assert isinstance(activated_group_degree, list)
        assert len(activated_group_degree) == 1
        cbf_lagrangian_degree = activated_group_degree[0]
        assert isinstance(cbf_lagrangian_degree, Degree)
        assert cbf_lagrangian_degree.x == 1
        assert cbf_lagrangian_degree.y == 2
        assert cbf_lagrangian_degree.c == 0
    for lambda_y_degree in test_subset_lagrangian_degree.lambda_y_degree:
        assert isinstance(lambda_y_degree, list)
        assert len(lambda_y_degree) == test_obj.n_u
        for each_element in lambda_y_degree:
            assert isinstance(each_element, Degree)
            assert each_element.x == 3
            assert each_element.y == 4
            assert each_element.c == 0
    for xi_y_degree in test_subset_lagrangian_degree.xi_y_degree:
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
    test_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=[[alpha]],
        relative_degree=[1],
        control_limits=(A, c),
    )
    test_subset = Subset(
        x=x,
        activated_poly_groups=[
            np.array([cbfs[0]], dtype=sym.Polynomial),
            np.array([cbfs[1]], dtype=sym.Polynomial),
            np.array([cbfs[2]], dtype=sym.Polynomial),
        ],
        deactivated_polys=None,
        equation_constraints=None,
        num_avaliable_switching_cbfs=3,
        mask_avaliable_switching_cbfs=np.array([1, 1, 1]),
        subset_component_indicator=None,
    )
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

    test_subset = Subset(
        x=x,
        activated_poly_groups=[
            np.array([cbfs[0]], dtype=sym.Polynomial),
        ],
        deactivated_polys=np.array([cbfs[1], cbfs[2]], dtype=sym.Polynomial),
        equation_constraints=None,
        num_avaliable_switching_cbfs=1,
        mask_avaliable_switching_cbfs=np.array([1, 0, 0]),
        subset_component_indicator=np.array([0, 0]),
    )
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
        alpha=[[alpha]],
        relative_degree=[1],
        control_limits=(A, c)
    )
    subset = Subset(
        x=x,
        activated_poly_groups=[
            np.array([cbfs[0]], dtype=sym.Polynomial),
        ],
        deactivated_polys=np.array([cbfs[1], cbfs[2], cbfs[3]], dtype=sym.Polynomial),
        equation_constraints=None,
        num_avaliable_switching_cbfs=1,
        mask_avaliable_switching_cbfs=np.array([1, 0, 0, 0]),
        subset_component_indicator=np.array([0, 0, 0]),
    )
    general_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=test_obj.n_u,
        activated_lagrangian_x_degree=2,
        activated_lagrangian_y_degree=2,
        deactivated_lagrangian_x_degree=2,
        deactivated_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=2,
        lambda_lagrangian_y_degree=0,
        xi_lagrangian_x_degree=2,
        xi_lagrangian_y_degree=0,
        state_eq_lagrangian_x_degree=None,
        state_eq_lagrangian_y_degree=None,
    )
    (test_success, _) = test_obj.check_feasibility_in_subset(
        subset=subset,
        general_degrees=general_degrees,
        eta=eta,
        eps=epsilon,
    )

    assert test_success
