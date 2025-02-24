import compatible_clf_union_cbf.union_cbf as mut
import numpy as np
from compatible_clf_union_cbf.clf_cbf import XYDegree

import pydrake.symbolic as sym


def test_subset_empty_check():
    x = sym.MakeVectorContinuousVariable(2, "x")
    r = np.array([0.1, 0.3, 0.49, 0.5, 0.51, 0.6])
    expected_results = [True, True, True, False, False, False]
    for i in range(r.shape[0]):
        h = np.array(
            [
                sym.Polynomial(-((x[0] - 0.5) ** 2 + x[1] ** 2 - r[i] ** 2)),
                sym.Polynomial(-((x[0] + 0.5) ** 2 + x[1] ** 2 - r[i] ** 2)),
            ]
        )
        subset = mut.UnionSubset(activated=h, deactivated=None, variables=x)

        emptiness_lagragian_degrees = mut.EmptinessLagrangianDegrees(
            activated=[XYDegree(x=2, y=0), XYDegree(x=2, y=0)], deactivated=None
        )
        result = subset.is_empty(
            emptiness_lagrangian_degrees=emptiness_lagragian_degrees
        )
        assert result == expected_results[i]


def test_getting_subset_from_01_vector():
    x = sym.MakeVectorContinuousVariable(2, "x")
    h = np.array(
        [
            sym.Polynomial(-((x[0] - 0.5) ** 2 + x[1] ** 2 - 0.5**2)),
            sym.Polynomial(-((x[0] + 0.5) ** 2 + x[1] ** 2 - 0.5**2)),
        ]
    )
    input = np.array([[0, 1], [1, 0], [1, 1]])
    expected_activate_results = [
        np.array([h[1]]),
        np.array([h[0]]),
        np.array([h[0], h[1]]),
    ]
    expected_deactivate_results = [
        np.array([h[0]]),
        np.array([h[1]]),
        None,
    ]
    case = mut.ActivationIndicator(vecotor01=input)
    results = case.to_union_subsets(h=h, variables=x)
    for i in range(input.shape[0]):
        for j in range(results[i].activated.shape[0]):
            assert results[i].activated[j].EqualTo(expected_activate_results[i][j])
        if expected_deactivate_results[i] is not None:
            for j in range(results[i].deactivated.shape[0]):
                assert (
                    results[i].deactivated[j].EqualTo(expected_deactivate_results[i][j])
                )


def test_empty_subset_check():
    x = sym.MakeVectorContinuousVariable(2, "x")
    radiuses = np.array([0.1, 0.2, 0.49, 0.5, 0.51, 0.6, 1.0])
    expected_results = [
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [1, 0], [1, 1]]),
        np.array([[0, 1], [1, 0], [1, 1]]),
        np.array([[0, 1], [1, 0], [1, 1]]),
        np.array([[0, 1], [1, 0], [1, 1]]),
    ]
    for i in range(radiuses.shape[0]):
        h = np.array(
            [
                -(
                    (x - np.array([0.5, 0])).dot(x - np.array([0.5, 0]))
                    - radiuses[i] ** 2
                ),
                -(
                    (x - np.array([-0.5, 0])).dot(x - np.array([-0.5, 0]))
                    - radiuses[i] ** 2
                ),
            ]
        )
        union_cbf = mut.UnionCBF(h=h, x=x)
        result = union_cbf.non_empty_disjoint_subsets()
        assert np.array_equal(result, expected_results[i])

def test_cbf_constraint_initialization():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(x[1]),
        -sym.Polynomial(0),
    ])
    g = np.array([
        [0],
        [1]
    ])
    # case 1:
    h = sym.Polynomial(1 - x[0] - x[1])
    kappa_h = 0.1
    cbf_const = mut.CbfConstraint(
        h=h,
        f=f,
        g=g,
        x=x,
        kappa=kappa_h,
        relative_degree=None
    )
    expected_lhs_coeff = np.array([sym.Polynomial(-1)])
    expected_rhs = sym.Polynomial(-(-x[1]+kappa_h*h))
    assert isinstance(cbf_const.lhs_coeff, np.ndarray)
    assert cbf_const.lhs_coeff.shape[0] == expected_lhs_coeff.shape[0]
    for i in range(expected_lhs_coeff.shape[0]):
        assert cbf_const.lhs_coeff[i].EqualTo(expected_lhs_coeff[i])
    assert cbf_const.rhs.EqualTo(expected_rhs)

    # case 2:
    h = sym.Polynomial(1 - x[0])
    kappa_h = [0.1, 0.2]
    relative_degree = 2
    cbf_const = mut.CbfConstraint(
        h=h,
        f=f,
        g=g,
        x=x,
        kappa=kappa_h,
        relative_degree=relative_degree
    )
    expected_rhs = sym.Polynomial(-((kappa_h[0]+kappa_h[1])*(-x[1]) + (kappa_h[0]*kappa_h[1])*h))
    expected_lhs_coeff = np.array([sym.Polynomial(-1)])
    assert isinstance(cbf_const.lhs_coeff, np.ndarray)
    assert cbf_const.lhs_coeff.shape[0] == expected_lhs_coeff.shape[0]
    for i in range(expected_lhs_coeff.shape[0]):
        assert cbf_const.lhs_coeff[i].EqualTo(expected_lhs_coeff[i])
    assert cbf_const.rhs.EqualTo(expected_rhs)