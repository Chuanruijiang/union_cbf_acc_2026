import compatible_clf_union_cbf.union_cbf as mut
import numpy as np

import pydrake.symbolic as sym


def test_empty_subset_check():
    x = sym.MakeVectorContinuousVariable(2, "x")
    radiuses = np.array([
        0.1,
        0.2,
        0.49,
        0.5,
        0.51,
        0.6,
        1.0
        ])
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
        h = np.array([
            -((x - np.array([0.5, 0])).dot(x - np.array([0.5, 0])) - radiuses[i]**2),
            -((x - np.array([-0.5, 0])).dot(x - np.array([-0.5, 0])) - radiuses[i]**2),
        ])
        union_cbf = mut.UnionCBF(h=h, x=x)
        result = union_cbf.non_empty_disjoint_subsets()
        assert np.array_equal(result, expected_results[i])
