import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.clf_union_cbf as mut


def test_calc_xi_lambda():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(x[1]),
        sym.Polynomial(0)
    ])
    g = np.array([
        [sym.Polynomial(0)],
        [sym.Polynomial(1)]
    ])
    V = sym.Polynomial(x.dot(x))
    h = sym.Polynomial(1 - x.dot(x))
    range_non_negative = np.array([sym.Polynomial(5 - x.dot(x))])
    Au = np.array([
        [1], [-1]
    ])
    bu = np.array([1, 1])
    kappa_V = 1
    kappa_h = 1
    epsilon = 0.1
    test_object = mut.CompatibleClfCbfInRange(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        V=V,
        h=h,
        range_non_negative=range_non_negative,
        range_strictly_positive=None,
        state_eq_const=None,
        Au=Au,
        bu=bu
    )
    expected_lambda = np.array([
        [sym.Polynomial(2*x[1])],
        [sym.Polynomial(2*x[1])],
        [1],
        [-1]
    ])
    expected_xi = np.array([
        sym.Polynomial(-2*x[0]*x[1] + 1 - x.dot(x) - epsilon),
        sym.Polynomial(-2*x[0]*x[1] - x.dot(x) - epsilon),
        1 - epsilon,
        1 - epsilon
    ])
    xi, lambda_ = test_object._calc_xi_lambda(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        epsilon=epsilon
    )
    assert xi.shape == expected_xi.shape
    assert lambda_.shape == expected_lambda.shape
    for i in range(xi.shape[0]):
        if isinstance(xi[i], sym.Polynomial):
            assert xi[i].EqualTo(expected_xi[i])
        else:
            assert xi[i] == expected_xi[i]
    for i in range(lambda_.shape[0]):
        for j in range(lambda_.shape[1]):
            if isinstance(lambda_[i][j], sym.Polynomial):
                assert lambda_[i][j].EqualTo(expected_lambda[i][j])
            else:
                assert lambda_[i][j] == expected_lambda[i][j]
