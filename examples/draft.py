import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.union_cbf as mut


def main():
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


if __name__ == "__main__":
    main()
