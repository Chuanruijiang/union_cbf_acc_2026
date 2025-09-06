from union_cbf_base.union_cbf import UnionCbf
from union_cbf_base.non_empty_subset import Subset
import union_cbf_base.utils as utils
import pydrake.symbolic as sym
import numpy as np

def main():
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
    eta = 0.1
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
        activation_index=np.array([1, 1])
    )
    (lambda_list, xi_list) = test_obj._lambda_xi(
        subset=subset,
        eta=eta,
        epsilon=epsilon,
    )

    expected_lambda_list = [
        np.array([
            [sym.Polynomial(2*x[0] - 1), sym.Polynomial(2*x[1])],
            [sym.Polynomial(1), sym.Polynomial(0)], 
            [sym.Polynomial(0), sym.Polynomial(1)], 
            [sym.Polynomial(-1), sym.Polynomial(0)], 
            [sym.Polynomial(0), sym.Polynomial(-1)]
        ]),
        np.array([
            [sym.Polynomial(2*x[0] + 1), sym.Polynomial(2*x[1])],
            [sym.Polynomial(1), sym.Polynomial(0)], 
            [sym.Polynomial(0), sym.Polynomial(1)], 
            [sym.Polynomial(-1), sym.Polynomial(0)], 
            [sym.Polynomial(0), sym.Polynomial(-1)]
        ]),
    ]
    expected_xi_list = [
        np.array([
            test_obj.alpha * cbfs[0] - eta,
            sym.Polynomial(1-epsilon), 
            sym.Polynomial(1-epsilon), 
            sym.Polynomial(1-epsilon), 
            sym.Polynomial(1-epsilon)
        ]),
        np.array([
            test_obj.alpha * cbfs[1] - eta,
            sym.Polynomial(1-epsilon), 
            sym.Polynomial(1-epsilon), 
            sym.Polynomial(1-epsilon), 
            sym.Polynomial(1-epsilon)
        ]),
    ]
    assert len(lambda_list) == len(expected_lambda_list)
    assert len(xi_list) == len(expected_xi_list)
    assert len(lambda_list) == len(xi_list)
    for i in range(len(lambda_list)):
        utils.check_polynomial_arrays_equal(lambda_list[i], expected_lambda_list[i], tol=1e-8)
        utils.check_polynomial_arrays_equal(xi_list[i], expected_xi_list[i], tol=1e-8)


if __name__ == "__main__":
    main()
