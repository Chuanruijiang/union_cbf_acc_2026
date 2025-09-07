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
    test_success = test_obj.check_feasibility_in_subset(
        subset=subset,
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


if __name__ == "__main__":
    main()
