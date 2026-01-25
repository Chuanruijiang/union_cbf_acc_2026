import numpy as np
import pydrake.symbolic as sym
import union_cbf_base.utils as utils
from union_cbf_base.union_cbf_I import (
    UnionCbfI,
    SubsetGeneralLagrangianDegrees,
)
from union_cbf_base.non_empty_subset import Subset

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

if __name__ == "__main__":
    test_check_feasibility_in_subset()