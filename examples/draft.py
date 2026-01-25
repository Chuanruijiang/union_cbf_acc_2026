import numpy as np
import pydrake.symbolic as sym
import union_cbf_base.utils as utils
from union_cbf_base.union_cbf_II import (
    Degree,
    UnionCbfII,
    CbfFeasibilityLagrangianDegrees,
)
from union_cbf_base.non_empty_subset import Subset

def test_check_single_cbf_feasibility():
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
    eta = 1e-3
    epsilon = 1e-3
    test_obj = UnionCbfII(x=x, f=f, g=g, alpha=alpha, control_limits=(A, c))
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        cbf=Degree(x=2, y=2, c=0),
        lambda_y=[
            Degree(x=2, y=0, c=0),
            Degree(x=2, y=0, c=0),
        ],
        xi_y=Degree(x=2, y=0, c=0),
    )
    feasible = test_obj.check_cbf_feasibility(
        cbf=cbfs[0],
        lagrangian_degrees=lagrangian_degrees,
        eta=eta,
        eps=epsilon
    )
    assert feasible is True

if __name__ == "__main__":
    test_check_single_cbf_feasibility()