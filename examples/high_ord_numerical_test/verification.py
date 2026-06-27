"""
This script verifies both the sufficient condition I and II
for the high-order system. The test order ranges from 1 to 4
and record the verification time.
This secript generates the verification time table I
in the journal extension.
"""

import numpy as np
import pydrake.symbolic as sym

import union_cbf_base.verification_base as verification
from union_cbf_base.union_cbf_I import (
    SubsetGeneralLagrangianDegrees,
)
from union_cbf_base.union_cbf_II import (
    CbfFeasibilityLagrangianDegrees,
    Degree as DegreeII
)
from examples.high_ord_numerical_test.dynamics import HighOrdSystem

def verify_sufficient_condition_I(verify_order: int):
    print(f"Verifying sufficient condition I for order {verify_order} system...")
    x = sym.MakeVectorContinuousVariable(2*verify_order, "x")
    high_ord = HighOrdSystem(order=verify_order)
    f, g = high_ord.affine_dynamics(x)
    (A, c) = high_ord.control_limits()
    num_control = A.shape[1]

    switching_cbfs = np.array([
        sym.Polynomial(x[0] + x[verify_order]),
        sym.Polynomial(x[0] - x[verify_order])
            ])
    
    relative_degree = [verify_order]

    alphas = [
        [0.001**(verify_order - i) for i in range(verify_order)],
        ]
    
    lagrangian_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=num_control,
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
    verification_flag = verification.verify_sufficient_condition_I(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=switching_cbfs,
        static_cbfs=None,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees,
        show_output=True,
        show_computation_time=True
    )
    assert verification_flag == True

def verify_sufficient_condition_II(verify_order: int):
    print(f"Verifying sufficient condition II for order {verify_order} system...")
    x = sym.MakeVectorContinuousVariable(2*verify_order, "x")
    high_ord = HighOrdSystem(order=verify_order)
    f, g = high_ord.affine_dynamics(x)
    (A, c) = high_ord.control_limits()
    num_control = A.shape[1]

    switching_cbfs = np.array([
        sym.Polynomial(x[0] + x[verify_order]),
        sym.Polynomial(x[0] - x[verify_order])
            ])
    
    relative_degree = [verify_order]
    alphas = [
        [0.01**(verify_order - i) for i in range(verify_order)],
        ]
    
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [
            [DegreeII(x=2, y=2, c=0)] * relative_degree[0]
        ],
        lambda_y=[DegreeII(x=2, y=0, c=0)]*num_control,
        xi_y=DegreeII(x=2, y=0, c=0),
        state_eq=None,
    )
    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=switching_cbfs,
        static_cbfs=None,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees,
        show_output=True,
        show_computation_time=True
    )
    assert verification_flag == True

if __name__ == "__main__":
    # verfication time recording for Verif-I:
    verify_sufficient_condition_I(verify_order=1)
    verify_sufficient_condition_I(verify_order=2)
    verify_sufficient_condition_I(verify_order=3)
    verify_sufficient_condition_I(verify_order=4) # terminal is killed in this case

    # verification time recording for Verif-II:
    verify_sufficient_condition_II(verify_order=1)
    verify_sufficient_condition_II(verify_order=2)
    verify_sufficient_condition_II(verify_order=3)
    verify_sufficient_condition_II(verify_order=4)
    verify_sufficient_condition_II(verify_order=5)
    verify_sufficient_condition_II(verify_order=6)
    verify_sufficient_condition_II(verify_order=7)
    verify_sufficient_condition_II(verify_order=8) # terminal is killed in this case