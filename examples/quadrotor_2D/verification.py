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
from dynamics import QuadrotorDynamics2D

def verify_quadrotor_sufficient_I():
    # This verification may take about an hour to finish
    x = sym.MakeVectorContinuousVariable(7, "x")
    quadrotor = QuadrotorDynamics2D()
    f, g = quadrotor.trig_poly_affine_dynamics(x)
    eq_constraint = quadrotor.equation_constraint(x)
    (A, c) = quadrotor.control_limits()
    num_control = A.shape[1]

    # normal HOCBF
    static_cbfs = np.array([
        sym.Polynomial(
            x[3] + (1 - np.cos(np.pi / 4)) - (x[6])**2 + (np.pi/2)**2
        ),
        ])
    # switching HOCBF
    switching_cbfs = np.array([
        sym.Polynomial(x[1]),
        sym.Polynomial(x[0])
            ])
    
    relative_degree = [1, 2]
    alphas = [
        [1.0],
        [0.001, 0.01],
        ]
    # clear the error of union cbf I verification.
    
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
        state_eq_lagrangian_x_degree=2,
        state_eq_lagrangian_y_degree=2,
    )
    verification_flag = verification.verify_sufficient_condition_I(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=eq_constraint,
        lagrangian_degrees=lagrangian_degrees,
        show_output=True
    )
    assert verification_flag == True
    

def verify_quadrotor_sufficient_II():
    # this verification takes about 5 hours to finish
    x = sym.MakeVectorContinuousVariable(7, "x")
    quadrotor = QuadrotorDynamics2D()
    f, g = quadrotor.trig_poly_affine_dynamics(x)
    eq_constraint = quadrotor.equation_constraint(x)
    (A, c) = quadrotor.control_limits()
    num_control = A.shape[1]
    
    static_cbfs = np.array([
        # sym.Polynomial(x[1]),
        sym.Polynomial(
            x[3] + (1 - np.cos(np.pi / 3)) - (x[6])**2 + (np.pi/2)**2
        ),
    ])
    switching_cbfs = np.array([
        sym.Polynomial(-x[0] - x[1] - 15),
        sym.Polynomial(-x[1] - 3),
        sym.Polynomial(x[0] - x[1] + 3),
    ])
    
    relative_degree = [1, 2]
    alphas = [
        [1.0],
        [0.1, 1.0]
    ]
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [
            [DegreeII(x=2, y=2, c=0)] * relative_degree[0],
            [DegreeII(x=2, y=2, c=0)] * relative_degree[1],
        ],
        lambda_y=[DegreeII(x=2, y=0, c=0)]*num_control,
        xi_y=DegreeII(x=2, y=0, c=0),
        state_eq=[DegreeII(x=2, y=2, c=0)],
    )
    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=None,
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=eq_constraint,
        lagrangian_degrees=lagrangian_degrees,
        show_output=True,
        show_computation_time=True
    )
    assert verification_flag == True


if __name__ == "__main__":
    verify_quadrotor_sufficient_II()
    # verify_quadrotor_sufficient_I()
