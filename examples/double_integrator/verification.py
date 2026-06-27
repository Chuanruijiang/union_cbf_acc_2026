"""
This scripts defines different types of verification examples.
For simplicity, we call the double integrator system as DI.
For double integrator in 2D positional space, it's called DI2D.
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

from examples.double_integrator.dynamics import (
    DoubleIntegratorPlant2D,
)

# the following verification functions are for
# double integrator in 2D positional space.
def verify_single_hocbf_di_2d():
    """
    This function tests the verification of a single HOCBF for
    the double integrator in 2D positional space. We formulate
    the single HOCBF as a union of HOCBF with only one component.
    then use sufficient condition II to verify it. It is the same
    as verifying a single static HOCBF.

    This function finishes the verification process in about
    2 seconds.
    """
    double_integrator = DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    num_control = A.shape[1]
    cbfs = np.array([
        sym.Polynomial(x[0] + 7),
    ])
    relative_degree = [2]
    alphas = [[0.01, 0.1]]
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
        switching_cbfs=cbfs,
        static_cbfs=None,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees,
        show_computation_time=True
    )

    assert verification_flag == True

def verify_intersection_cbf_di_2d():
    """
    This function tests the verification of the feasibility
    within the intersection of 3 HOCBFs for the double 
    integrator in 2D positional space. Two of the static 
    HOCBFs are formulated by "static" and the third one is 
    formulated by "switching", whcich is a union of HOCBF 
    with a single component.

    This verification finishes in around 3-4 minutes.
    """
    double_integrator = DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    switching_cbfs = np.array([
        sym.Polynomial(-x[0] - 5),
    ])
    static_cbfs = np.array([
        sym.Polynomial(x[0] + 7),
        sym.Polynomial(-x[1] + 1)
    ])
    betas = [
        [0.01, 0.1],
        [0.01, 0.1],
        [0.01, 0.1]
    ]
    relative_degrees = [2, 2, 2]

    intersection_lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis=[
            [DegreeII(x=2, y=2, c=0), DegreeII(x=2, y=2, c=0)],
            [DegreeII(x=2, y=2, c=0), DegreeII(x=2, y=2, c=0)],
            [DegreeII(x=2, y=2, c=0), DegreeII(x=2, y=2, c=0)]
        ],
        lambda_y=[DegreeII(x=2, y=0, c=0), DegreeII(x=2, y=0, c=0)],
        xi_y=DegreeII(x=2, y=0, c=0),
        state_eq=None,
    )

    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
        relative_degree=relative_degrees,
        alpha=betas,
        state_eq_constr=None,
        lagrangian_degrees=intersection_lagrangian_degrees,
        show_computation_time=True
    )

    assert verification_flag == True

def verify_HOCBFs_validity():
    """
    This function tests the verification of the validity of 
    a union of two HOCBFs for the double integrator in 2D 
    positional space. The union is formulated as a group of 
    switching HOCBFs.
    The validty verification in this function uses the SOS
    program. 
    """
    double_integrator = DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    
    switching_cbfs = np.array([
        sym.Polynomial(x[1]),
        sym.Polynomial(-x[0])
        ])

    # the obstacle is a circle with randius 0.45 centered at (0.5, -1.0)
    unsafe_polys = np.array([
        sym.Polynomial((x[0] - 0.5)**2 + (x[1] + 1.0)**2 - 0.45**2)
    ])

    is_valid = verification.verify_switching_cbfs_validity(
        x=x[:2],
        switching_cbfs=switching_cbfs,
        unsafe_points=None,
        unsafe_polys=unsafe_polys,
        unsafe_poly_lagrangian_x_degrees=[0],
        cbf_lagrangian_x_degree=2
    )
    
    assert is_valid == True

def verify_HOCBFs_validity_with_unsafe_points():
    """
    This function tests the verification of the validity of 
    a union of two HOCBFs for the double integrator in 2D 
    positional space. The union is formulated as a group of 
    switching HOCBFs.
    The validty verification in this function checks the 
    unsafe points.
    """
    double_integrator = DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)

    # switching HOCBF
    switching_cbfs = np.array([
        sym.Polynomial(x[1]),
        sym.Polynomial(-x[0])
    ])
    # unsafe points sampled from the unsafe region
    # to test the verification method, we generate 100 
    # random points inside the region:
    # x_0 ∈ [0.01, 0.99], x_1 < -0.01
    unsafe_points = np.random.uniform(
        low=[0.01, -0.99],
        high=[0.99, -0.01],
        size=(100, 2)
        )

    is_valid= verification.verify_switching_cbfs_validity(
        x=x[:2],
        switching_cbfs=switching_cbfs,
        unsafe_points=unsafe_points,
        unsafe_polys=None,
        unsafe_poly_lagrangian_x_degrees=None,
        cbf_lagrangian_x_degree=None
    )
    assert is_valid == True

def verify_sufficient_condition_I_di_2d():
    """
    This function tests the verification of Verif-I for the 
    double integrator example.

    This verification is expected to take around 8 minutes.
    """
    # create a 2D double integrator system
    x = sym.MakeVectorContinuousVariable(4, "x")
    double_integrator = DoubleIntegratorPlant2D()
    f, g = double_integrator.affine_dynamics(x)
    (Au, bu) = double_integrator.control_input_limits()
    # HOCBFs:
    switching_cbfs = np.array([
        sym.Polynomial(-x[0] + 1),
        sym.Polynomial( x[1] - 1),
    ])
    relative_degree = [2]
    alphas = [[0.01, 0.1]]
    # lagrangian degrees:
    general_lagragian_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=2,
        activated_lagrangian_x_degree=2,
        activated_lagrangian_y_degree=2,
        deactivated_lagrangian_x_degree=2,
        deactivated_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=2,
        lambda_lagrangian_y_degree=0,
        xi_lagrangian_x_degree=2,
        xi_lagrangian_y_degree=0,
        state_eq_lagrangian_x_degree=None,
        state_eq_lagrangian_y_degree=None
    )
    # verification:
    verification_flag = verification.verify_sufficient_condition_I(
        x=x,
        f=f,
        g=g,
        control_limits=(Au, bu),
        switching_cbfs=switching_cbfs,
        static_cbfs=None,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=general_lagragian_degrees,
        show_output=True,
        show_computation_time=True
    )
    assert verification_flag == True

def verify_sufficient_condition_II_di_2d():
    """
    This function tests the verification of Verif-II for the 
    double integrator example.

    This verification is expected to take around 6-7 minutes.
    """
    double_integrator = DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    
    static_cbfs = np.array([
        sym.Polynomial(x[0] + 7),
        sym.Polynomial(-x[1] + 1)
    ])
    switching_cbfs = np.array([
        sym.Polynomial(-x[0] - 5),
        sym.Polynomial(x[1] + 1)
    ])
    betas = [
        [0.01, 0.1],
        [0.01, 0.1],
        [0.01, 0.1],
    ]
    relative_degrees = [2, 2, 2]

    intersection_lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis=[
            [DegreeII(x=2, y=2, c=0), DegreeII(x=2, y=2, c=0)],
            [DegreeII(x=2, y=2, c=0), DegreeII(x=2, y=2, c=0)],
            [DegreeII(x=2, y=2, c=0), DegreeII(x=2, y=2, c=0)]
        ],
        lambda_y=[
            DegreeII(x=2, y=0, c=0),
            DegreeII(x=2, y=0, c=0)
            ],
        xi_y=DegreeII(x=2, y=0, c=0),
        state_eq=None,
    )

    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
        relative_degree=relative_degrees,
        alpha=betas,
        state_eq_constr=None,
        lagrangian_degrees=intersection_lagrangian_degrees,
        show_output=True,
        show_computation_time=True
    )

    assert verification_flag == True


if __name__ == "__main__":
    verify_single_hocbf_di_2d()
    # verify_intersection_cbf_di_2d()
    # verify_HOCBFs_validity()
    # verify_HOCBFs_validity_with_unsafe_points()
    # verify_sufficient_condition_I_di_2d()
    #verify_sufficient_condition_II_di_2d()
