"""
This scripts defines different types of verification examples.
For simplicity, we call the double integrator system as DI.
For double integrator in 2D position by with constant x-axis
velocity, it's called DI2D-const-vel.
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

from dynamics import (
    DoubleIntegratorPlantConstantVel
)

# The following verification functions are for 
# double integrator with constant velocity in x-axis
def verify_single_hocbf_di_2d_const_vel():
    """
    This function tests the verification of double_integrator with
    constant horizontal velocity. The safe region is a half-plane
    {x | h(x) = -x[0] + x[1] - 5 >= 0}
    The verification of this function is supposed to pass.
    """
    di_const_vel = DoubleIntegratorPlantConstantVel(vx_const=1.0)
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = di_const_vel.affine_dynamics(x)
    (A, c) = di_const_vel.control_input_limits()
    num_control = A.shape[1]
    cbfs = np.array([
        sym.Polynomial(-x[0] + x[1] - 5),
    ])
    relative_degree = [2]
    alphas = [[0.1, 1.0]]
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [
            [DegreeII(x=2, y=2, c=0)] * relative_degree[0]
        ],
        lambda_y=[DegreeII(x=3, y=0, c=0)]*num_control,
        xi_y=DegreeII(x=3, y=0, c=0),
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
        lagrangian_degrees=lagrangian_degrees
    )

    assert verification_flag == True

def verify_intersection_cbf_di_2d_const_vel():
    """"
    The verification of this intersection case SHOULD FAIL.
    becaues the two CBFs actually form a blocking corner  
    as the di system moves along x-axis. But since the system
    always has a positive velocity in x-axis, it cannot
    stay inside the safe region forever.
    """
    di_const_vel = DoubleIntegratorPlantConstantVel(vx_const=1.0)
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = di_const_vel.affine_dynamics(x)
    (A, c) = di_const_vel.control_input_limits()
    num_control = A.shape[1]
    static_cbfs = np.array([
        sym.Polynomial(20 - x[1]),
    ])
    switching_cbfs = np.array([
        # in fact, only this HOCBF alone does PASS verification
        # since this single HOCBF does not form any blocking
        # corners. See function 
        # "verify_single_hocbf_di_2d_const_vel()"
        # for more details.
        # but now we also have another HOCBF that, together
        # with this one, forms a blocking coner.
        sym.Polynomial(-x[0] + x[1] - 5),
    ])  
    relative_degree = [2, 2]
    alphas = [[0.1, 1.0], [0.1, 1.0]]
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [
            [DegreeII(x=2, y=2, c=0)] * relative_degree[0],
            [DegreeII(x=2, y=2, c=0)] * relative_degree[1],
        ],
        lambda_y=[DegreeII(x=3, y=0, c=0)]*num_control,
        xi_y=DegreeII(x=3, y=0, c=0),
        state_eq=None,
    )
    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees
    )

    assert verification_flag == False
    
def verify_sufficient_condition_II_di_2d_const_vel():
    """
    The verification based on the sufficent condition II
    should FAIL since the first switching CBFs and the 
    normal CBF form a closed safe set as the di system
    moves along x-axis. See the function
    "verify_intersection_cbf_di_2d_const_vel()"
    above for details.
    """
    di_const_vel = DoubleIntegratorPlantConstantVel(vx_const=1.0)
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = di_const_vel.affine_dynamics(x)
    (A, c) = di_const_vel.control_input_limits()
    
    static_cbfs = np.array([
        sym.Polynomial(20 - x[1]),
    ])
    switching_cbfs = np.array([
        # the first switching CBF and the normal CBF
        # could pass the verification since the blocking
        # corner is formed behind the system along
        # the x-axis direction and the system always has
        # a positive velocity in x-axis.
        sym.Polynomial(x[0] + x[1] - 5),
        # the second switching CBF and the normal CBF
        # could fail the verification.
        sym.Polynomial(-x[0] + x[1] - 5),
    ])  
    relative_degree = [2, 2]
    alphas = [[0.1, 1.0], [0.1, 1.0]]

    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [
            [DegreeII(x=2, y=2, c=0)] * relative_degree[0],
            [DegreeII(x=2, y=2, c=0)] * relative_degree[1],
        ],
        lambda_y=[DegreeII(x=3, y=0, c=0)]*A.shape[1],
        xi_y=DegreeII(x=3, y=0, c=0),
        state_eq=None,
    )
    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees,
        show_output=True
    )

    assert verification_flag == False

def verify_sufficient_condition_I_di_2d_const_vel():
    # for the same normal CBFs and switching CBFs also with
    # the same alpha parameters, the verification of method I
    # PASSES while method II fails. This shows that method I
    # is less conservative than method II.
    x = sym.MakeVectorContinuousVariable(3, "x")
    di_const_vel = DoubleIntegratorPlantConstantVel(vx_const=1.0)
    f, g = di_const_vel.affine_dynamics(x)
    (Au, bu) = di_const_vel.control_input_limits()
    static_cbfs = np.array([
        sym.Polynomial(25 - x[1]),
    ])
    switching_cbfs = np.array([
        sym.Polynomial(-x[0] + x[1] - 10),
        sym.Polynomial(x[0] + x[1] - 10),
    ])
    relative_degree = [2, 2]
    alphas = [[0.1, 1.0], [0.1, 1.0]]
    # lagrangian degrees:
    general_lagragian_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=1,
        activated_lagrangian_x_degree=2,
        activated_lagrangian_y_degree=2,
        deactivated_lagrangian_x_degree=2,
        deactivated_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=3,
        lambda_lagrangian_y_degree=0,
        xi_lagrangian_x_degree=3,
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
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=general_lagragian_degrees,
        show_output=True
    )
    assert verification_flag == True

if __name__ == "__main__":
    # verify_single_hocbf_di_2d_const_vel()
    # verify_intersection_cbf_di_2d_const_vel()
    # verify_sufficient_condition_II_di_2d_const_vel()
    verify_sufficient_condition_I_di_2d_const_vel()
