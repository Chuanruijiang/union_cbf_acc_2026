# """
# In this plot, we verify the union of a circular CBF and a linear 
# CBF for a single integrator system with constant horizontal
# velocity.
# The expected results should be:
# the verif-I method passed,
# but the verif-II method failed since it is impossible to keep
# the system with a contant horizontal velocity inside a circle.
# """

import numpy as np
import pydrake.symbolic as sym

from union_cbf_base.union_cbf_I import (
    UnionCbfI,
    SubsetGeneralLagrangianDegrees,
)
from union_cbf_base.union_cbf_II import (
    UnionCbfII,
    CbfFeasibilityLagrangianDegrees,
    Degree as DegreeII,
)
from examples.single_integrator_2D_const_vel.dynamics import(
    SingleIntegrator
)

def verification_union_verif_I():
    """
    This function verifies the union of a circular CBF and a linear 
    CBF for a single integrator system with constant horizontal
    velocity using verif-I method.
    The expected results should be passed
    """
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator(v_const=1.0)
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    union_cbfs = np.array([
        sym.Polynomial(x[0] - x[1]),
        sym.Polynomial(8 - (x[0] + 2)**2 - (x[1] + 2)**2)
    ])
    
    alpha = 1.0
    lagrangian_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=1,
        cbfs_lagrangian_x_degree=2,
        cbfs_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=2,
        lambda_lagrangian_y_degree=0,
        xi_lagrangian_x_degree=2,
        xi_lagrangian_y_degree=0,
    )
    union_object = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        control_limits=(A, c),
    )
    (verification_flag,_) = union_object.verification_feasibility_condition_I(
        union_cbfs=union_cbfs,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_verification_time=False,
    )
    assert verification_flag == True

def verification_union_verif_II():
    """
    This function loads the 4 linear CBFs and verifies them using verif-II
    method.
    """
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    num_controls = A.shape[1]
    union_cbfs = np.array([
        sym.Polynomial(x[0] - x[1]),
        sym.Polynomial(8 - (x[0] + 2)**2 - (x[1] + 2)**2)
    ])
    alpha = 1.0

    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        cbf=DegreeII(x=2, y=2, c=0),
        lambda_y=[DegreeII(x=2, y=0, c=0)]*num_controls,
        xi_y=DegreeII(x=2, y=0, c=0),
    )
    union_object = UnionCbfII(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        control_limits=(A, c),
    )
    (verification_flag,_) = union_object.verification_feasibility_condition_II(
        union_cbfs=union_cbfs,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_verification_time=True,
    )
    assert verification_flag == True

if __name__ == "__main__":
    verification_union_verif_I()
    verification_union_verif_II()
