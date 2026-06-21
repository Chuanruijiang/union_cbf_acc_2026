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
    general_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=A.shape[1],
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
    union_object = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=[[alpha]],
        relative_degree=[1],
        control_limits=(A, c),
    )
    (verification_flag, _) = union_object.verification_feasibility_condition_I(
        switching_cbfs=union_cbfs,
        static_cbfs=None,
        general_degrees=general_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_verification_time=False,
    )
    assert verification_flag == True

def verification_union_verif_II():
    """
    This function verifies a union of a linear and a circular CBF.
    The verification result is supposed to be failed since it is
    impossible to keep the single integrator with a constant
    velocity always safe within a closed circular region. 
    """
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    num_controls = A.shape[1]
    switching_cbfs = np.array([
        sym.Polynomial(x[0] - x[1]),
        sym.Polynomial(8 - (x[0] + 2)**2 - (x[1] + 2)**2)
    ])
    alpha = [[1.0]]
    relative_degree = [1]

    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis=[[DegreeII(x=2, y=2, c=0)]],
        lambda_y=[DegreeII(x=2, y=0, c=0)]*num_controls,
        xi_y=DegreeII(x=2, y=0, c=0),
        state_eq=None,
    )
    union_object = UnionCbfII(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        relative_degree=relative_degree,
        control_limits=(A, c),
    )
    verification_flag = union_object.verification_feasibility_condition_II(
        switching_cbfs=switching_cbfs,
        static_cbfs=None,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_computation_time=True,
    )
    assert verification_flag == True

if __name__ == "__main__":
    verification_union_verif_I()
    verification_union_verif_II()
