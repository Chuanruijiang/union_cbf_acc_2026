"""
This script compaares the verification time of verif-I and verif-II
for the following two types of CBFs:
 1. union of the 4 linear CBFs
 2. the higher-order CBF
We use the CBFs that are saved in the script "compare_cbfs.py", the
detailed expression of these CBFs can be found in that script.

In this script, we first load these CBFs, and then record the time
for verification.
"""
import time
import numpy as np
import os

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
from examples.single_integrator_2D.dynamics import SingleIntegrator2D
from examples.single_integrator_2D.compare_cbfs import (
    load_unsafe_region_and_cbfs
)

def verification_union_verif_I():
    """
    This function loads the 4 linear CBFs and verifies them using verif-I
    method.
    """
    filename = "unsafe_region_and_cbfs.pkl"
    cbf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator2D()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    (_, union_cbfs, _, _, _) = load_unsafe_region_and_cbfs(
        x_vars=x, 
        pickle_path=cbf_path
    )
    alpha = 1.0
    lagrangian_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=2,
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
        show_verification_time=True,
    )
    assert verification_flag == True

def verification_union_verif_II():
    """
    This function loads the 4 linear CBFs and verifies them using verif-II
    method.
    """
    filename = "unsafe_region_and_cbfs.pkl"
    cbf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator2D()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    num_controls = A.shape[1]
    (_, union_cbfs, _, _, _) = load_unsafe_region_and_cbfs(
        x_vars=x, 
        pickle_path=cbf_path
    )
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

def verification_higher_order_cbf_verif_II():
    """
    This function loads the higher-order CBF and verifies it using verif-II
    method.
    """
    filename = "unsafe_region_and_cbfs.pkl"
    cbf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator2D()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    num_controls = A.shape[1]
    (_, _, _, _, higher_order_cbf) = load_unsafe_region_and_cbfs(
        x_vars=x, 
        pickle_path=cbf_path
    )
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
        union_cbfs=np.array([higher_order_cbf]),
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
    verification_higher_order_cbf_verif_II()