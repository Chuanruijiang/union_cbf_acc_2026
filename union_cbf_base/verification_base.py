"""
This block provides functions for to achieve all the verification
tasks described in the Section 5 of the journal paper. In the
examples, we only need to call functions from this block for
verification and synthesis.
"""

import numpy as np
from typing import List, Optional, Tuple

import pydrake.symbolic as sym

from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.inclusion import UnsafeExclusion
from union_cbf_base.union_cbf_I import (
    UnionCbfI,
    SubsetGeneralLagrangianDegrees
    )
from union_cbf_base.union_cbf_II import (
    UnionCbfII,
    CbfFeasibilityLagrangianDegrees
)
from union_cbf_base.utils import lower_lie_derivatives

def verify_hocbfs_overlap(
    x:np.ndarray,
    f:np.ndarray,
    g:np.ndarray,
    state_eq_constr: Optional[np.ndarray],
    cbfs:np.ndarray,
    relative_degree: List[int],
    alphas: List[List[float]],
) -> bool:
    """
    This function verifies whether the forward invariant sets of the
    given HOCBFs have non-empty intersection. The cbfs input should be 
    an array of symbolic polynomials.
    """ 
    assert cbfs.shape[0] == len(relative_degree) == len(alphas)   
    activated_poly_groups = []
    for i in range(len(cbfs)):
        forward_invariant_polys = lower_lie_derivatives(
            poly=cbfs[i],
            vector_field=f,
            variables=x,
            relative_degree=relative_degree[i],
            betas=alphas[i]
        )
        forward_invariant_polys = np.hstack(
            (cbfs[i], forward_invariant_polys)
        )
        activated_poly_groups.append(forward_invariant_polys)
    subset = Subset(
        x=x,
        activated_poly_groups=activated_poly_groups,
        deactivated_polys=None,
        equation_constraints=state_eq_constr
    )
    is_empty = subset.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=0,
        lagrangian_deact_c_degree=0
    )
    return not is_empty

def verify_switching_cbfs_validity(
    x: np.ndarray,
    switching_cbfs: np.ndarray,
    unsafe_polys: Optional[np.ndarray],
    unsafe_points: Optional[np.ndarray],
    unsafe_poly_lagrangian_x_degrees: Optional[List[int]],
    cbf_lagrangian_x_degree: Optional[int],
) -> bool:
    """
    This function verifies whether the switching CBFs are valid by checking
    if their forward invariant sets do not intersect with the unsafe sets.
    if the unsafe_polys input is None, then only the unsafe_points are used
    The input "x" should be the only state variables that the CBFs depend on.
    """
    if unsafe_points is None:
        assert unsafe_polys is not None
        assert unsafe_poly_lagrangian_x_degrees is not None
        assert cbf_lagrangian_x_degree is not None
    
    num_cbfs = switching_cbfs.shape[0]
    for i in range(num_cbfs):
        current_cbf = switching_cbfs[i]
        unsafe_exclusion = UnsafeExclusion(
            h=current_cbf,
            x=x,
            unsafe_polys=unsafe_polys,
            unsafe_points=unsafe_points
        )
        if unsafe_points is not None:
            is_valid = unsafe_exclusion.verify_unsafe_point_exclusion()
            
        if unsafe_polys is not None:
            is_valid = unsafe_exclusion.verify_unsafe_exclusion(
                unsafe_poly_x_degrees=unsafe_poly_lagrangian_x_degrees,
                h_x_degree=cbf_lagrangian_x_degree,
            )
        if not is_valid:
            return False
    return True

def verify_sufficient_condition_I(
    x: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    control_limits: Optional[Tuple[np.ndarray, np.ndarray]],
    switching_cbfs: np.ndarray,
    static_cbfs: Optional[np.ndarray],
    relative_degree: List[int],
    alpha: List[List[float]],
    state_eq_constr: Optional[np.ndarray],
    lagrangian_degrees: SubsetGeneralLagrangianDegrees,
    show_output: bool = False,
    show_computation_time: bool = False
    ) -> bool:
    union_cbf_obj = UnionCbfI(
        x=x,
        f=f,
        g=g,
        control_limits=control_limits,
        relative_degree=relative_degree,
        alpha=alpha,
        state_eq_constr=state_eq_constr
    )
    (is_feasible, verification_time) = union_cbf_obj.verification_feasibility_condition_I(
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
        general_degrees=lagrangian_degrees,
        eta=1e-5,
        eps=1e-5,
        show_output=show_output,
        show_verification_time=show_computation_time
    )
    if show_computation_time:
        print(f"Total verification time: {verification_time}")
    return is_feasible

def verify_sufficient_condition_II(
    x: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    control_limits: Optional[Tuple[np.ndarray, np.ndarray]],
    switching_cbfs: np.ndarray,
    static_cbfs: Optional[np.ndarray],
    relative_degree: List[int],
    alpha: List[List[float]],
    state_eq_constr: Optional[np.ndarray],
    lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
    show_output: bool = False,
    show_computation_time: bool = False
    ) -> bool:
    union_cbf_obj = UnionCbfII(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        relative_degree=relative_degree,
        control_limits=control_limits,
        state_eq_constr=state_eq_constr
    )
    is_feasible = union_cbf_obj.verification_feasibility_condition_II(
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-5,
        eps=1e-5,
        show_output = show_output,
        show_computation_time=show_computation_time
    )
    return is_feasible

