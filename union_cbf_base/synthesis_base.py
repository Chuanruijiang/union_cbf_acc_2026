"""
This block achieves the synthesis of switching HOCBFs in 
a 2D dimensional domain. The detailed compuation process
is described in the Section 6 of the Journal paper.
"""

import os
import pickle
import numpy as np
import pydrake.symbolic as sym
from typing import List, Tuple, Optional

from union_cbf_base.utils import(
    serialize_polynomial
)
from union_cbf_base.union_cbf_II import (
    CbfFeasibilityLagrangianDegrees,
)
from union_cbf_base.verification_base import (
    verify_hocbfs_overlap,
    verify_switching_cbfs_validity,
    verify_sufficient_condition_II
)


def line_search_bias(
    x_vars: np.ndarray,
    cbf_coeffs: List[float],
    unsafe_points: Optional[np.ndarray],
    unsafe_polys: Optional[np.ndarray],
    unsafe_poly_lagrangian_x_degrees: Optional[List[int]],
    cbf_lagrangian_x_degree: Optional[int],
) -> sym.Polynomial:
    """
    This function performs a line search to find the maximum bias
    term for the linear HOCBF defined by the given coefficients.
    Since we are focusing on linear 2D HOCBFs in the position domain,
    then the HOCBF should be in the form of:
        h(x) = a*x[0] + b*x[1] + c
    where the x[0] and x[1] are the position in the 2D space. The
    to this function should be a list of [a, b, c].
    Hence, this function will search for suitable c (we does not change
    the a, b in this function) such that the region {x|h(x) >= 0} excludes
    all the unsafe points, or the unsafe regions presented by polynomials.
    """
    assert len(cbf_coeffs) == 3
    a = cbf_coeffs[0]
    b = cbf_coeffs[1]
    c_lower = -15.0
    c_upper = 15.0
    tolerance = 1e-2
    max_iterations = 20
    best_c = None

    for _ in range(max_iterations):
        mid_c = (c_lower + c_upper) / 2.0
        h_poly = sym.Polynomial(
            a*x_vars[0] + b*x_vars[1] + mid_c
        )
        cbf_valid = verify_switching_cbfs_validity(
            x=x_vars[0:2],
            switching_cbfs=np.array([h_poly]),
            unsafe_points=unsafe_points,
            unsafe_polys=unsafe_polys,
            unsafe_poly_lagrangian_x_degrees=unsafe_poly_lagrangian_x_degrees,
            cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
        )
        if cbf_valid:
            best_c = mid_c
            c_lower = mid_c
        else:
            c_upper = mid_c
        if c_upper - c_lower < tolerance:
            break
    if best_c is not None:
        final_cbf = sym.Polynomial(
            a*x_vars[0] + b*x_vars[1] + best_c
        )
        return final_cbf
    else:
        return None

def get_weight_from_angle(angle: float) -> List[float]:
    """
    This function computes the weight vector for a linear HOCBF
    given the angle (in radians) of the normal vector of the HOCBF.
    """
    w_x0 = np.cos(angle)
    w_x1 = np.sin(angle)
    norm = np.sqrt(w_x0**2 + w_x1**2)
    w_x0 /= norm
    w_x1 /= norm
    if np.abs(w_x0) <= 1e-8:
        w_x0 = 0.0
    if np.abs(w_x1) <= 1e-8:
        w_x1 = 0.0
    return [w_x0, w_x1]

def find_linear_hocbf(
    x_vars: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    control_limits: Optional[Tuple[np.ndarray, np.ndarray]],
    static_cbfs: Optional[np.ndarray],
    previously_found_cbf: Optional[sym.Polynomial],
    previous_relative_degree: Optional[int],
    previous_alpha: Optional[List[float]],
    relative_degree: List[int],
    alpha: List[List[float]],
    state_eq_constr: Optional[np.ndarray],
    unsafe_points: Optional[np.ndarray],
    unsafe_polys: Optional[np.ndarray],
    initial_angle: float,
    angle_increment_min: float,
    angle_increment_max: float,
    samples: int,
    lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
    unsafe_poly_lagrangian_x_degrees: Optional[List[int]],
    cbf_lagrangian_x_degree: Optional[int],
) -> Tuple[sym.Polynomial, float]:
    """
    This function finds a linear HOCBF by searching through different
    angles for the normal vector of the HOCBF and also line searching
    the bias term for each angle.
    """
    bias_init = 0.01
    possible_angles = np.linspace(angle_increment_min, angle_increment_max, num=samples)
    all_cbfs = static_cbfs
    all_relative_degree = relative_degree
    all_alpha = alpha
    if previously_found_cbf is not None:
        assert previous_relative_degree is not None
        assert previous_alpha is not None
        all_cbfs = np.append(all_cbfs, np.array([previously_found_cbf]), axis=0)
        all_relative_degree = relative_degree + [previous_relative_degree]
        all_alpha = alpha + [previous_alpha]
    for theta_r in possible_angles:
        theta = initial_angle + theta_r
        weights = get_weight_from_angle(theta)
        coeffs = [weights[0], weights[1], bias_init]
        hocbf = line_search_bias(
            x_vars=x_vars,
            cbf_coeffs=coeffs,
            unsafe_points=unsafe_points,
            unsafe_polys=unsafe_polys,
            unsafe_poly_lagrangian_x_degrees=unsafe_poly_lagrangian_x_degrees,
            cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
        )
        assert hocbf is not None
        print(f"Trying angle {theta} radians, found hocbf: {hocbf}")
        
        # check the feasibility
        is_feasible = verify_sufficient_condition_II(
            x=x_vars,
            f=f,
            g=g,
            control_limits=control_limits,
            switching_cbfs=np.array([hocbf]),
            static_cbfs=static_cbfs,
            relative_degree=relative_degree,
            alpha=alpha,
            state_eq_constr=state_eq_constr,
            lagrangian_degrees=lagrangian_degrees
        )
        all_cbfs = (
            np.append(all_cbfs, np.array([hocbf]), axis=0)
            if hocbf is not None
            else all_cbfs
        )
        is_non_trivial = verify_hocbfs_overlap(
            x=x_vars,
            f=f,
            g=g,
            state_eq_constr=state_eq_constr,
            cbfs=all_cbfs,
            relative_degree=all_relative_degree,
            alphas=all_alpha,
        )
        if is_feasible and is_non_trivial:
            print(f"Found a valid linear HOCBF at angle {theta} radians.")
            return hocbf, theta
    print(f"Failed to find a valid linear HOCBF starting \
            from {initial_angle} radians to {initial_angle + np.pi/2} radians.\
            please try other angle ranges or other initial angles.")
    return None, None

def if_interested_states_included(
    x_vars: np.ndarray,
    switching_cbfs: np.ndarray,
    interested_states: np.ndarray
) -> bool:
    """
    Thid function checks whether all the given interested states
    are included in the safe set defined by the union_cbfs.
    The interested states should be initial states or some waypoint
    states that we want to ensure they are included in the safe set.
    We can evaluate the each CBF of the union_cbfs at each of
    the interested states. If for each interested state, there is at
    least one CBF that outputs a non-negative value, then the 
    interested states are all included in the safe set.
    Note that the interested states should be a 2D array with each
    row being a state.
    """
    assert switching_cbfs is not None
    num_cbfs = switching_cbfs.shape[0]
    cbf_func_values = [
        switching_cbfs[i].EvaluateIndeterminates(
            indeterminates=x_vars[0:2],
            indeterminates_values=interested_states.T
        )
        for i in range(num_cbfs)
    ]
    cbf_func_val_stacked = np.stack(arrays=cbf_func_values, axis=0)
    result = np.any(cbf_func_val_stacked >= 0, axis=0)
    return np.all(result)

def find_2d_linear_hocbfs(
    x_vars: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    control_limits: Optional[Tuple[np.ndarray, np.ndarray]],
    static_cbfs: Optional[np.ndarray],
    relative_degree: List[int],
    alpha: List[List[float]],
    state_eq_constr: Optional[np.ndarray],
    unsafe_points: Optional[np.ndarray],
    unsafe_polys: Optional[np.ndarray],
    initial_angle: float,
    angle_diff: float,
    iter_num: int,
    interested_states: np.ndarray,
    lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
    unsafe_poly_lagrangian_x_degrees: Optional[List[int]],
    cbf_lagrangian_x_degree: Optional[int],
) -> np.ndarray:
    """
    This function finds multiple linear HOCBFs in 2D positional space
    to exclude all the unsafe points or unsafe regions. The found
    HOCBFs should includes all the provided initial positions for the
    system.
    The output of this system is self.switching_cbfs, which is an array
    of polynomials.
    """
    stop_flag = False
    iteration = 0
    angle_init = initial_angle
    switching_cbfs = []
    previously_found_cbf = None
    previous_relative_degree = None
    previous_alpha = None
    while(stop_flag == False and iteration < iter_num):
        print(f"Finding HOCBFs iteration {iteration}...")
        print(f"initial angle is {angle_init} radians.")
        (hocbf, angle_find) = find_linear_hocbf(
            x_vars=x_vars,
            f=f,
            g=g,
            control_limits=control_limits,
            static_cbfs=static_cbfs,
            previously_found_cbf=previously_found_cbf,
            previous_relative_degree=previous_relative_degree,
            previous_alpha=previous_alpha,
            relative_degree=relative_degree,
            alpha=alpha,
            state_eq_constr=state_eq_constr,
            unsafe_points=unsafe_points,
            unsafe_polys=unsafe_polys,
            initial_angle=angle_init,
            angle_increment_min=0.0,
            angle_increment_max=np.pi/4,
            samples=10,
            lagrangian_degrees=lagrangian_degrees,
            unsafe_poly_lagrangian_x_degrees=unsafe_poly_lagrangian_x_degrees,
            cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
        )
        assert hocbf is not None
        previously_found_cbf = hocbf
        previous_relative_degree = relative_degree[-1]
        previous_alpha = alpha[-1]
        switching_cbfs.append(hocbf)
        if if_interested_states_included(
            x_vars=x_vars,
            switching_cbfs=np.array(switching_cbfs),
            interested_states=interested_states
        ):
            stop_flag = True
            print("All the interested states are included in the safe set.")
            return np.array(switching_cbfs)
        elif iteration >= iter_num:
            print("Reached the maximum number of iterations.")
            return np.array(switching_cbfs)
        else:
            angle_init = angle_find + angle_diff
            iteration += 1

def save_synthesized_hocbfs(
    x_vars: np.ndarray,
    switching_cbfs: np.ndarray,
    static_cbfs: Optional[np.ndarray],
    pickle_path: str
):
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}
    # save the cbf data:
    x_set = sym.Variables(x_vars)
    if static_cbfs is not None:
        data["static_cbfs"] = [
            serialize_polynomial(h_i, x_set) 
            for h_i in static_cbfs
            ]
    assert switching_cbfs is not None
    data["switching_cbfs"] = [
        serialize_polynomial(h_i, x_set)
        for h_i in switching_cbfs
        ]

    if os.path.exists(pickle_path):
        overwrite_cmd = input(
            f"File {pickle_path} already exists. Overwrite the file? Press [Y/n]:"
        )
        if overwrite_cmd in ("Y", "y"):
            save_cmd = True
        else:
            save_cmd = False
    else:
        save_cmd = True

    if save_cmd:
        with open(pickle_path, "wb") as handle:
            pickle.dump(data, handle)
    
    