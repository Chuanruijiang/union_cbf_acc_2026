import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import os
import os.path
import pickle
from typing import Optional
import numpy as np
import pydrake.symbolic as sym

from compatible_clf_union_cbf.utils import(
    compute_minimum_on_boundary,
    serialize_polynomial,
    deserialize_polynomial,
    BackoffScale
)
from compatible_clf_union_cbf.clf import(
    ClfSynthesis
)
from compatible_clf_union_cbf.union_cbf import(
    UnionCbfSynthesisGivenClf
)

from dynamics import (
    system_dynamics,
    control_limits
)


def get_pkl_file_path():
    filename = "single_integrator_synthesized_clf_union_cbf.pkl"
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path

def save_clf_cbf(
    V: Optional[sym.Polynomial],
    h: np.ndarray,
    x_set: sym.Variables,
    kappa_V: Optional[float],
    kappa_h: np.ndarray,
    pickle_path: str,
):
    """
    Save the CLF and CBF to a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}
    if V is not None:
        data["V"] = serialize_polynomial(V, x_set)
    data["h"] = [serialize_polynomial(h_i, x_set) for h_i in h]
    if kappa_V is not None:
        data["kappa_V"] = kappa_V
    data["kappa_h"] = kappa_h

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

def load_clf_cbf(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    if "V" in data.keys():
        ret["V"] = deserialize_polynomial(data["V"], x_set)
    ret["h"] = np.array(
        [
            deserialize_polynomial(h_i, x_set)
            for h_i in data["h"]
        ]
    )
    if "kappa_V" in data.keys():
        ret["kappa_V"] = data["kappa_V"]
    ret["kappa_h"] = data["kappa_h"]
    return ret

def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    x_set = sym.Variables(x)
    f, g = system_dynamics()
    V_init = sym.Polynomial(x[0]**2 + x[1]**2)*0.25

    # Environment parameters
    rho = 1
    expected_kappaV = 0.2
    kappaV = 0.8
    kappa_diff = kappaV - expected_kappaV
    kappah = [1,1,1]
    
    #(Au, bu) = control_limits()
    (Au, bu) = (None, None)
    
    epsilon_0 = 1
    points_to_include = np.array([
        [-1 ,1.5],
        [-1.5, 1],
        [-2, 1.5],
        [-1.8, 2.1],
        [-1.2, 3],
        [-2.5, 1.5],
        [-2.5, 2],
        [-3,3],
        [-3.5, 1.2],
        [-3.5, 2.2],
        [-5, 1.5],
        [-5, 2.3],
        [-6, 0.8],
        [-7, 0],
        [-7, 0.5],
        [-8, 1],
        [-8, 2.5],
        [-8, 0],
        [-8, 0.5],
        [-9, 0],
        [-9, 1],
        [-10, 0],
        [-10, 1.5],
        [-10.5, 2],
        [-11, 1.5],
        [-11, 2],
        [-11, 0],
        [-11, -1],
        [-11, -1.5]
    ])
    points_inlusion_weights = np.ones(points_to_include.shape[0])

    unsafe_polys = np.array([
        sym.Polynomial(x[0] + 5),
        sym.Polynomial(-x[0] - 3),
        sym.Polynomial(x[1] + 1),
        sym.Polynomial(-x[1] + 1),
    ])

    # degree specification:
    ball_inclusion_ball_x_degree = 2
    ball_inclusion_poly_x_degree = 2
    clf_lagrangian_lambda_y_x_degree = [2, 2]
    clf_lagrangian_xi_y_x_degree = 2
    clf_lagrangian_rho_minus_V_x_degree = 2
    V_x_degree = 2

    clf_synthesis = ClfSynthesis(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        Au=Au,
        bu=bu,
        state_eq_constraint=None
        )
    
    V_result = clf_synthesis.bilinear_alternation(
        clf_init = V_init,
        rho=rho,
        kappaV=kappaV,
        ball_radius=epsilon_0,
        ball_inclusion_ball_x_degree=ball_inclusion_ball_x_degree,
        ball_inclusion_h_x_degree=ball_inclusion_poly_x_degree,
        clf_lagrangain_lambda_y_x_degree=clf_lagrangian_lambda_y_x_degree,
        clf_lagrangain_xi_y_x_degree=clf_lagrangian_xi_y_x_degree,
        clf_lagrangain_rho_minus_V_x_degree=clf_lagrangian_rho_minus_V_x_degree,
        V_x_degree=V_x_degree,
        state_eq_constraints_x_degree=None,
        included_points=points_to_include,
        points_inclusion_weights=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=20,
        lagrangian_coeff_tol=1e-3,
    )
    assert V_result is not None

    # data = load_clf_cbf(get_pkl_file_path(), x_set)
    # V_result = data["V"]
    # cbf_1_result = data["h"][0]

    # compute the minimum value of V on the boundary of the ball(ϵ_0):
    V_min = compute_minimum_on_boundary(
        x=x,
        p=V_result,
        q=sym.Polynomial(epsilon_0**2 - x.dot(x))
    )
    print(V_min)
    epsilon = kappa_diff*V_min
    print(f"We use this epsilon for the following CBF synthesis: {epsilon}")

    # # CBF synthesis:
    cbf_synthesis_given_clf = UnionCbfSynthesisGivenClf(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        clf=V_result,
        rho=rho,
        num_cbf=3,
        unsafe_polys=unsafe_polys,
        Au=Au,
        bu=bu,
        state_eq_constraints=None,
        kappaV=kappaV,
        kappah=kappah,
        epsilon_0=epsilon_0,
        epsilon=epsilon,
        cbf_x_degrees=[1, 1, 1],
        cbf_ball_inclusion_ball_x_degree=2,
        cbf_ball_inclusion_cbf_x_degree=2,
        compatible_lambda_y_x_degrees=[2, 2],
        compatible_xi_y_x_degree=2,
        compatible_rho_minus_V_x_degree=2,
        compatible_deact_cbf_x_degree=[2, 2],
        compatible_h_x_degree=2,
        state_eq_x_degrees=None,
        safety_h_x_degree=2,
        safety_unsafe_polys_x_degree=[2, 2, 2, 2]
    )

    # synthesis the first CBF:
    itermax = 10
    back_off_scales = [BackoffScale(rel=0.02, abs=None)] * itermax
    cbf_1_result = cbf_synthesis_given_clf.synthesis_first_cbf(
        cbf_init=sym.Polynomial(x[0] + 1.5),
        points_to_include=points_to_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=np.array([
            [0 , 0]
            ]),
        anchor_bounds=(
            np.array([0]), np.array([1.5])
            ),
        max_iter=itermax,
        back_off_scale=back_off_scales,
    )
    assert cbf_1_result is not None

    # # save the synthesis results first:
    # save_clf_cbf(
    #     V=V_result,
    #     h=np.array([cbf_1_result]),
    #     x_set=x_set,
    #     kappa_V=kappaV,
    #     kappa_h=kappah,
    #     pickle_path=get_pkl_file_path()
    # )
    
    (
        points_to_include,
        points_inlusion_weights,
    ) = cbf_synthesis_given_clf.remove_included_points(
        included_points=points_to_include,
        points_inclusion_weights=points_inlusion_weights,
        solved_cbf=cbf_1_result
    )

    # synthesis the second CBF:
    itermax = 2
    back_off_scales = [BackoffScale(rel=0.02, abs=None)] * itermax
    cbf_2_result = cbf_synthesis_given_clf.synthesis_other_cbf(
        cbf_init=sym.Polynomial(x[1] - 5),
        cbf_index=1,
        deact_cbfs=np.array([cbf_1_result]),
        points_to_include=points_to_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=itermax,
        back_off_scale=back_off_scales
    )
    assert cbf_2_result is not None

    (
        points_to_include,
        points_inlusion_weights,
    ) = cbf_synthesis_given_clf.remove_included_points(
        included_points=points_to_include,
        points_inclusion_weights=points_inlusion_weights,
        solved_cbf=cbf_2_result
    )

    # synthesis the third CBF:
    itermax = 5
    back_off_scales = [BackoffScale(rel=0.02, abs=None)] * itermax
    cbf_3_result = cbf_synthesis_given_clf.synthesis_other_cbf(
        cbf_init=sym.Polynomial(-x[0] + x[1] - 15),
        cbf_index=2,
        deact_cbfs=np.array([cbf_1_result, cbf_2_result]),
        points_to_include=points_to_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=itermax,
        back_off_scale=back_off_scales
    )
    assert cbf_3_result is not None

    # # Save the results:
    # save_clf_cbf(
    #     V=V_result,
    #     h=np.array([cbf_1_result, cbf_2_result, cbf_3_result]),
    #     x_set=x_set,
    #     kappa_V=kappaV,
    #     kappa_h=kappah,
    #     pickle_path=get_pkl_file_path()
    # )


    

if __name__ == "__main__":
    main()
