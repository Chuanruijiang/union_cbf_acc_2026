"""
In this file we synthesize compatible CLF and single quadrtic CBF
for the non-linear toy example. We still use the code base for the
synthesis of CLF and union of CBFs, but only synthesis the first CBF.
and set the CBF degree to be 2.
"""
import os
import sys
import os.path
import pickle
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

from typing import Optional
import numpy as np
import pydrake.symbolic as sym

from union_cbf_base.utils import(
    compute_minimum_on_boundary,
    serialize_polynomial,
    deserialize_polynomial,
    BackoffScale
)
from union_cbf_base.clf import(
    ClfSynthesis
)
from union_cbf_base.union_cbf import(
    UnionCbfSynthesisGivenClf
)
from dynamics import(
    system_dynamics,
    state_equation_constraint,
    control_limits,
    original_to_extended_state_space
)

def get_pkl_file_path(
    name_file: str
):  
    filename = name_file
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path

def save_synthesis_results(
    V_res: sym.Polynomial,
    h_res: np.ndarray,
    x_set: sym.Variables,
    pickle_path: str,
):
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}
    
    data["V"] = serialize_polynomial(V_res, x_set)
    data["h"] = [serialize_polynomial(h_i, x_set) for h_i in h_res]

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

def load_clf(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    ret["V"] = deserialize_polynomial(data["V"], x_set)
    return ret

def main():
    pi = np.pi
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)
    
    data = load_clf(
        pickle_path=get_pkl_file_path(
            name_file="non_linear_toy_clf_init.pkl"
        ),
        x_set=x_set
    )
    V_init = data["V"]

    f, g = system_dynamics(x)
    state_eq_const = state_equation_constraint(x)

    # (Au, bu) = control_limits(x)
    (Au, bu) = None, None

    # set parameters:
    epsilon_0 = 0.1
    rho = 1
    kappaV = 0.05
    kappa_diff = 0.01
    kappah = [1.0, 1.0]

    # unsafe Region:
    unsafe_polys = np.array([
        sym.Polynomial(-x[0] + 0.3),
        sym.Polynomial(-x[1] + np.sin(-pi/4))
    ])

    clf_synthesis = ClfSynthesis(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        Au=Au,
        bu=bu,
        state_eq_constraint=state_eq_const
    )

    # define the points to be included in the γ, θ domain
    points_to_include_2d = np.array([
        [-0.8, pi/4],
        [-1.5, pi/4],
        [-0.8, -pi/4.5],
        [-1.1, -pi/4.5],
        [0.5, -pi/3.5],
        [0.5, -pi/3],
        [1.5, -pi/3],
        [0.9, -pi/3],
        [0.9, pi/6],
        [1.8, -pi/8],
        [2, pi/2.5],
        [-2, pi/8],
        [0.9, pi/4],
        [1.8, pi/6]
    ])
    points_to_include = original_to_extended_state_space(
        input_points=points_to_include_2d
        )
    points_inlusion_weights = np.ones(points_to_include.shape[0])
    
    # synthesize the clf
    V_result = clf_synthesis.bilinear_alternation(
        clf_init=V_init,
        rho=rho,
        kappaV=kappaV+kappa_diff,
        ball_radius=epsilon_0,
        ball_inclusion_ball_x_degree=2,
        ball_inclusion_h_x_degree=2,
        clf_lagrangain_lambda_y_x_degree=[2, 2],
        clf_lagrangain_xi_y_x_degree=2,
        clf_lagrangain_rho_minus_V_x_degree=2,
        V_x_degree=2,
        included_points=points_to_include,
        points_inclusion_weights=points_inlusion_weights,
        state_eq_constraints_x_degree=[2],
        anchor_points=None,
        anchor_bounds=None,
        max_iter=20,
        lagrangian_coeff_tol=1e-3,
    )
    assert V_result is not None

    cbf_synthesis_given_clf = UnionCbfSynthesisGivenClf(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        clf=V_result,
        rho=rho,
        num_cbf=2,
        unsafe_polys=unsafe_polys,
        Au=Au,
        bu=Au,
        state_eq_constraints=state_eq_const,
        kappaV=kappaV,
        kappah=kappah,
        epsilon_0=1e-6,
        epsilon=1e-6,
        cbf_x_degrees=[2, 2],
        cbf_ball_inclusion_ball_x_degree=2,
        cbf_ball_inclusion_cbf_x_degree=2,
        compatible_lambda_y_x_degrees=[2, 2],
        compatible_xi_y_x_degree=2,
        compatible_rho_minus_V_x_degree=2,
        compatible_deact_cbf_x_degree=[2],
        compatible_h_x_degree=2,
        state_eq_x_degrees=[2],
        safety_h_x_degree=2,
        safety_unsafe_polys_x_degree=[2, 2]
    )

    back_off_scales = [BackoffScale(rel=0.00, abs=None)] * 10
    back_off_scales[8] = BackoffScale(rel=0.02, abs=None)

    cbf_result = cbf_synthesis_given_clf.synthesis_first_cbf(
        cbf_init=sym.Polynomial(0.01 - x.dot(x)),
        points_to_include=points_to_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=10,
        back_off_scale=back_off_scales

    )
    assert cbf_result is not None

    save_synthesis_results(
        V_res=V_result,
        h_res=np.array([cbf_result]),
        x_set=x_set,
        pickle_path=get_pkl_file_path("non_linear_toy_clf_single_cbf_synthesized.pkl")
    )




if __name__ == "__main__":
    main()






