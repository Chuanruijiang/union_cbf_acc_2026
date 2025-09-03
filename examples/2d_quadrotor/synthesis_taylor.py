import os
import sys
import os.path
import pickle

import numpy as np
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    serialize_polynomial,
    deserialize_polynomial,
    compute_minimum_on_boundary,
    BackoffScale,
)
# from compatible_clf_union_cbf.clf import ClfSynthesis
from union_cbf_base.union_cbf import UnionCbfSynthesisGivenClf
from union_cbf_base.clf import ClfSynthesis
from dynamics import Quadrotor2dPlant


sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../.."))


def get_pkl_file_path(name_file: str):
    filename = name_file
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path


def load_clf(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    ret["V"] = deserialize_polynomial(data["V"], x_set)
    return ret


def save_synthesized_clf(
    V: sym.Polynomial,
    x_set: sym.Variables,
    pickle_path: str,
):
    """
    Save the CLF and CBF to a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}

    data["V"] = serialize_polynomial(V, x_set)

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


def main():
    x = sym.MakeVectorContinuousVariable(6, "x")
    x_set = sym.Variables(x)

    data = load_clf(
        pickle_path=get_pkl_file_path(name_file="2d_quadrotor_clf_init_taylor.pkl"),
        x_set=x_set,
    )
    V_init = data["V"]

    quadrotor = Quadrotor2dPlant()
    f, g = quadrotor.taylor_affine_dynamics(x)

    # (Au, bu) = control_limits(x)
    (Au, bu) = None, None

    # set parameters:
    epsilon_0 = 0.1
    rho = 1
    kappaV = 0.005
    kappa_diff = 0.001
    kappah = [1.0, 1.0]

    # unsafe Region:
    unsafe_polys = np.array(
        [sym.Polynomial(x[0] - 1.25), sym.Polynomial(-x[1] + 1.5)]
    )

    # define the points to be included
    points_to_include = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [-1, 2, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0],
            [1, -2, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1.8, 1.75, 0, 0, 0, 0],
            [1.8, 1.7, 0, 0, 0, 0],
            [1.7, 1.6, 0, 0, 0, 0],
            [-0.5, 0.5, 0, 0, 0, 0],
            [-0.5, -0.5, 0, 0, 0, 0],
            [1.5, 2, 0, 0, 0, 0],
            [0.5, 1.5, 0, 0, 0, 0],
        ]
    )
    points_inclusion_weights = np.ones(
        points_to_include.shape[0],
    )

    # synthesize the clf
    clf_synthesis = ClfSynthesis(
        x=x, sys_dyn_f=f, sys_dyn_g=g, Au=Au, bu=bu, state_eq_constraint=None
    )
    back_off_scale = [
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None)
    ]
    V_result = clf_synthesis.bilinear_alternation(
        clf_init=V_init,
        rho=rho,
        kappaV=kappaV + kappa_diff,
        ball_radius=epsilon_0,
        ball_inclusion_ball_x_degree=2,
        ball_inclusion_h_x_degree=2,
        clf_lagrangain_lambda_y_x_degree=[2, 2],
        clf_lagrangain_xi_y_x_degree=2,
        clf_lagrangain_rho_minus_V_x_degree=2,
        V_x_degree=2,
        included_points=points_to_include,
        points_inclusion_weights=points_inclusion_weights,
        state_eq_constraints_x_degree=[2],
        anchor_points=None,
        anchor_bounds=None,
        max_iter=10,
        backoff_scale=back_off_scale,
    )
    assert V_result is not None
    
    # # save the synthesized CLF
    # save_synthesized_clf(
    #     V=V_result,
    #     x_set=x_set,
    #     pickle_path=get_pkl_file_path(
    #         name_file="2d_quadrotor_clf_synthesized_taylor.pkl"
    #         ),
    # )

    # # load synthesized CLF:
    # data = load_clf(
    #     pickle_path=get_pkl_file_path(name_file="2d_quadrotor_clf_synthesized_taylor.pkl"),
    #     x_set=x_set,
    # )
    # V_result = data["V"]

    # # compute epsilon for CBF synthesis:
    # V_min = compute_minimum_on_boundary(
    #     x=x, p=V_result, q=sym.Polynomial(epsilon_0**2 - x.dot(x))
    # )
    # print(V_min)
    # epsilon = kappa_diff * V_min
    # print(f"We use this epsilon for the following CBF synthesis: {epsilon}")

    # # CBF synthesis:
    # cbf_synthesis_given_clf = UnionCbfSynthesisGivenClf(
    #     x=x,
    #     sys_dyn_f=f,
    #     sys_dyn_g=g,
    #     clf=V_result,
    #     rho=rho,
    #     num_cbf=2,
    #     unsafe_polys=unsafe_polys,
    #     Au=Au,
    #     bu=Au,
    #     state_eq_constraints=None,
    #     kappaV=kappaV,
    #     kappah=kappah,
    #     epsilon_0=epsilon_0,
    #     epsilon=1e-10,
    #     cbf_x_degrees=[1, 1],
    #     cbf_ball_inclusion_ball_x_degree=2,
    #     cbf_ball_inclusion_cbf_x_degree=2,
    #     compatible_lambda_y_x_degrees=[2, 2],
    #     compatible_xi_y_x_degree=2,
    #     compatible_rho_minus_V_x_degree=2,
    #     compatible_deact_cbf_x_degree=[2],
    #     compatible_h_x_degree=2,
    #     state_eq_x_degrees=None,
    #     safety_h_x_degree=2,
    #     safety_unsafe_polys_x_degree=[2, 2],
    # )

    # # synthesis the first cbf:
    # cbf_1_result = cbf_synthesis_given_clf.synthesis_first_cbf(
    #     cbf_init=sym.Polynomial(-x[0] - x[4] + 0.1),
    #     points_to_include=points_to_include,
    #     weights_to_include=np.ones(points_to_include.shape[0]),
    #     anchor_points=None,
    #     anchor_bounds=None,
    #     max_iter=5,
    # )
    # assert cbf_1_result is not None



if __name__ == "__main__":
    main()
