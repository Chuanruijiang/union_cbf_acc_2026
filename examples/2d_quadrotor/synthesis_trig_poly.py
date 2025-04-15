import os
import sys
import os.path
import pickle

import numpy as np
import pydrake.symbolic as sym

from compatible_clf_union_cbf.utils import (
    serialize_polynomial,
    deserialize_polynomial,
    BackoffScale,
)
from compatible_clf_union_cbf.clf import ClfSynthesis
from dynamics import (
    Quadrotor2dTrigPlant
)

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
    x = sym.MakeVectorContinuousVariable(7, "x")
    x_set = sym.Variables(x)

    data = load_clf(
        pickle_path=get_pkl_file_path(name_file="2d_quadrotor_clf_init_trig.pkl"),
        x_set=x_set,
    )
    V_init = data["V"]

    quadrotor = Quadrotor2dTrigPlant()
    f, g = quadrotor.affine_dynamics(x)
    state_eq_const = quadrotor.equality_constraint(x)

    # (Au, bu) = control_limits(x)
    (Au, bu) = None, None

    # set parameters:
    epsilon_0 = 0.1
    rho = 1
    kappaV = 0.05
    kappa_diff = 0.01

    clf_synthesis = ClfSynthesis(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        Au=Au,
        bu=bu,
        state_eq_constraint=state_eq_const
    )

    points_to_include_original_space = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [-1, 2, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0],
            [1, -2, 0, 0, 0, 0],
        ]
    )
    points_to_include = quadrotor.to_trig_state(
        points_to_include_original_space
    )
    points_inlusion_weights = np.ones(
        points_to_include.shape[0],
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
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
        BackoffScale(rel=0.01, abs=None),
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
        points_inclusion_weights=points_inlusion_weights,
        state_eq_constraints_x_degree=[2],
        anchor_points=None,
        anchor_bounds=None,
        max_iter=20,
        backoff_scale=back_off_scale,
    )
    assert V_result is not None


if __name__ == "__main__":
    main()
