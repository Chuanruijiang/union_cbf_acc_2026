import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import os.path
import pickle

import numpy as np
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    compute_minimum_on_boundary,
    serialize_polynomial,
    deserialize_polynomial,
    BackoffScale,
)
from compatible_clf_cbf.clf_cbf import(
    CompatibleClfCbf,
    CompatibleStatesOptions,
    ExcludeSet,
    CompatibleLagrangianDegrees,
    SafetySetLagrangianDegrees,
    ExcludeRegionLagrangianDegrees,
    XYDegree
)

from dynamics import NonlinearToyPlant


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


def load_cbf(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    ret["h"] = deserialize_polynomial(data["h"], x_set)
    return ret


def save_synthesized_cbf(
    h: sym.Polynomial,
    x_set: sym.Variables,
    pickle_path: str,
):
    """
    Save the CLF and CBF to a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}

    data["h"] = serialize_polynomial(h, x_set)

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


def main():
    pi = np.pi
    nx = 3
    nu = 2
    x = sym.MakeVectorContinuousVariable(nx, "x")
    system_obj = NonlinearToyPlant()

    (f, g) = system_obj.affine_dynamics(x)
    state_eq_const = system_obj.state_eq_constraint(x)
    control_limits = system_obj.control_limits()

    # set parameters:
    alpha = 0.1

    # unsafe Region:
    unsafe_polys = np.array([
        sym.Polynomial(x[0] - 0.3),
        sym.Polynomial(x[1] - np.sin(-pi/4))
    ])
    exclude_set = ExcludeSet(l=unsafe_polys)

    # define the points to be included in the γ, θ domain
    points_to_include_2d = np.array(
        [
            [-0.8, pi / 4],
            [-1.5, pi / 4],
            [-0.8, -pi / 4.5],
            [-1.1, -pi / 4.5],
            [0.5, -pi / 3.5],
            [0.5, -pi / 3],
            [1.5, -pi / 3],
            [0.9, -pi / 3],
            [0.9, pi / 6],
            [1.8, -pi / 8],
            [2, pi / 2.5],
            [-2, pi / 8],
            [0.9, pi / 4],
            [1.8, pi / 6],
        ]
    )
    points_to_include = system_obj.original_to_extended_state_space(
        input_points=points_to_include_2d
    )
    points_inlusion_weights = np.array([1])
    points_inclusion_obj = CompatibleStatesOptions(
        candidate_compatible_states=points_to_include,
        anchor_states=None,
        h_anchor_bounds=None,
        weight_V=None,
        weight_h=points_inlusion_weights,
        relative_degrees=None,
        weight_lower_lie_derivatives=None,
        V_margin=None,
        h_margins=None
        )
    
    cbf_synthesis_obj = CompatibleClfCbf(
        f=f,
        g=g,
        x=x,
        exclude_sets=[exclude_set],
        within_set=None,
        Au=control_limits[0],
        bu=control_limits[1],
        with_clf=False,
        state_eq_constraints=state_eq_const
    )

    # specify lagrangian degrees:
    safety_lagrangian_degrees = SafetySetLagrangianDegrees(
        exclude=[
            ExcludeRegionLagrangianDegrees(
                cbf=[2], unsafe_region=[2,2], state_eq_constraints=[2]
            )
        ],
        within=[]
    )
    compatible_lagrangian_degrees = CompatibleLagrangianDegrees(
        lambda_y=[XYDegree(x=2, y=0)]*nu,
        xi_y=XYDegree(x=2, y=0),
        y=None,
        y_cross=None,
        rho_minus_V=None,
        h_plus_eps=[XYDegree(x=2, y=2)],
        lower_lie_derivative=None,
        state_eq_constraints=[XYDegree(x=2, y=2)]
    )

    (_, cbf_result) = cbf_synthesis_obj.bilinear_alternation(
        V_init=None,
        h_init=np.array([sym.Polynomial(0.01 - x[1]**2 - x[2]**2)]),
        compatible_lagrangian_degrees=compatible_lagrangian_degrees,
        safety_sets_lagrangian_degrees=safety_lagrangian_degrees,
        kappa_V=None,
        kappa_h=np.array([alpha]),
        barrier_eps=np.array([0]),
        x_equilibrium=None,
        clf_degree=None,
        cbf_degrees=[2],
        max_iter=10,
        compatible_states_options=points_inclusion_obj,
    )

    assert cbf_result is not None
    


if __name__ == "__main__":
    main()
