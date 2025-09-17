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

from union_cbf_base.union_cbf import UnionCbf
from union_cbf_base.non_empty_subset import Subset
from dynamics import NonlinearToyPlant

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
    system_obj = NonlinearToyPlant()
    (f, g) = system_obj.affine_dynamics(x)
    state_eq_const = system_obj.state_eq_constraint(x)
    control_limits = system_obj.control_limits()

    # set parameters:
    eta = 1e-3
    epsilon = 0.01
    alpha = 0.1

    # unsafe Region:
    unsafe_polys = np.array([
        sym.Polynomial(x[0] - 0.3),
        sym.Polynomial(x[1] - np.sin(-pi/4))
    ])

    # # define the points to be included in the γ, θ domain
    # points_to_include_2d = np.array([
    #     [-0.8, pi/4],
    #     [-1.5, pi/4],
    #     [-0.8, -pi/4.5],
    #     [-1.1, -pi/4.5],
    #     [0.5, -pi/3.5],
    #     [0.5, -pi/3],
    #     [1.5, -pi/3],
    #     [0.9, -pi/3],
    #     [0.9, pi/6],
    #     [1.8, -pi/8],
    #     [2, pi/2.5],
    #     [-2, pi/8],
    #     [0.9, pi/4],
    #     [1.8, pi/6]
    # ])

    # we manucally pick the following 1-degree CBFs to form the union:
    cbfs = np.array([
        sym.Polynomial(x[0] - 0.35),
        sym.Polynomial(x[1] - np.sin(-pi/(4.5)))
    ])

    union_obj = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs,
        alpha=alpha,
        control_limits=control_limits,
        unsafe_polys=unsafe_polys,
        state_eq_const=state_eq_const,
    )
    cbf_valid = union_obj.validity_verification_of_all_cbfs(
        unsafe_poly_lagrangian_x_degrees=[2]*unsafe_polys.shape[0],
        cbf_lagrangian_x_degree=2
    )
    # union_feasible_thm2 = union_obj.verification_of_theorem_2(
    #     cbf_lagrangian_x_degree=2,
    #     cbf_lagrangian_y_degree=2,
    #     lambda_y_lagrangian_x_degree=2,
    #     lambda_y_lagrangian_y_degree=0,
    #     xi_y_lagrangian_x_degree=2,
    #     xi_y_lagrangian_y_degree=0,
    #     eta=eta,
    #     epsilon=epsilon,
    #     state_eq_lagrangian_x_degree=2,
    #     state_eq_lagrangian_y_degree=2
    # )
    subset_obj = Subset(
        x=x,
        cbfs=cbfs,
        activation_index=np.array([0, 1])
    )
    check_subset = union_obj.check_feasibility_in_subset(
        subset=subset_obj,
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=eta,
        epsilon=epsilon,
        state_eq_lagrangian_x_degree=2,
        state_eq_lagrangian_y_degree=2
    )
    assert check_subset, "The feasibility in the subset fails."
    union_feasible_thm3 = union_obj.verification_of_theorem_3(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=eta,
        epsilon=epsilon,
        state_eq_lagrangian_x_degree=2,
        state_eq_lagrangian_y_degree=2
    )

    assert cbf_valid, "One of the CBFs fails the validity verification."
    assert union_feasible_thm2, "The feasibility of theorem 2 fails."
    assert union_feasible_thm3, "The feasibility of theorem 3 fails."






if __name__ == "__main__":
    main()






