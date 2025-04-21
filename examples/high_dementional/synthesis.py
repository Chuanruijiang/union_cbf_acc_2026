"""
In this script, we will synthesize a CLF and union of 2 CBFs for high
dementional systems with differnt dimensions.
We start from dim=2, 3,...8
For each dimension case, we design the unsafe region as:
X_u = {x | p1(x)≥0, p2(x)≥0}
where p1(x) = -1 - p1ᵀx, p2(x) = 1 - p2ᵀx, and p1ᵀp2 = 0 meaning that
p1(x) and p2(x) are orthogonal hyperplanes in the state space.
We initialize the CLF using cost to go function of LQR, and initilize
the CBFs as p1ᵀx + 0.1 and p2ᵀx - 5.
For different dimensions, p1 and p2 are different:
dim=2: p1 = [1, 0], p2 = [0, 1]
dim=3: p1 = [1, 1, 0], p2 = [0, 0, 1]
dim=4: p1 = [1, 1, 0, 0], p2 = [0, 0, 1, 1]
dim=5: p1 = [1, 1, 1, 0, 0], p2 = [0, 0, 0, 1, 1]
dim=6: p1 = [1, 1, 1, 0, 0, 0], p2 = [0, 0, 0, 1, 1, 1]
dim=7: p1 = [1, 1, 1, 1, 0, 0, 0], p2 = [0, 0, 0, 0, 1, 1, 1]
dim=8: p1 = [1, 1, 1, 1, 0, 0, 0, 0], p2 = [0, 0, 0, 0, 1, 1, 1, 1]
We also record the computation time for each dimension case.
"""

import os.path
import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import (
    BackoffScale,
    serialize_polynomial,
    deserialize_polynomial,
)
from compatible_clf_union_cbf.clf import ClfSynthesis
from compatible_clf_union_cbf.union_cbf import (
    UnionCbfSynthesisGivenClf
)
from dynamics import HighDementionalDyanmics


def get_pkl_file_path(filename: str):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path


def save_synthesized_clf(
    V: list[sym.Polynomial],
    x_sets: list[sym.Variables],
    pickle_path: str,
):
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}
    for i in range(len(V)):
        data["V"+str(i)] = serialize_polynomial(V[i], x_sets[i])

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


def load_synthesized_clf(pickle_path: str):
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    V = []
    x_sets = []
    for i in range(len(data)):
        V.append(deserialize_polynomial(data["V"+str(i)]))
        x_sets.append(sym.Variables(V[i].GetVariables()))
    return V, x_sets


def get_unsafe_region(x: np.ndarray, dim: int) -> np.ndarray:
    assert x.shape[0] == dim
    if dim % 2 == 0:
        p1 = np.array([1]*(int(dim/2)) + [0]*(int(dim/2)))
        p2 = np.array([0]*(int(dim/2)) + [1]*(int(dim/2)))
    else:
        p1 = np.array([1]*(int((dim+1)/2)) + [0]*(int((dim-1)/2)))
        p2 = np.array([0]*(int((dim+1)/2)) + [1]*(int((dim-1)/2)))

    return np.array([
        sym.Polynomial(-1 - p1.dot(x)),
        sym.Polynomial(1 - p2.dot(x))
    ])


def points_to_include(dim: int) -> np.ndarray:
    points = np.array([
        [0, 0],
        [0, 1],
        [-0.5, 0.5],
        [-0.5, -0.5],
        [0.6, -0.5],
        [0.4, 0.5],
        [0.4, 1.2],
        [1.3, 1.1],
        [2, 0.3],
        [-0.5, 1.8],
        [-1, 1.5],
        [-1.3, 1.5],
        [-2, 1.3],
        [-1.5, 1.8],
        [-2.1, 2],
    ])
    if points.shape[1] == dim:
            points_include = points
    else:
        points_include = np.zeros((points.shape[0], dim))
    if dim % 2 == 0:
        p1 = np.array([1]*(int(dim/2)) + [0]*(int(dim/2)))
        p2 = np.array([0]*(int(dim/2)) + [1]*(int(dim/2)))
        points_include[:, 0:int(dim/2)] = np.concatenate(
            [
                (points[:, 0]/int(dim/2)).reshape((points.shape[0], 1))
            ]*int(dim/2),
            axis=1
        )
        points_include[:, int(dim/2):] = np.concatenate(
            [
                (points[:, 1]/int(dim/2)).reshape((points.shape[0], 1))
            ]*int(dim/2),
            axis=1
        )
    else:
        p1 = np.array([1]*(int((dim+1)/2)) + [0]*(int((dim-1)/2)))
        p2 = np.array([0]*(int((dim+1)/2)) + [1]*(int((dim-1)/2)))
        points_include[:, 0:int((dim+1)/2)] = np.concatenate(
            [
                (points[:, 0]/int((dim+1)/2)).reshape((points.shape[0], 1))
            ]*int((dim+1)/2), 
            axis=1
        )
        points_include[:, int((dim+1)/2):] = np.concatenate(
            [
                (points[:, 1]/int((dim-1)/2)).reshape((points.shape[0], 1))
            ]*int((dim-1)/2), 
            axis=1
        )
    return points_include


def main(dim: int):
    assert dim >= 2 and dim <= 8, "dim should be in [2,8]"

    x = sym.MakeVectorContinuousVariable(dim, "x")
    high_dementional_system = HighDementionalDyanmics(dim)
    (f, g) = high_dementional_system.affine_dynamics(x)
    (Au, bu) = (None, None)
    V_init = sym.Polynomial(x.dot(x))

    # necessary parameters
    epsilon0 = 0.1
    kappaV = 0.005
    kappaV_diff = 0.001
    kappa_h = [1.0, 1.0]
    # unsafe region
    unsafe_polys = get_unsafe_region(x, dim)

    # specify the points to be included in the state space:
    points_include = points_to_include(dim)
    points_inlcude_weights = np.ones(points_include.shape[0])

    # synthesize the CLF:
    clf_synthesis = ClfSynthesis(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        Au=Au,
        bu=bu,
        state_eq_constraint=None,
    )
    V_result = clf_synthesis.bilinear_alternation(
        clf_init=V_init,
        rho=1,
        kappaV=kappaV + kappaV_diff,
        ball_radius=epsilon0,
        ball_inclusion_ball_x_degree=2,
        ball_inclusion_h_x_degree=2,
        clf_lagrangain_lambda_y_x_degree=[1]*dim,
        clf_lagrangain_xi_y_x_degree=2,
        clf_lagrangain_rho_minus_V_x_degree=2,
        V_x_degree=2,
        included_points=points_include,
        points_inclusion_weights=points_inlcude_weights,
        state_eq_constraints_x_degree=None,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=5,
        lagrangian_coeff_tol=1e-3,
        backoff_scale=None,
    )
    assert V_result is not None
    
    # save the synthesized CLF if needed:


    # synthesize the first CBF:




if __name__ == "__main__":
    main(dim=7)
