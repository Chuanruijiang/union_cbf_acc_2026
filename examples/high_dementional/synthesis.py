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


import numpy as np
import time
import pydrake.symbolic as sym
from union_cbf_base.utils import (
    BackoffScale,
    compute_minimum_on_boundary
)
from union_cbf_base.clf import ClfSynthesis
from union_cbf_base.union_cbf import (
    UnionCbfSynthesisGivenClf
)
from dynamics import HighDementionalDyanmics


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


def initialize_cbf(x: np.ndarray, dim: int) -> np.ndarray:
    assert x.shape[0] == dim
    if dim % 2 == 0:
        p1 = np.array([1]*(int(dim/2)) + [0]*(int(dim/2)))
        p2 = np.array([0]*(int(dim/2)) + [1]*(int(dim/2)))
    else:
        p1 = np.array([1]*(int((dim+1)/2)) + [0]*(int((dim-1)/2)))
        p2 = np.array([0]*(int((dim+1)/2)) + [1]*(int((dim-1)/2)))

    return np.array([
        sym.Polynomial(0.5 + p1.dot(x)),
        sym.Polynomial(-3 + p2.dot(x))
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
    time_start = time.time()
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
    time_end = time.time()
    clf_synthesis_time = time_end - time_start
    print(f"Time taken for CLF synthesis: {clf_synthesis_time:.2f} seconds")
    assert V_result is not None

    # compute the V_min and the epsilon:
    V_min = compute_minimum_on_boundary(
        x=x, p=V_result, q=sym.Polynomial(x.dot(x)-epsilon0**2)
    )
    epsilon = V_min * kappaV_diff
    cbf_inits = initialize_cbf(x, dim)

    # synthesize the first CBF:
    cbf_synthesis_given_clf = UnionCbfSynthesisGivenClf(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        clf=V_result,
        rho=1,
        num_cbf=2,
        unsafe_polys=unsafe_polys,
        Au=Au,
        bu=bu,
        state_eq_constraints=None,
        kappaV=kappaV,
        kappah=kappa_h,
        epsilon_0=epsilon0,
        epsilon=epsilon,
        cbf_x_degrees=[1, 1],
        cbf_ball_inclusion_ball_x_degree=2,
        cbf_ball_inclusion_cbf_x_degree=2,
        compatible_lambda_y_x_degrees=[1]*dim,
        compatible_xi_y_x_degree=2,
        compatible_rho_minus_V_x_degree=2,
        compatible_h_x_degree=2,
        compatible_deact_cbf_x_degree=[2],
        state_eq_x_degrees=None,
        safety_h_x_degree=2,
        safety_unsafe_polys_x_degree=[2, 2]
    )
    # synthesis the first cbf:
    time_start = time.time()
    cbf_1_result = cbf_synthesis_given_clf.synthesis_first_cbf(
        cbf_init=cbf_inits[0],
        points_to_include=points_include,
        weights_to_include=points_inlcude_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=5,
    )
    time_end = time.time()
    cbf_1_synthesis_time = time_end - time_start
    print(f"Time taken for CBF 1 synthesis: {cbf_1_synthesis_time:.2f} seconds")
    assert cbf_1_result is not None

    (
        points_include,
        points_inlusion_weights,
    ) = cbf_synthesis_given_clf.remove_included_points(
        included_points=points_include,
        points_inclusion_weights=points_inlcude_weights,
        solved_cbf=cbf_1_result,
    )

    # synthesize the second CBF:
    time_start = time.time()
    cbf_2_result = cbf_synthesis_given_clf.synthesis_other_cbf(
        cbf_init=cbf_inits[1],
        cbf_index=1,
        deact_cbfs=np.array([cbf_1_result]),
        points_to_include=points_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=5,
        back_off_scale=None,
    )
    time_end = time.time()
    cbf_2_synthesis_time = time_end - time_start
    print(f"Time taken for CBF 2 synthesis: {cbf_2_synthesis_time:.2f} seconds")
    assert cbf_2_result is not None

    total_synthesis_time = (
        clf_synthesis_time + cbf_1_synthesis_time + cbf_2_synthesis_time
    )
    print(f"Total synthesis time: {total_synthesis_time:.2f} seconds")


if __name__ == "__main__":
    main(dim=6)
