"""
In this script, we wanted to investigate the average iteration time
for bilinear alternation of synthesizing each CBFs. We wanted to know,
as the number of CBFs grouws, will the synthesis time for each CBF also
grow? 
In this example, we set the obstacle as: (x0 -(-1))^2 + (x1)^2 ≤ 0.5^2
the CBFs are initialized as:
    cbf1 = p1ᵀx + 0.15 where p1 = [ 1, 0]
    cbf2 = p2ᵀx - 5   where p2 = [ 1, 1]
    cbf3 = p3ᵀx - 5   where p3 = [ 0, 1]
    cbf4 = p4ᵀx - 5   where p4 = [-1, 1]
if the dim is higher than 2, we just add 0 to the p1, p2, p3, p4. So
that, in the higher dimensional space, the obstacle is a surface of a
hyper-pillar.
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
    return np.array([
            sym.Polynomial((0.5)**2 - (x[0] + 1)**2 - x[1]**2)
    ])


def initialize_cbf(x: np.ndarray, dim: int) -> np.ndarray:
    assert dim >= 2 and dim <= 4
    assert x.shape[0] == dim
    if dim == 2:
        p1 = np.array([1, 0])
        p2 = np.array([1, 1])
        p3 = np.array([0, 1])
        p4 = np.array([-1, 1])
    else:
        p1 = np.array([1, 0] + [0]*(dim-2))
        p2 = np.array([1, 1] + [0]*(dim-2))
        p3 = np.array([0, 1] + [0]*(dim-2))
        p4 = np.array([-1, 1] + [0]*(dim-2))

    return np.array([
        sym.Polynomial(p1.dot(x) + 0.15),
        sym.Polynomial(p2.dot(x) - 5),
        sym.Polynomial(p3.dot(x) - 5),
        sym.Polynomial(p4.dot(x) - 5)
    ])


def points_to_include(dim: int) -> np.ndarray:
    assert dim >= 2 and dim <= 4
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
        [-2, 1.1],
        [-3, 1],
        [-2.5, 1.5],
        [-4, -0.5]
    ])
    if dim == 2:
        assert points.shape[1] == dim
        points_include = points
    else:
        points_include = np.zeros((points.shape[0], dim))
        points_include[:, 0:2] = points
    return points_include


def main(dim: int):
    assert dim >= 2 and dim <= 4, "dim should be in [2,4]"

    x = sym.MakeVectorContinuousVariable(dim, "x")
    high_dementional_system = HighDementionalDyanmics(dim)
    (f, g) = high_dementional_system.affine_dynamics(x)
    (Au, bu) = (None, None)
    V_init = sym.Polynomial(x.dot(x))

    # necessary parameters
    epsilon0 = 0.1
    kappaV = 0.005
    kappaV_diff = 0.001
    kappa_h = [1.0, 1.0, 1.0, 1.0]
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
        num_cbf=4,
        unsafe_polys=unsafe_polys,
        Au=Au,
        bu=bu,
        state_eq_constraints=None,
        kappaV=kappaV,
        kappah=kappa_h,
        epsilon_0=epsilon0,
        epsilon=epsilon,
        cbf_x_degrees=[1, 1, 1, 1],
        cbf_ball_inclusion_ball_x_degree=2,
        cbf_ball_inclusion_cbf_x_degree=2,
        compatible_lambda_y_x_degrees=[1]*dim,
        compatible_xi_y_x_degree=2,
        compatible_rho_minus_V_x_degree=2,
        compatible_h_x_degree=2,
        compatible_deact_cbf_x_degree=[2, 2, 2],
        state_eq_x_degrees=None,
        safety_h_x_degree=2,
        safety_unsafe_polys_x_degree=[2]
    )
    # synthesis the first cbf:
    iter_num = 5
    time_start = time.time()
    cbf_1_result = cbf_synthesis_given_clf.synthesis_first_cbf(
        cbf_init=cbf_inits[0],
        points_to_include=points_include,
        weights_to_include=points_inlcude_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=iter_num,
    )
    time_end = time.time()
    assert cbf_1_result is not None
    cbf_1_synthesis_time = time_end - time_start
    print(f"Time taken for CBF 1 synthesis: {cbf_1_synthesis_time:.2f} seconds")
    print(f"Average iteration time for CBF 1 synthesis: {cbf_1_synthesis_time/iter_num:.2f} seconds")

    (
        points_include,
        points_inlusion_weights,
    ) = cbf_synthesis_given_clf.remove_included_points(
        included_points=points_include,
        points_inclusion_weights=points_inlcude_weights,
        solved_cbf=cbf_1_result,
    )

    # synthesize the second CBF:
    iter_num = 5
    time_start = time.time()
    cbf_2_result = cbf_synthesis_given_clf.synthesis_other_cbf(
        cbf_init=cbf_inits[1],
        cbf_index=1,
        deact_cbfs=np.array([cbf_1_result]),
        points_to_include=points_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=iter_num,
        back_off_scale=None,
    )
    time_end = time.time()
    assert cbf_2_result is not None
    cbf_2_synthesis_time = time_end - time_start
    print(f"Time taken for CBF 2 synthesis: {cbf_2_synthesis_time:.2f} seconds")
    print(f"Average iteration time for CBF 2 synthesis: {cbf_2_synthesis_time/iter_num:.2f} seconds")

    (
        points_include,
        points_inlusion_weights,
    ) = cbf_synthesis_given_clf.remove_included_points(
        included_points=points_include,
        points_inclusion_weights=points_inlcude_weights,
        solved_cbf=cbf_2_result,
    )

    # synthesize the third CBF:
    iter_num = 5
    time_start = time.time()
    cbf_3_result = cbf_synthesis_given_clf.synthesis_other_cbf(
        cbf_init=cbf_inits[2],
        cbf_index=2,
        deact_cbfs=np.array([cbf_1_result, cbf_2_result]),
        points_to_include=points_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=iter_num,
        back_off_scale=None,
    )
    time_end = time.time()
    assert cbf_3_result is not None
    cbf_3_synthesis_time = time_end - time_start
    print(f"Time taken for CBF 3 synthesis: {cbf_3_synthesis_time:.2f} seconds")
    print(f"Average iteration time for CBF 3 synthesis: {cbf_3_synthesis_time/iter_num:.2f} seconds")

    # synthesize the fourth CBF:
    (
        points_include,
        points_inlusion_weights,
    ) = cbf_synthesis_given_clf.remove_included_points(
        included_points=points_include,
        points_inclusion_weights=points_inlcude_weights,
        solved_cbf=cbf_2_result,
    )

    # synthesize the third CBF:
    iter_num = 5
    time_start = time.time()
    cbf_4_result = cbf_synthesis_given_clf.synthesis_other_cbf(
        cbf_init=cbf_inits[3],
        cbf_index=3,
        deact_cbfs=np.array([cbf_1_result, cbf_2_result, cbf_3_result]),
        points_to_include=points_include,
        weights_to_include=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=iter_num,
        back_off_scale=None,
    )
    time_end = time.time()
    assert cbf_4_result is not None
    cbf_4_synthesis_time = time_end - time_start
    print(f"Time taken for CBF 4 synthesis: {cbf_4_synthesis_time:.2f} seconds")
    print(f"Average iteration time for CBF 4 synthesis: {cbf_4_synthesis_time/iter_num:.2f} seconds")


if __name__ == "__main__":
    main(dim=4)
