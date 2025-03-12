import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import(
    compute_minimum_on_boundary
)
from compatible_clf_union_cbf.clf import(
    ClfSynthesis
)
from compatible_clf_union_cbf.union_cbf import(
    UnionCbfSynthesisGivenClf
)
from dynamics import system_dynamics

def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = system_dynamics()
    V_init = sym.Polynomial(x[0]**2 + x[1]**2)*0.25

    # Environment parameters
    rho = 1
    expected_kappaV = 0.2
    kappaV = 0.8
    kappa_diff = kappaV - expected_kappaV
    
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
        [-8, 1],
        [-8, 2.5],
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
        Au=None,
        bu=None,
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

    # compute the minimum value of V on the boundary of the ball(ϵ_0):
    V_min = compute_minimum_on_boundary(
        x=x,
        p=V_result,
        q=sym.Polynomial(epsilon_0**2 - x.dot(x))
    )
    print(V_min)
    epsilon = kappa_diff*V_min
    print(f"We use this epsilon for the following CBF synthesis: {epsilon}")

    # CBF synthesis:
    cbf_synthesis_given_clf = UnionCbfSynthesisGivenClf(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        clf=V_result,
        rho=rho,
        num_cbf=3,
        unsafe_polys=unsafe_polys,
        Au=None,
        bu=None,
        state_eq_constraints=None,
        kappaV=kappaV,
        kappah=[1, 1, 1],
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
        max_iter=20,
    )

    assert cbf_1_result is not None



if __name__ == "__main__":
    main()




    
        
