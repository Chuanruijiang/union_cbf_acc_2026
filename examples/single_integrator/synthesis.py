import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.clf import(
    ClfSynthesis
)
from dynamics import system_dynamics

def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = system_dynamics()
    V_init = sym.Polynomial(x[0]**2 + x[1]**2)*0.25

    # Environment parameters
    rho = 1
    kappaV = 0.8
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
        bu=None
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
        included_points=points_to_include,
        points_inclusion_weights=points_inlusion_weights,
        anchor_points=None,
        anchor_bounds=None,
        max_iter=25,
        lagrangian_coeff_tol=1e-3,
    )

    assert V_result is not None

if __name__ == "__main__":
    main()




    
        
