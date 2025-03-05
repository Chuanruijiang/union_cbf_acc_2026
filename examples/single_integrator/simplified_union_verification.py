import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.clf_union_cbf import(
    CompatibleClfUnionCbfs
)
from dynamics import system_dynamics

def main():
    # initialize system dynamics.
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = system_dynamics()
    # Define the CLF and CBFs
    V = sym.Polynomial(x[0]**2 + x[1]**2)
    rho = 49
    cbf_center = np.array([
        [3, 0],
        [-5, 0]
    ])
    cbf_radiuses = np.array([5, 5])
    h = np.array([
        sym.Polynomial(cbf_radiuses[i]**2 - (x - cbf_center[i]).dot(x - cbf_center[i]))
        for i in range(cbf_center.shape[0])
    ])
    # Define the parameters
    kappa_V = 0.1
    kappa_h = [1.0, 1.0]

    compatible_object = CompatibleClfUnionCbfs(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        clf=V,
        cbfs=h,
        state_eq_constraints=None,
        Au=None,
        bu=None
    )

    simplified_compatibility = compatible_object.simplified_union_verification(
        epsilon0_start=1,
        epsilon0_lower_bound=0.1,
        epsilon_start=0.1,
        epsilon_lower_bound=0.01,
        kappa_V=kappa_V,
        rho=rho,
        kappa_h=kappa_h,
        ball_inclusion_ball_x_degree=2,
        ball_inclusion_cbf_x_degree=2,
        qp_feasible_in_ball_lambda_y_x_degree=2,
        qp_feasible_in_ball_xi_y_x_degree=2,
        qp_feasible_in_ball_ball_x_degree=2,
        qp_feasible_in_ball_state_eq_x_degree=None,
        activated_cbf_x_degree=2,
        lambda_y_x_degrees=[2, 2],
        xi_y_x_degree=2,
        deactivated_cbfs_common_degree=2,
        step_two_ball_x_degree=2,
        clf_x_degree=2,
        state_eq_x_degrees=None
    )

    assert simplified_compatibility
    print("Simplified union verification is successful.")

if __name__ == "__main__":
    main()

    