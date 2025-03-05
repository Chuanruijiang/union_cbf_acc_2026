import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.clf_union_cbf import(
    CompatibleClfUnionCbfs,
    StepOne,
    StepTwo
)


def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(),
        sym.Polynomial()
    ])
    g = np.array([
        [sym.Polynomial(1), sym.Polynomial()],
        [sym.Polynomial(), sym.Polynomial(1)],
    ])
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
    
    kappa_V = 0.1
    kappa_h = [1.0, 1.0]

    check_step_two = StepTwo(
        clf=rho - V,
        cbfs=h,
        ball=sym.Polynomial(1 - x.dot(x)),
        sys_dyn_f=f,
        sys_dyn_g=g,
        x=x,
        r_start=0.1,
        r_lower_bound=0.01,
        state_eq_constraints=None,
        Au=None,
        bu=None
    )
    epsilon = check_step_two.simplified_step_two_verification(
        activated_cbf_x_degree=2,
        lambda_y_x_degrees=[2, 2],
        xi_y_x_degree=2,
        deactivated_cbfs_common_degree=2,
        ball_x_degree=2,
        clf_x_degree=2,
        state_eq_x_degrees=None,
        rho=rho,
        kappa_V=kappa_V,
        kappa_h=kappa_h
    )

    assert epsilon is not None
    print("Test Passed")
    



if __name__ == "__main__":
    main()
