import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.clf_union_cbf import(
    CompatibilityInRangeLagragianDegrees,
    BallInclusionLagrangianDegree,
    StepOne,
    SubsetVerifyLagrangianDegree,
    CompatibleClfCbfInSubset,
    StepTwo,
    CompatibleClfUnionCbfs
)
from compatible_clf_union_cbf.union_cbf import(
    UnionSubset
)
from compatible_clf_union_cbf.utils import Degree


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
        [-5, 0],
        [3, 0]
    ])
    cbf_radiuses = np.array([5, 5])
    h = np.array([
        sym.Polynomial(cbf_radiuses[i]**2 - (x - cbf_center[i]).dot(x - cbf_center[i]))
        for i in range(cbf_center.shape[0])
    ])
    ball = sym.Polynomial(1 - x.dot(x))
    check_subset = UnionSubset(
        activated=np.array([rho-V, h[1]]),
        deactivated=np.array([h[0], ball]),
        variables=x
    )
    
    kappa_V = 0.1
    kappa_h = [1, 1]

    compatibile_object = CompatibleClfUnionCbfs(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        clf=V,
        cbfs=h,
        state_eq_constraints=None,
    )

    general_compatibility = compatibile_object.general_union_verification(
        epsilon0_start=1,
        epsilon0_lower_bound=0.1,
        epsilon_start=0.1,
        epsilon_lower_bound=0.01,
        kappa_V=kappa_V,
        rho=rho,
        kappa_h=kappa_h,
        ball_inclusion_ball_x_degrees=[2,2],
        ball_inclusion_cbf_x_degrees=[2,2],
        qp_feasible_in_ball_lambda_y_x_degrees=[2, 2],
        qp_feasible_in_ball_xi_y_x_degrees=[2, 2],
        qp_feasible_in_ball_ball_x_degrees=[2, 2],
        qp_feasible_in_ball_state_eq_x_degrees=None,
        compatible_in_subset_x_degree=2,
        compatible_in_subset_y_degree=2,
        compatible_in_subset_c_degree=2,
    )

    assert general_compatibility



    print("Test Passed")
    



if __name__ == "__main__":
    main()
