import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.clf_union_cbf import(
    CompatibilityInRangeLagragianDegrees,
    BallInclusionLagrangianDegree,
    StepOne,
    SubsetVerifyLagrangianDegree,
    CompatibleClfCbfInSubset,
    StepTwo
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

    check_object_step1 = StepOne(
        clf = V,
        cbfs = h,
        sys_dyn_f=f,
        sys_dyn_g=g,
        x=x,
        r_start=1,
        r_lower_bound=0.1
        )
    # specify compatibility lagragian degrees:
    compatibility_lagrangian_degree = CompatibilityInRangeLagragianDegrees(
        lambda_y=[
            Degree(x=2, y=0, c=0),
            Degree(x=2, y=0, c=0)
        ],
        xi_y=Degree(x=2, y=0, c=0),
        range_non_negative=[
            Degree(x=2, y=2, c=0)
        ],
        range_strictly_positive=None,
        state_eq_const=None
    )
    compatibility_lagrangian_degrees = [compatibility_lagrangian_degree] * h.shape[0]

    # specify the ball inclusion lagragian degrees:
    ball_inclusion_larangian_degree = BallInclusionLagrangianDegree(
        r_minus_xTx=Degree(x=2, y=0, c=0),
        h=Degree(x=2, y=0, c=0)
    )
    ball_inclusion_lagrangian_degrees = [ball_inclusion_larangian_degree] * h.shape[0]
    
    eps_0_out = check_object_step1.step_one_verification(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        ball_inclusion_lagrangian_degrees=ball_inclusion_lagrangian_degrees,
        compatibility_lagrangian_degrees=compatibility_lagrangian_degrees
    )
    assert eps_0_out is not None


    check_object_step2 = StepTwo(
        clf=rho - V,
        cbfs=h,
        ball=ball,
        sys_dyn_f=f,
        sys_dyn_g=g,
        x=x,
        r_start=0.1,
        r_lower_bound=0.01
    )

    eps_output = check_object_step2.step_two_verification(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        lagragian_x_degree=2,
        lagragian_y_degree=2,
        lagragian_c_degree=2,
    )

    assert eps_output is not None



    print("Test Passed")
    



if __name__ == "__main__":
    main()
