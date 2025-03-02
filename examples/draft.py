import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.clf_union_cbf import(
    CompatibilityInRangeLagragianDegrees,
    BallInclusionLagrangianDegree,
    StepOne,
    SubsetVerifyLagrangianDegree,
    CompatibleClfCbfInSubset
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
    
    compatibility_in_subset_lagrangian_degrees = SubsetVerifyLagrangianDegree(
        lambda_y=[
            [Degree(x=2, y=0, c=2), Degree(x=2, y=0, c=2)],
        ],
        xi_y=[
            Degree(x=2, y=0, c=2),
        ],
        activated_cbfs=[
            Degree(x=2, y=2, c=2),
        ],
        clf = Degree(x=2, y=2, c=2),
        deactivated_cbfs=[
            Degree(x=2, y=2, c=2)
        ],
        ball = Degree(x=2, y=2, c=2),
        state_eq_constraints=None
    )

    compatibility_object = CompatibleClfCbfInSubset(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        subset=check_subset,
        Au = None,
        bu = None,
        state_eq_constraints=None,
    )

    (
        compatibility_in_subset_lagrangians
    ) = compatibility_object.verify_compatibility_in_subset(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        epsilon = 0.1,
        lagrangian_degrees=compatibility_in_subset_lagrangian_degrees
    )

    assert compatibility_in_subset_lagrangians is not None
    print("Test Passed")
    



if __name__ == "__main__":
    main()
