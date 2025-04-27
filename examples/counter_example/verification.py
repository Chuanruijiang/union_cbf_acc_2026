"""
In this script, we will verify a CLF and union of 4 CBFs for single integrator
to show that our compatible CLF and union of CBFs are less conservative.
In this example, we let the CLF to be V(x) = xᵀx, and the CBFs are:

h1(x) = x0 + x1 + 1
h2(x) = -x0 + x1 - 3
h3(x) = 0.5(x0 + 3)(x0 + 1) - x1

We first wanted to verify that the for all x in {x|h1(x)>=0}, h1(x) is compatible with the CLF.
then we verify that for all x in {x|h2(x)>=0, h1(x)<=0}, h2(x) is compatible with the CLF.
Then we verify that for all x in {x|h3(x)>=0, h2(x)<=0, h1(x)<=0}, h3(x) is compatible with
the CLF.
"""

import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import compute_minimum_on_boundary
from compatible_clf_union_cbf.union_cbf import UnionCbfSynthesisGivenClf
from dynamics import system_dynamics


def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = system_dynamics()
    V = sym.Polynomial(x.dot(x))
    h = np.array([
        sym.Polynomial(x[0] + x[1] + 1),
        sym.Polynomial(-x[0] + x[1] - 3),
        sym.Polynomial(0.5 * (x[0] + 3) * (x[0] + 1) - x[1]),
    ])
    unsafe_polys = np.array([
        sym.Polynomial((0.1)**2 - (x[0]-(-2))**2 - (x[1])**2),
    ])

    # Environment parameters
    epsilon_0 = 0.1
    rho = 25
    expected_kappaV = 0.1
    kappaV = 0.15
    kappa_diff = kappaV - expected_kappaV
    kappah = [1, 1, 1]

    # (Au, bu) = control_limits()
    (Au, bu) = (None, None)

    # compute the minimum value of V on the boundary of the ball(ϵ_0):
    V_min = compute_minimum_on_boundary(
        x=x, p=V, q=sym.Polynomial(epsilon_0**2 - x.dot(x))
    )
    print(V_min)
    epsilon = kappa_diff * V_min
    print(f"We use this epsilon for the following CBF synthesis: {epsilon}")

    # create union of CBFs object:
    cbf_synthesis_given_clf = UnionCbfSynthesisGivenClf(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        clf=V,
        rho=rho,
        num_cbf=3,
        unsafe_polys=unsafe_polys,
        Au=Au,
        bu=bu,
        state_eq_constraints=None,
        kappaV=kappaV,
        kappah=kappah,
        epsilon_0=epsilon_0,
        epsilon=epsilon,
        cbf_x_degrees=[1, 1, 2],
        cbf_ball_inclusion_ball_x_degree=2,
        cbf_ball_inclusion_cbf_x_degree=2,
        compatible_lambda_y_x_degrees=[2, 2],
        compatible_xi_y_x_degree=2,
        compatible_rho_minus_V_x_degree=2,
        compatible_deact_cbf_x_degree=[2, 2],
        compatible_h_x_degree=2,
        state_eq_x_degrees=None,
        safety_h_x_degree=2,
        safety_unsafe_polys_x_degree=[2]
    )

    # veryfy the first CBF:
    cbf_synthesis_given_clf.verification_first_cbf(cbf=h[0])
    # verify the second CBF:
    cbf_synthesis_given_clf.verification_other_cbf(
        cbf=h[1],
        cbf_index=1,
        deact_cbfs=h[:1],
    )
    # verify the third CBF:
    cbf_synthesis_given_clf.verification_other_cbf(
        cbf=h[2],
        cbf_index=2,
        deact_cbfs=h[:2],
    )


if __name__ == "__main__":
    main()
