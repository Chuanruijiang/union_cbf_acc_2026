"""
In this script, we will verify a union of CBFs
h1(x) = x0 + x1 + 1
h2(x) = -x0 + x1 - 3
h3(x) = 0.5(x0 + 3)(x0 + 1) - x1

"""

import numpy as np
import pydrake.symbolic as sym
from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf import UnionCbf
from dynamics import (
    system_dynamics,
    control_limits
)


def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = system_dynamics()
    cbfs = np.array([
        sym.Polynomial(0.5 * (x[0] + 3) * (x[0] + 1) - x[1]),
        sym.Polynomial(0.5 * (x[0] + 3) * (x[0] + 1) + x[1])
    ])
    unsafe_polys = np.array([
        sym.Polynomial((0.1)**2 - (x[0]-(-2))**2 - (x[1])**2),
    ])
    union_object = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs,
        alpha=0.1,
        control_limits=control_limits(),
    )

    verfication_thm2 = union_object.verification_of_theorem_2(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=2,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=2,
        eta=1e-2,
        epsilon=0.01,
    )

    verfication_thm3 = union_object.verification_of_theorem_3(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=2,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=2,
        eta=1e-2,
        epsilon=0.01,
    )

    assert verfication_thm2 == True
    print("The verification of therorem 2 is successful!")

    assert verfication_thm3 == True
    print("The verification of therorem 3 is successful!")


if __name__ == "__main__":
    main()
