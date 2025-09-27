"""
This script compares the verification time a single high degree CBF
to the union of 4 linear CBFs.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import numpy as np
import pydrake.symbolic as sym
from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf import UnionCbf
from dynamics import (
    system_dynamics,
    control_limits
)

def verification_union_linear():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = system_dynamics()
    cbfs = np.array([
        sym.Polynomial(x[0] - 1.01),
        sym.Polynomial(-x[0] - 1.01),
        sym.Polynomial(x[1] - 1.01),
        sym.Polynomial(-x[1] - 1.01),
    ])
    union_object = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs,
        alpha=0.1,
        control_limits=control_limits(),
    )

    start_time = time.time()
    verfication_thm2 = union_object.verification_of_theorem_2(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=1e-2,
        epsilon=0.01,
    )
    end_time = time.time()

    assert verfication_thm2 == True
    print("The verification of therorem 2 is successful!")
    print(f"Time for verifying theorem 2: {end_time - start_time} seconds")

    start_time = time.time()
    verfication_thm3 = union_object.verification_of_theorem_3(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=1e-2,
        epsilon=0.01,
    )
    end_time = time.time()

    assert verfication_thm3 == True
    print("The verification of therorem 3 is successful!")
    print(f"Time for verifying theorem 3: {end_time - start_time} seconds")


def verification_high_degree_polynomial():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = system_dynamics()
    cbfs = np.array([
        sym.Polynomial(x[0]**6 + x[1]**6 - 2.30)
    ])
    union_object = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs,
        alpha=0.1,
        control_limits=control_limits(),
    )
    start_time = time.time()
    verfication_thm3 = union_object.verification_of_theorem_3(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=1e-2,
        epsilon=0.01,
    )
    end_time = time.time()
    assert verfication_thm3 == True
    print("The verification of polynomial is successful!")
    print(f"Time for verifying poly: {end_time - start_time} seconds")


def main():
    # verification_union_linear()
    verification_high_degree_polynomial()

if __name__ == "__main__":
    main()

