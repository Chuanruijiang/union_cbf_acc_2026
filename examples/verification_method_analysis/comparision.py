"""
In this script, we compare the verification efficency of the two
theorems. 

We still use the same CBF setup as the we used in different_alignments.py.
and we compare the verifcation time of the two theorems for both the sparse
and dense alignments.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import numpy as np
from typing import Tuple, Optional, Union

import pydrake.symbolic as sym

from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf import UnionCbf
from examples.trajectory_tracking import plant
from different_alignments import (
    sparse_cbf_example,
    dense_cbf_example,
    verification_sparse,
    verification_dense
)

def verification_sparse_thm2():
    verification_sparse()

def verification_dense_thm2():
    verification_dense()

def verification_sparse_thm3():
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = plant.SingleIntegratorPlant()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    cbfs = sparse_cbf_example(x)
    union_object = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs,
        alpha=0.1,
        control_limits=(A, c),
    )

    start_time = time.time()
    verfication_thm3 = union_object.verification_of_theorem_3(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=1e-3,
        epsilon=0.01,
    )
    end_time = time.time()

    assert verfication_thm3 == True
    print("The verification of therorem 3 for sparse alignment is successful!")
    print(f"Time taken: {end_time - start_time} seconds")

def verification_dense_thm3():
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = plant.SingleIntegratorPlant()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    cbfs = dense_cbf_example(x)
    union_object = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs,
        alpha=0.1,
        control_limits=(A, c),
    )

    start_time = time.time()
    verfication_thm3 = union_object.verification_of_theorem_3(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=1e-3,
        epsilon=0.01,
    )
    end_time = time.time()

    assert verfication_thm3 == True
    print("The verification of therorem 3 for dense alignment is successful!")
    print(f"Time taken: {end_time - start_time} seconds")

def main():
    # verification_sparse_thm2()
    # verification_dense_thm2()
    verification_sparse_thm3()
    verification_dense_thm3()

if __name__ == "__main__":
    main()
