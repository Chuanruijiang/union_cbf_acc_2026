"""
This script shows the capability of the proposed verificaition SOS framework.
We consider the single integrator system with 2D, 3D and 4D state sapce. 
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

def single_integrator_parts(
    dim: int
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray]
]:
    """
    This function returns the f, g and control limits for single integrators
    in different dimensions.
    """
    f = np.array([sym.Polynomial(0) for _ in range(dim)])
    g = sym.Polynomial(0) * np.ones((dim, dim))
    for i in range(dim):
        g[i, i] = sym.Polynomial(1)
    
    A = np.zeros((4, dim))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 0] = -1.0
    A[3, 1] = -1.0
    c = 0.5 * np.ones((4,))

    return (f, g, A, c)

def cbf_collection(x: np.ndarray, dim: int) -> np.ndarray:
    assert x.shape[0] == dim
    center1 = np.zeros((dim,))
    center2 = np.zeros((dim,))
    center1[0] = -3.0
    radius = 2.0
    cbfs = np.array([
        sym.Polynomial(radius**2 - (x - center1).dot(x - center1)),
        sym.Polynomial(radius**2 - (x - center2).dot(x - center2))
    ])
    return cbfs

def verification(dim: int):
    x = sym.MakeVectorContinuousVariable(dim, "x")
    (f, g, A, c) = single_integrator_parts(dim)
    cbfs = cbf_collection(x, dim)
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

def main():
    for dim in range(2, 7):
        print(f"Verifying for dimension {dim}")
        verification(dim)

if __name__ == "__main__":
    main()