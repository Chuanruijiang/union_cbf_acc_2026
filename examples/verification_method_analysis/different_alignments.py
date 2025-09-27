"""
In this script, we wanted to show that the verification of
switching condition 1 is sensitive to the alignments of the
CBFs. 

We first consider three circles aligned in a sparse manner
such that ovelapping only exisists between two consecutive
circles. In this case, we have three cirlces with centers:
(-3, 0), (0, 0), (3, 0) and radius 2.

Then, we consider a denser alignment where the three circles
has a common overlapping region. In this case, we have three
circles centered at each vertex of an equilateral triangle
with side length 2 and radius 2. The centers are:
(-1, 0), (1, 0), (0, sqrt(3)).

We compare the verification time for both cases.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import numpy as np
from typing import Tuple, Optional, Union

import pydrake.symbolic as sym

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf import UnionCbf
from nonlinear_dynamics import NonlinearToyPlant

def sparse_cbf_example(
        x: Optional[np.ndarray] = None
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, float]
]:
    """ Create three sparse aligned circles as CBFs.

    The three circles are centered at (-0.3, 0), (0, 0), (0.3, 0)
    with radius 0.2.

    Args:
        x: The state variable, a 2D vector.

    Returns:
        A numpy array of shape (3,) representing the three CBFs.
    """
    if x is not None and (x.shape == (2,)):
        cbf1 = sym.Polynomial(0.2**2 - (x[0] + 0.3)**2 - (x[1] - 0)**2)
        cbf2 = sym.Polynomial(0.2**2 - (x[0] - 0)**2 - (x[1] - 0)**2)
        cbf3 = sym.Polynomial(0.2**2 - (x[0] - 0.3)**2 - (x[1] - 0)**2)
        return np.array([cbf1, cbf2, cbf3])
    else:
        centers = np.array([[-0.3, 0], [0, 0], [0.3, 0]])
        radius = 0.2
        return (centers, radius)

def dense_cbf_example(
        x:Optional[np.ndarray] = None
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, float]
]:
    """ Create three densely aligned circles as CBFs.

    The three circles are centered at (-0.1, 0), (0.1, 0), (0, 0.1*sqrt(3))
    with radius 2.

    Args:
        x: The state variable, a 2D vector.

    Returns:
        A numpy array of shape (3,) representing the three CBFs.
    """
    if x is not None and (x.shape == (2,)):
        cbf1 = sym.Polynomial(0.2**2 - (x[0] + 0.1)**2 - (x[1] - 0)**2)
        cbf2 = sym.Polynomial(0.2**2 - (x[0] - 0.1)**2 - (x[1] - 0)**2)
        cbf3 = sym.Polynomial(0.2**2 - (x[0] - 0)**2 - (x[1] - 0.1*np.sqrt(3))**2)
        return np.array([cbf1, cbf2, cbf3])
    else:
        centers = np.array([[-0.1, 0], [0.1, 0], [0, 0.1*np.sqrt(3)]])
        radius = 0.2
        return (centers, radius)

def plot_examples():

    (sparse_centers, sparse_radius) = sparse_cbf_example()
    (dense_centers, dense_radius) = dense_cbf_example()
    """
    Plot the the examples above for visualization. 
    """
    set_of_colors = ['green', 'blue', 'red']
    fig1, ax1 = plt.subplots(figsize=(5, 5))

    # Plot sparse example
    for i in range(sparse_centers.shape[0]):
        circle = patches.Circle(
            xy=sparse_centers[i, :],
            radius=sparse_radius,
            facecolor=set_of_colors[i],
            alpha=0.3
            )
        ax1.add_patch(circle)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    # Plot dense example
    for i in range(dense_centers.shape[0]):
        circle = patches.Circle(
            xy=dense_centers[i, :],
            radius=dense_radius,
            facecolor=set_of_colors[i],
            alpha=0.3
            )
        ax2.add_patch(circle)
    
    legend_patches = [
        patches.Patch(color=set_of_colors[0], alpha=0.3, label=r'$\mathcal{X}_1$'),
        patches.Patch(color=set_of_colors[1], alpha=0.3, label=r'$\mathcal{X}_2$'),
        patches.Patch(color=set_of_colors[2], alpha=0.3, label=r'$\mathcal{X}_3$'),
    ]

    ax1.set_title("Sparse Layout of CBFs", fontsize=16)
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.4, 0.6)
    ax1.set_xlabel(r"$x_1$", fontsize=16)
    ax1.set_ylabel(r"$x_2$", fontsize=16)
    ax1.legend(handles=legend_patches, fontsize=16)
    ax2.set_title("Dense Layout of CBFs", fontsize=16)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.4, 0.6)
    ax2.set_xlabel(r"$x_1$", fontsize=16)
    ax2.set_ylabel(r"$x_2$", fontsize=16)
    ax2.legend(handles=legend_patches, fontsize=16)

def verification_sparse():
    x = sym.MakeVectorContinuousVariable(2, "x")
    toy = NonlinearToyPlant()
    f, g = toy.affine_dynamics(x)
    A, c = toy.control_limits()
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
    verfication_thm2 = union_object.verification_of_theorem_2(
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
    assert verfication_thm2 == True
    print("The verification of therorem 2 for sparse alignment is successful!")
    print(f"Time taken: {end_time - start_time} seconds")

def verification_dense():
    x = sym.MakeVectorContinuousVariable(2, "x")
    toy = NonlinearToyPlant()
    f, g = toy.affine_dynamics(x)
    A, c = toy.control_limits()
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
    verfication_thm2 = union_object.verification_of_theorem_2(
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

    assert verfication_thm2 == True
    print("The verification of therorem 2 for dense alignment is successful!")
    print(f"Time taken: {end_time - start_time} seconds")

def main():
    plot_examples()
    # verification_sparse()
    # verification_dense()

if __name__ == "__main__":
    main()
