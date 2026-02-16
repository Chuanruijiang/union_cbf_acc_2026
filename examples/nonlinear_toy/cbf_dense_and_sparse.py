# """
# In this script, we wanted to show that the verification of
# switching condition 1 is sensitive to the alignments of the
# CBFs. 

# We first consider three circles aligned in a sparse manner
# such that ovelapping only exisists between two consecutive
# circles. In this case, we have three cirlces with centers:
# (-0.15, 0.15), (0, 0), (0.15, -0.15) and radius 0.2.

# Then, we consider a denser alignment where the three circles
# has a common overlapping region. In this case, we have three
# circles centered at each vertex of an equilateral triangle
# with side length 0.2 and radius 0.2. The centers are:
# (-0.1, 0), (0.1, 0), (0, 0.1*sqrt(3)).

# We compare the verification time for both cases.
# """

import numpy as np
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pydrake.symbolic as sym
from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf_I import (
    UnionCbfI,
    SubsetGeneralLagrangianDegrees,
    Degree as DegreeI,
)
from union_cbf_base.union_cbf_II import (
    UnionCbfII,
    CbfFeasibilityLagrangianDegrees,
    Degree as DegreeII,
)
from examples.nonlinear_toy.dynamics import NonlinearToyPlant

# We first use the following funcitons to define the sparse
# and dense aligments of the CBFs. We create these CBfs and
# then plot their super levelsets.
def sparse_cbf_example(
        x: Optional[np.ndarray] = None
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, float]
]:
    """ Create three sparse aligned circles as CBFs.

    The three circles are centered at (-0.15, 0.15), (0, 0), (0.15, -0.15)
    with radius 0.2.

    Args:
        x: The state variable, a 2D vector.

    Returns:
        A numpy array of shape (3,) representing the three CBFs.
    """
    if x is not None and (x.shape == (2,)):
        cbf1 = sym.Polynomial(0.2**2 - (x[0] + 0.15)**2 - (x[1] - 0.15)**2)
        cbf2 = sym.Polynomial(0.2**2 - (x[0] - 0)**2 - (x[1] - 0)**2)
        cbf3 = sym.Polynomial(0.2**2 - (x[0] - 0.15)**2 - (x[1] + 0.15)**2)
        return np.array([cbf1, cbf2, cbf3])
    else:
        centers = np.array([[-0.15, 0.15], [0, 0], [0.15, -0.15]])
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
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot sparse example
    for i in range(sparse_centers.shape[0]):
        circle = patches.Circle(
            xy=sparse_centers[i, :],
            radius=sparse_radius,
            facecolor=set_of_colors[i],
            alpha=0.3
            )
        ax[0].add_patch(circle)

    # Plot dense example
    for i in range(dense_centers.shape[0]):
        circle = patches.Circle(
            xy=dense_centers[i, :],
            radius=dense_radius,
            facecolor=set_of_colors[i],
            alpha=0.3
            )
        ax[1].add_patch(circle)

    ax[0].set_title("Sparse Layout of CBFs", fontsize=20)
    ax[0].set_xlim(-0.4, 0.4)
    ax[0].set_ylim(-0.4, 0.4)
    ax[0].set_xlabel(r"$x_1$", fontsize=18)
    ax[0].set_ylabel(r"$x_2$", fontsize=18)
    ax[0].set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax[0].set_yticklabels([
        r"$-0.4$",
        r"$-0.2$",
        r"$0$",
        r"$0.2$",
        r"$0.4$"
    ])
    ax[0].set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax[0].set_xticklabels([
        r"$-0.4$",
        r"$-0.2$",
        r"$0$",
        r"$0.2$",
        r"$0.4$"
    ])
    # ax[0].legend(handles=legend_patches, fontsize=16)
    ax[1].set_title("Dense Layout of CBFs", fontsize=20)
    ax[1].set_xlim(-0.4, 0.4)
    ax[1].set_ylim(-0.4, 0.4)
    ax[1].set_xlabel(r"$x_1$", fontsize=18, labelpad=5)
    ax[1].set_ylabel(r"$x_2$", fontsize=18, labelpad=5)
    ax[1].set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax[1].set_yticklabels([
        r"$-0.4$",
        r"$-0.2$",
        r"$0$",
        r"$0.2$",
        r"$0.4$"
    ])
    ax[1].set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax[1].set_xticklabels([
        r"$-0.4$",
        r"$-0.2$",
        r"$0$",
        r"$0.2$",
        r"$0.4$"
    ])

    legend_patches = [
        patches.Patch(color=set_of_colors[0], alpha=0.3, label=r'$\mathcal{X}_1$'),
        patches.Patch(color=set_of_colors[1], alpha=0.3, label=r'$\mathcal{X}_2$'),
        patches.Patch(color=set_of_colors[2], alpha=0.3, label=r'$\mathcal{X}_3$'),
    ]
    
    fig.legend(
        handles=legend_patches,
        fontsize=20,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=3)

# We then use the follwing functions to verify the sparse and
# dense examples using both the verif-I and verif-II methods.
# We are going to record the time for four experiemnts:
# spare: Verif-I and Verif-II
# dense: Verif-I and Verif-II.
def verification_with_verif_II(
    cbf_case: str = "sparse"
):
    x = sym.MakeVectorContinuousVariable(2, "x")
    toy = NonlinearToyPlant()
    f, g = toy.affine_dynamics(x)
    A, c = toy.control_limits()
    num_controls = A.shape[1]

    if cbf_case == "sparse":
        print("Verifying sparse alignment with Verif-II...")
        cbfs = sparse_cbf_example(x)
    elif cbf_case == "dense":
        print("Verifying dense alignment with Verif-II...")
        cbfs = dense_cbf_example(x)
    else:
        raise ValueError("Invalid cbf_case. Must be 'sparse' or 'dense'.")
    
    union_object = UnionCbfII(
        x=x,
        f=f,
        g=g,
        alpha=1.0,
        control_limits=(A, c),
    )
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        cbf = DegreeII(x=2, y=2, c=0),
        lambda_y=[DegreeII(x=2, y=0, c=0)]*num_controls,
        xi_y=DegreeII(x=2, y=0, c=0)
    )
    verfication_flag = union_object.verification_feasibility_condition_II(
        union_cbfs=cbfs,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_verification_time=True,
    )
    assert verfication_flag == True
    print("The verification of Verif-II for sparse alignment is successful!")

def verification_with_verif_I(
    cbf_case: str = "sparse"
):
    x = sym.MakeVectorContinuousVariable(2, "x")
    toy = NonlinearToyPlant()
    f, g = toy.affine_dynamics(x)
    A, c = toy.control_limits()
    num_controls = A.shape[1]

    if cbf_case == "sparse":
        print("Verifying sparse alignment with Verif-I...")
        cbfs = sparse_cbf_example(x)
    elif cbf_case == "dense":
        print("Verifying dense alignment with Verif-I...")
        cbfs = dense_cbf_example(x)
    else:
        raise ValueError("Invalid cbf_case. Must be 'sparse' or 'dense'.")
    
    union_object = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=1.0,
        control_limits=(A, c),
    )
    lagrangian_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=num_controls,
        cbfs_lagrangian_x_degree=2,
        cbfs_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=2,
        lambda_lagrangian_y_degree=0,
        xi_lagrangian_x_degree=2,
        xi_lagrangian_y_degree=0
    )
    verfication_flag = union_object.verification_feasibility_condition_I(
        union_cbfs=cbfs,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_verification_time=True,
    )
    assert verfication_flag == True
    print("The verification of Verif-I for dense alignment is successful!")

if __name__ == "__main__":
    # plot_examples()
    verification_with_verif_I(
        # cbf_case="sparse",
        cbf_case="dense",
    )
    verification_with_verif_II(
        # cbf_case="sparse",
        cbf_case="dense",
    )

