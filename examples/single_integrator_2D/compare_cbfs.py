# """
# In this script, we define the unsafe region and 
# provide the union of CBFs for that unsafe region.
# We also generate the compositional CBF and a high
# degree polynomial CBF for the same unsafe region,
# and save all the CBFs for future simulation.
# """

import os
import pickle
from typing import Optional, Tuple
import numpy as np
import matplotlib.lines as lines

import matplotlib.pyplot as plt
import pydrake.symbolic as sym

from union_cbf_base.plot import(
    plot_2D_function,
    plot_unsafe_region,
    plot_union_cbfs,
)
from union_cbf_base.utils import (
    serialize_polynomial,
    deserialize_polynomial
)

# the following function defines the unsafe region
# and the union of CBFs.
def unsafe_region_and_union_cbf(
    x: np.ndarray,
) -> Tuple[
    np.ndarray, # unsafe region
    np.ndarray # union of CBFs
]:
    """
    The unsafe region is a square with the center at (0, 0)
    and rotated by 45 degrees. This square can be represented
    by intersection of 4 half planes:
    1.  x[0] - x[1] + 2.9 >=0 
    2. -x[0] - x[1] + 2.9 >=0
    3. -x[0] + x[1] + 2.9 >=0
    4.  x[0] + x[1] + 2.9 >=0
    The union of CBfs, on the other hand, can be represented
    by the negative of the expressions above.
    1.  x[0] - x[1] + 3 <=0
    2. -x[0] - x[1] + 3 <=0
    3. -x[0] + x[1] + 3 <=0
    4.  x[0] + x[1] + 3 <=0
    """
    assert x.shape == (2,)
    unsafe_region = np.array([
        sym.Polynomial(x[0] - x[1] + 2.6),
        sym.Polynomial(-x[0] - x[1] + 2.6),
        sym.Polynomial(-x[0] + x[1] + 2.6),
        sym.Polynomial(x[0] + x[1] + 2.6),
    ])
    union_cbf = np.array([
        -sym.Polynomial(x[0] - x[1] + 2.8),
        -sym.Polynomial(-x[0] - x[1] + 2.8),
        -sym.Polynomial(-x[0] + x[1] + 2.8),
        -sym.Polynomial(x[0] + x[1] + 2.8),
    ])

    return unsafe_region, union_cbf

# the composition CBF is exponential, hence we need
# to define it as Expression class instead of Polynomial
# class. We also evaluate the composition CBF when 
# plotting its contour. 
def composition_cbf(
    union_cbfs: np.ndarray,
) -> sym.Expression:
    num_cbfs = union_cbfs.shape[0]
    k = 5
    b = np.log(num_cbfs)
    composite_cbf = (1 / k) * sym.log(
        sum(
            sym.exp(k * union_cbfs[i].ToExpression())
            for i in range(num_cbfs)
            )
    ) - b / k
    return composite_cbf

def evaluate_composition_cbf(
    f: sym.Expression,
    x: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    sampling_rate: int,
) -> np.ndarray:
    assert x.shape == (2,)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_range[0], x_range[1], sampling_rate),
        np.linspace(y_range[0], y_range[1], sampling_rate)
        )
    grid_x_val = np.concatenate(
        [grid_x.reshape(1, -1), grid_y.reshape(1, -1)],
        axis=0
    )
    f_vals = np.zeros(shape=(grid_x_val.shape[1],))
    for i, each_point in enumerate(grid_x_val.T):
        assert each_point.shape == (2,)
        env = {x[i]: each_point[i] for i in range(2)}
        f_vals[i] = f.Evaluate(env)
    grid_f = f_vals.reshape(grid_x.shape)

    return np.array([grid_x, grid_y, grid_f])

# besides the composition CBF, we can also use a high degree
# polynomial to approximate the safe region.
def high_degree_polynomial_cbf(
    x: np.ndarray,
    k: int = 2,
    bias: float = 0.1
) -> sym.Polynomial:
    """
    This function defines a high degree polynomial h(x)
    such that the region {x | h(x) < 0} covers the unsafe
    region defined in the previous function. 
    since we let the biases to be 2.9 in the previous
    function, then the following polynomial can be used:
    h(x) = ((x0 + x1)^(2k) + (x0 - x1)^(2k) - 2.9^(2k))
    where k is a positive integer.
    The actual degree of the polynomial is 2k.
    """
    assert x.shape == (2,)
    h = (x[0] + x[1])**(2*k) + \
        (x[0] - x[1])**(2*k) - \
        (2.8 + bias)**(2*k)
    return sym.Polynomial(h)

# we save the unsafe region and the CBFs for future simulation.
def save_unsafe_region_and_cbfs(
    unsafe_region: np.ndarray,
    union_cbfs: np.ndarray,
    composite_cbf_param: Tuple[float, float],
    high_degree_cbf: sym.Polynomial,
    x_vars: np.ndarray,
    pickle_path: str
):
    """
    Save the switching and normal CBFs in a pickle file.
    Noted that even if the composition CBF is an expression, we can still
    reconstruct it using the union of CBFs and the parameter k and b.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    assert isinstance(composite_cbf_param, tuple) and len(composite_cbf_param) == 2

    data = {
        "unsafe_polys": [
            serialize_polynomial(h_i, sym.Variables(x_vars)) 
            for h_i in unsafe_region
        ],
        "union_cbfs": [
            serialize_polynomial(h_i, sym.Variables(x_vars)) 
            for h_i in union_cbfs
        ],
        "composite_cbf_param": composite_cbf_param,
        "high_degree_cbf": serialize_polynomial(high_degree_cbf, sym.Variables(x_vars))
    }

    if os.path.exists(pickle_path):
        overwrite_cmd = input(
            f"File {pickle_path} already exists. Overwrite the file? Press [Y/n]:"
        )
        if overwrite_cmd in ("Y", "y"):
            save_cmd = True
        else:
            save_cmd = False
    else:
        save_cmd = True

    if save_cmd:
        with open(pickle_path, "wb") as handle:
            pickle.dump(data, handle)

def load_unsafe_region_and_cbfs(
    x_vars: np.ndarray,
    pickle_path: str
) -> Tuple[
    np.ndarray, # unsafe region
    np.ndarray, # union of CBFs
    sym.Expression, # composition CBF
    Tuple[float, float], # compositional CBF parameter
    sym.Polynomial # high degree polynomial CBF
]:
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    
    x_set = sym.Variables(x_vars)
    unsafe_polys = np.array([
        deserialize_polynomial(h_i, x_set)
        for h_i in data["unsafe_polys"]
        ])
    union_cbfs = np.array([
        deserialize_polynomial(h_i, x_set)
        for h_i in data["union_cbfs"]
        ])
    composite_cbf_param = data["composite_cbf_param"]
    # construct the composition CBF:
    k, b = composite_cbf_param
    composite_cbf = (1 / k) * sym.log(
        sum(
            sym.exp(k * union_cbfs[i].ToExpression())
            for i in range(union_cbfs.shape[0])
            )
    ) - b / k
    high_degree_cbf = deserialize_polynomial(data["high_degree_cbf"], x_set)

    return(
        unsafe_polys,
        union_cbfs,
        composite_cbf,
        composite_cbf_param,
        high_degree_cbf
    )



def main(save: bool = False):
    x = sym.MakeVectorContinuousVariable(2, "x")
    # createt unsafe region and union of CBFs:
    unsafe_region, union_cbfs = unsafe_region_and_union_cbf(x)
    # create the composition CBF and evaluate it for plotting:
    composite_cbf = composition_cbf(union_cbfs)
    # create the high degree polynomial CBF:
    high_degree_cbf = high_degree_polynomial_cbf(x, k=3, bias=0.3)

    x0_low, x0_high = -4.5, 5.5
    x1_low, x1_high = -4.5, 4.5
    composite_cbf_data = evaluate_composition_cbf(
        f=composite_cbf,
        x=x,
        x_range=(x0_low, x0_high),
        y_range=(x1_low, x1_high),
        sampling_rate=500
    )
    # plot the problem setup:
    fig, ax = plt.subplots(
        # figsize=(6/1.2, 4.5/1.2)
    )
    ax.set_xlim(x0_low, x0_high)
    ax.set_ylim(x1_low, x1_high)
    # plot the unsafe region:
    plot_unsafe_region(
        ax=ax,
        x_states=x,
        unsafe_regions=[unsafe_region],
        x_range=(x0_low, x0_high),
        y_range=(x1_low, x1_high),
    )
    # plot the union of CBFs bounaries:
    plot_union_cbfs(
        ax=ax,
        x_states=x,
        switching_cbfs=union_cbfs,
        x_range=(x0_low, x0_high),
        y_range=(x1_low, x1_high),
        line_color="black",
        contour_line_style='--',
        contour_line_width=2,
        mark_region=True,
        region_color="lightgreen",
        region_alpha=1.0
    )
    # plot the composition CBF boundary:
    plot_2D_function(
        ax=ax,
        f=composite_cbf_data,
        x=x,
        x_range=(x0_low, x0_high),
        y_range=(x1_low, x1_high),
        sampling_rate=None,
        with_contour=True,
        line_color="red",
        contour_line_style='-',
        contour_line_width=2,
        with_region_filled=False
    )
    # plot the high degree polynomial CBF boundary:
    plot_union_cbfs(
        ax=ax,
        x_states=x,
        switching_cbfs=np.array([high_degree_cbf]),
        x_range=(x0_low, x0_high),
        y_range=(x1_low, x1_high),
        line_color="blue",
        contour_line_style='-.',
        contour_line_width=2,
        mark_region=False,
    )

    ax.set_xlabel("x0", fontsize=16)
    ax.set_ylabel("x1", fontsize=16)
    ax.set_title("Safe Region Converage Comparison", fontsize=16)
    
    legend_handles = [
        lines.Line2D([0], [0], color='black', lw=2, linestyle='--', label=r'$h_i(x)=0$'),
        lines.Line2D([0], [0], color='red', lw=2, linestyle='-', label=r'$h_{comp}(x) = 0$'),
        lines.Line2D([0], [0], color='blue', lw=2, linestyle='-.', label=r'$h_{high}(x) = 0$'),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=14,
        loc="upper right",
        ncol=1
    )

    if save==True:
        # save the cbf polys and expressions for future simulation:
        filename = "unsafe_region_and_cbfs.pkl"
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
        )
        save_unsafe_region_and_cbfs(
            x_vars=x,
            unsafe_region=unsafe_region,
            union_cbfs=union_cbfs,
            composite_cbf_param=(5, np.log(union_cbfs.shape[0])),
            high_degree_cbf=high_degree_cbf,
            pickle_path=data_path
        )


if __name__ == "__main__":
    main(save=False)