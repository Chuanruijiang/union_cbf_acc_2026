import os
import sys
import os.path
import pickle
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import numpy as np
import pydrake.symbolic as sym
import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt

from union_cbf_base.utils import(
    deserialize_polynomial
)
from union_cbf_base.plot import(
    plot_2D_function,
)

def load_union_cbfs(x_set: sym.Variables) -> dict:
    with open("examples/non_linear_toy/union_cbfs.pkl", "rb") as handle:
        data = pickle.load(handle)

    cbfs = np.array([
        deserialize_polynomial(data["cbfs"][i], x_set)
        for i in range(len(data["cbfs"]))
    ])
    return cbfs

def load_synthesized_cbf(x_set: sym.Variables) -> sym.Polynomial:
    with open("examples/non_linear_toy/synthesized_high_degree_cbf.pkl", "rb") as handle:
        data = pickle.load(handle)
    
    h = deserialize_polynomial(data["cbf"], x_set)
    return h

def get_function_value(
    x: np.ndarray,
    f: sym.Polynomial,
    gamma_range: tuple[float, float],
    theta_range: tuple[float, float],
    sampling_rate: int
) -> np.ndarray:
    grid_gamma, grid_theta = np.meshgrid(
        np.linspace(gamma_range[0], gamma_range[1], sampling_rate),
        np.linspace(theta_range[0], theta_range[1], sampling_rate)
    )
    grid_x_val = np.concatenate(
        [grid_gamma.reshape(1, -1),
         np.sin(grid_theta.reshape(1, -1)),
         np.cos(grid_theta.reshape(1, -1))-1],
        axis=0
    )
    f_vals = f.EvaluateIndeterminates(x, grid_x_val)
    grid_f = f_vals.reshape(grid_gamma.shape)
    return np.array([
        grid_gamma,
        grid_theta,
        grid_f
    ])

def plot_unsafe_region(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    unsafe_polys: np.ndarray,
    gamma_range: tuple[float, float],
    theta_range: tuple[float, float],
    sampling_rate: int,
    color: str,
    alpha: float
):
    """
    Noted that the unsafe region is defined by p_i(x) >= 0,
    where i = 1, 2, ..., n and each p_i(x) is a polynomial.
    """
    grid_gamma, grid_theta = np.meshgrid(
        np.linspace(gamma_range[0], gamma_range[1], sampling_rate),
        np.linspace(theta_range[0], theta_range[1], sampling_rate)
    )
    grid_x_val = np.concatenate(
        [grid_gamma.reshape(1, -1),
         np.sin(grid_theta.reshape(1, -1)),
         np.cos(grid_theta.reshape(1, -1))-1],
        axis=0
    )
    num_p = unsafe_polys.shape[0]
    p_grid_values = np.zeros(shape=(num_p, grid_gamma.shape[1], grid_gamma.shape[0]))
    for i in range(num_p):
        p_vals = unsafe_polys[i].EvaluateIndeterminates(x, grid_x_val)
        p_grid = p_vals.reshape(grid_gamma.shape)
        p_grid_values[i, :, :] = p_grid
    
    intersection = np.min(p_grid_values, axis=0)
    intersection_region = ax.contourf(
        grid_gamma,
        grid_theta,
        intersection,
        levels=[0, np.inf],
        colors=color,
        alpha=alpha
    )
    return intersection_region

def plot_union_region(
    ax: matplotlib.axes.Axes,
    gird_gamma: np.ndarray,
    grid_theta: np.ndarray,
    grid_cbfs: np.ndarray,
    color: str,
    alpha: float
):
    if len(grid_cbfs.shape) == 3:
        max_gird_cbfs = np.max(grid_cbfs, axis=0)
    else:
        max_gird_cbfs = grid_cbfs
    grid_union_cbfs = max_gird_cbfs

    union_region = ax.contourf(
        gird_gamma,
        grid_theta,
        grid_union_cbfs,
        levels=[0, np.inf],
        colors=color,
        alpha=alpha
    )

    return union_region

def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '../../'))
    pi = np.pi
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)
    union_cbfs = load_union_cbfs(x_set)
    synthesized_cbf = load_synthesized_cbf(x_set)

    # computing function data for plot:
    synthesized_cbf_data = get_function_value(
        x=x,
        f=synthesized_cbf,
        gamma_range=(-2.0, 2.0),
        theta_range=(-np.pi/2, np.pi/2),
        sampling_rate=1000
    )
    cbf0_data = get_function_value(
        x=x,
        f=union_cbfs[0],
        gamma_range=(-2.0, 2.0),
        theta_range=(-np.pi/2, np.pi/2),
        sampling_rate=1000
    )
    cbf1_data = get_function_value(
        x=x,
        f=union_cbfs[1],
        gamma_range=(-2.0, 2.0),
        theta_range=(-np.pi/2, np.pi/2),
        sampling_rate=1000
    )

    # plot:
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # plot the unsafe region:
    unsafe_polys = np.array([
        sym.Polynomial(-x[0] + 0.3),
        sym.Polynomial(-x[1] + np.sin(-pi/4))
    ])
    unsafe_region = plot_unsafe_region(
        ax=ax,
        x=x,
        unsafe_polys=unsafe_polys,
        gamma_range=(-2.0, 2.0),
        theta_range=(-pi/2, pi/2),
        sampling_rate=1000,
        color="grey",
        alpha=1.0
    )

    # plot the syntheized cbf region h(x)=0
    plot_2D_function(
        ax=ax,
        f=synthesized_cbf_data,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_contour=True,
        color="darkgreen",
        contour_line_style="--"
    )

    # plot the union of cbf region h_i(x)>=0, i=1,2
    plot_union_region(
        ax=ax,
        gird_gamma=cbf0_data[0],
        grid_theta=cbf0_data[1],
        grid_cbfs=np.array([cbf0_data[2], cbf1_data[2]]),
        color="green",
        alpha=0.3
    )
    plot_2D_function(
        ax=ax,
        f=cbf0_data,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_contour=True,
        color="blue",
    )
    plot_2D_function(
        ax=ax,
        f=cbf1_data,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_contour=True,
        color="purple",
    )

    ax.set_xlabel(r"$\gamma$", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"$\theta(rad)$", fontsize=18, fontweight="bold")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-pi/2, -pi/4, 0, pi/4, pi/2])
    ax.set_xticklabels(
        [r"-2", r"-1", r"0", r"1", r"2"], 
        fontsize=18
        )
    ax.set_yticklabels([
        r"$-\frac{\pi}{2}$",
        r"$-\frac{\pi}{4}$",
        r"0",
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$"
        ], 
        fontsize=18
        )
    ax.set_xlim(-2, 2)
    ax.set_ylim(-pi/2, pi/2)
    ax.set_aspect("equal")


if __name__ == "__main__":
    main()