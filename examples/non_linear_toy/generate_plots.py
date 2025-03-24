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

from compatible_clf_union_cbf.utils import(
    deserialize_polynomial
)
from compatible_clf_union_cbf.plot import(
    plot_2D_function,
)


def get_pkl_file_path(
    name_file: str
):  
    filename = name_file
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path

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

def plot_compatible_region(
    ax: matplotlib.axes.Axes,
    gird_gamma: np.ndarray,
    grid_theta: np.ndarray,
    grid_cbfs: np.ndarray,
    grid_clf: np.ndarray,
    color: str,
    alpha: float
):
    if len(grid_cbfs.shape) == 3:
        max_gird_cbfs = np.max(grid_cbfs, axis=0)
    else:
        max_gird_cbfs = grid_cbfs
    grid_compatible_region = np.minimum(max_gird_cbfs, grid_clf)

    compatible_region = ax.contourf(
        gird_gamma,
        grid_theta,
        grid_compatible_region,
        levels=[0, np.inf],
        colors=color,
        alpha=alpha
    )

    return compatible_region

def load_data(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    ret["V"] = deserialize_polynomial(data["V"], x_set)
    ret["h"] = np.array([
        deserialize_polynomial(h_i, x_set) for h_i in data["h"]
    ])
    return ret

def clf_single_cbf():
    pi = np.pi
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)

    data = load_data(
        pickle_path=get_pkl_file_path(
            name_file="non_linear_toy_clf_single_cbf_synthesized.pkl"
        ),
        x_set=x_set
    )

    clf = data["V"]
    cbfs = data["h"]

    clf_data_for_plot = get_function_value(
        x=x,
        f=1 - clf,
        gamma_range=(-2.0, 2.0),
        theta_range=(-pi/2, pi/2),
        sampling_rate=1000
    )
    cbf_data_for_plot = get_function_value(
        x=x,
        f=cbfs[0],
        gamma_range=(-2.0, 2.0),
        theta_range=(-pi/2, pi/2),
        sampling_rate=1000
    )
    
    # initializing plot object:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(r"$\gamma$", fontsize=18)
    ax.set_ylabel(r"$\theta(rad)$", fontsize=18)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-pi/2, -pi/4, 0, pi/4, pi/2])
    ax.set_xticklabels([r"-2", r"-1", r"0", r"1", r"2"], fontsize=18)
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

    # plot the clf region 1-V(x)>=0 
    rho_minu_V,_ = plot_2D_function(
        ax=ax,
        f=clf_data_for_plot,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_region_filled=False,
        with_contour=True,
        color="red",
        alpha=0.3
    )

    # plot the cbf region h(x)>=0
    cbf_contour, _ = plot_2D_function(
        ax=ax,
        f=cbf_data_for_plot,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_contour=True,
        with_region_filled=False,
        color="blue",
        alpha=0.3,
    )

    #plot compatible region:
    compateble_region = plot_compatible_region(
        ax=ax,
        gird_gamma=clf_data_for_plot[0],
        grid_theta=clf_data_for_plot[1],
        grid_cbfs=cbf_data_for_plot[2],
        grid_clf=clf_data_for_plot[2],
        color="green",
        alpha=0.3
    )

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

    # add legend:
    ax.legend(
        [
            rho_minu_V.legend_elements()[0][0],
            cbf_contour.legend_elements()[0][0],
        ],
        ["$V(x)=1$", "$h(x)=0$"],
        loc="upper right",
        prop={"size": 18}
    )

    fig.show()

def clf_union_cbf():
    pi = np.pi
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)

    data = load_data(
        pickle_path=get_pkl_file_path(
            name_file="non_linear_toy_clf_union_cbf_synthesized.pkl"
        ),
        x_set=x_set
    )

    clf = data["V"]
    cbfs = data["h"]

    clf_data_for_plot = get_function_value(
        x=x,
        f=1 - clf,
        gamma_range=(-2.0, 2.0),
        theta_range=(-pi/2, pi/2),
        sampling_rate=1000
    )
    cbf1_data_for_plot = get_function_value(
        x=x,
        f=cbfs[0],
        gamma_range=(-2.0, 2.0),
        theta_range=(-pi/2, pi/2),
        sampling_rate=1000
    )
    cbf2_data_for_plot = get_function_value(
        x=x,
        f=cbfs[1],
        gamma_range=(-2.0, 2.0),
        theta_range=(-pi/2, pi/2),
        sampling_rate=1000
    )

    
    # initializing plot object:
    fig = plt.figure()
    ax = fig.add_subplot()
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

    # plot the clf region 1-V(x)>=0 
    rho_minus_V_contour,_ = plot_2D_function(
        ax=ax,
        f=clf_data_for_plot,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_contour=True,
        with_region_filled=False,
        color="red",
        alpha=0.3
    )

    # plot the cbf region h(x)>=0
    cbf1_contour,_ = plot_2D_function(
        ax=ax,
        f=cbf1_data_for_plot,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_contour=True,
        with_region_filled=False,
        color="blue",
        alpha=0.3,
    )
    cbf2_contour,_ = plot_2D_function(
        ax=ax,
        f=cbf2_data_for_plot,
        x=x,
        x_range=(-2, 2),
        y_range=(-pi/2, pi/2),
        sampling_rate=None,
        with_contour=True,
        with_region_filled=False,
        color="purple",
        alpha=0.3,
    )

    # plot compatible region:
    compatible_region = plot_compatible_region(
        ax=ax,
        gird_gamma=clf_data_for_plot[0],
        grid_theta=clf_data_for_plot[1],
        grid_cbfs=np.array([cbf1_data_for_plot[2], cbf2_data_for_plot[2]]),
        grid_clf=clf_data_for_plot[2],
        color="green",
        alpha=0.3
    )

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

    # add legend:
    ax.legend(
        [   
            rho_minus_V_contour.legend_elements()[0][0],
            cbf1_contour.legend_elements()[0][0],
            cbf2_contour.legend_elements()[0][0],
        ],
        ["$V(x)=1$", "$h_1(x)=0$", "$h_2(x)=0$"],
        loc="upper right",
        prop={"size": 18}
    )


    fig.show()

def main():
    clf_single_cbf()
    clf_union_cbf()


if __name__ == "__main__":
    main()