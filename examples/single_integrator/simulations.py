"""
This file simulates the synthesized CBF and CLF with CLF-CBF-QP
we use the single integrator dynamics as the example. 
"""
import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import os.path
import pickle
from typing import Optional, Tuple
import numpy as np
import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt
import pydrake.symbolic as sym

from compatible_clf_union_cbf.utils import(
    deserialize_polynomial,
)
from dynamics import system_dynamics_forward
from compatible_clf_union_cbf.controller import(
     cbf_clf_qp,
     compared_cbf_qp,
     )
from compatible_clf_union_cbf.plot import(
    plot_2D_function,
    plot_intersection_region,
    plot_union_region,
    plot_compatible_region
)

def compute_compared_2D_function(
    x: np.ndarray,
    h_polys: np.ndarray,
    smooth_k: float,
    buffer_b: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a set of h_i(x) functions, compute the compared function:
    h(x) = 1/k * log(sum_i(exp(k * h_i(x))) + b/k
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_range[0], x_range[1], sampling_rate),
        np.linspace(y_range[0], y_range[1], sampling_rate)
    )
    grid_x_val = np.concatenate(
        [grid_x.reshape(1, -1), grid_y.reshape(1, -1)],
        axis=0
    )
    h_i_values = np.array([
        h_poly.EvaluateIndeterminates(x, grid_x_val)
        for h_poly in h_polys
    ])
    h_value = (1/smooth_k) * np.log(
        np.sum(np.exp(smooth_k * h_i_values), axis=0)
    ) - buffer_b/smooth_k

    compared_cbf_value_grid = h_value.reshape(grid_x.shape)

    return (grid_x, grid_y, compared_cbf_value_grid)

def get_pkl_file_path():
    filename = "single_integrator_synthesized_clf_union_cbf.pkl"
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path

def load_clf_cbf(pickle_path: str, x_set: sym.Variables) -> dict:
    ret = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    if "V" in data.keys():
        ret["V"] = deserialize_polynomial(data["V"], x_set)
    ret["h"] = np.array(
        [
            deserialize_polynomial(h_i, x_set)
            for h_i in data["h"]
        ]
    )
    if "kappa_V" in data.keys():
        ret["kappa_V"] = data["kappa_V"]
    ret["kappa_h"] = data["kappa_h"]
    return ret

def simulate():
    x = sym.MakeVectorContinuousVariable(2, "x")
    x_set = sym.Variables(x)
    f = system_dynamics_forward()[0]
    g = system_dynamics_forward()[1]
    loading_path = get_pkl_file_path()
    data = load_clf_cbf(loading_path, x_set)
    V = data["V"]
    h = data["h"]
    kappa_V = data["kappa_V"]
    kappa_h = data["kappa_h"]

    # all the starting points:
    starting_points = np.array([
        [-1 ,1.5],
        [-1.5, 1],
        [-2, 1.5],
        [-1.8, 2.1],
        [-1.2, 3],
        [-2.5, 1.5],
        [-2.5, 2],
        [-3,3],
        [-3.5, 1.2],
        [-3.5, 2.2],
        [-5, 1.5],
        [-5, 2.3],
        [-6, 0.8],
        [-7, 0],
        [-7, 0.5],
        [-8, 1],
        [-8, 2.5],
        [-8, 0],
        [-8, 0.5],
        [-9, 0],
        [-9, 1],
        [-10, 0],
        [-10, 1.5],
        [-10.5, 2],
        [-11, 1.5],
        [-11, 2],
        [-11, 0],
        [-11, -1],
        [-11, -1.5]
    ])

    # # simulate the system:
    # t_range = np.linspace(0, 25, 1000)
    # t_delta = t_range[1] - t_range[0]
    # x_traj = np.zeros((starting_points.shape[0], 2, t_range.shape[0]))
    # for k in range(starting_points.shape[0]):
    #     x_start = starting_points[k]
    #     x_traj[k, :, 0] = x_start
    #     for i in range(1, t_range.shape[0]):
    #         u = cbf_clf_qp(
    #             x_value=x_traj[k, :, i-1],
    #             x=x,
    #             f=f,
    #             g=g,
    #             V=V,
    #             h=h,
    #             kappa_V=kappa_V,
    #             kappa_h=kappa_h,
    #             relative_degrees=None,
    #             Au=None,
    #             bu=None,
    #             Q=np.eye(2)
    #         )
    #         x_delta = (f + g @ u) * t_delta
    #         x_traj[k, :, i] = x_traj[k, :, i-1] + x_delta

    # plot the trajectory:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18)
    ax.set_xticks([-12, -10, -8, -6, -4, -2, 0, 2])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels([r"-12", r"-10", r"-8", r"-6", r"-4", r"-2", r"0", r"2"], fontsize=18)
    ax.set_yticklabels([r"-3", r"-2", r"-1", r"0", r"1", r"2", r"3"], fontsize=18)
    ax.set_xlim(-12, 2)
    ax.set_ylim(-3, 3)

    # # plot trajectory:
    # for k in range(starting_points.shape[0]):
    #     ax.plot(x_traj[k, 0, :], x_traj[k, 1, :], label=f"Trajectory {k}")

    # plot the unsafe region:
    unsafe_polys = np.array([
        sym.Polynomial(x[0] + 5),
        sym.Polynomial(-x[0] - 3),
        sym.Polynomial(x[1] + 1),
        sym.Polynomial(-x[1] + 1),
    ])
    plot_intersection_region(
        ax=ax,
        p=unsafe_polys,
        x=x,
        x_range=(-15, 5),
        y_range=(-5, 5),
        sampling_rate=1000
    )

    # plot the region of V(x)<=1:
    rho_minus_V_contour,_ = plot_2D_function(
        ax=ax,
        f=1-V,
        x=x,
        x_range=(-15, 5),
        y_range=(-5, 5),
        sampling_rate=1000,
        with_contour=True,
        with_region_filled=False,
        color="red",
        alpha=0.3
    )

    compatible_region = plot_compatible_region(
        ax=ax,
        h=h,
        V=V,
        x=x,
        x_range=(-15, 5),
        y_range=(-5, 5),
        sampling_rate=1000,
        color="green",
        alpha=0.3
    )

    # plot the safe region:
    cbf_1_contour,_ = plot_2D_function(
            ax=ax,
            f=h[0],
            x=x,
            x_range=(-15, 5),
            y_range=(-5, 5),
            sampling_rate=1000,
            with_contour=True,
            with_region_filled=False,
            color="blue",
            alpha=0.3
    )
    cbf_2_contour,_ = plot_2D_function(
            ax=ax,
            f=h[1],
            x=x,
            x_range=(-15, 5),
            y_range=(-5, 5),
            sampling_rate=1000,
            with_contour=True,
            with_region_filled=False,
            color="purple",
            alpha=0.3,
            contour_line_style="--"
    )
    cbf_3_contour,_ = plot_2D_function(
            ax=ax,
            f=h[2],
            x=x,
            x_range=(-15, 5),
            y_range=(-5, 5),
            sampling_rate=1000,
            with_contour=True,
            with_region_filled=False,
            color="green",
            alpha=0.3,
            contour_line_style="-."
    )

    # add the legend:
    ax.legend(
        [
            rho_minus_V_contour.legend_elements()[0][0],
            cbf_1_contour.legend_elements()[0][0],
            cbf_2_contour.legend_elements()[0][0],
            cbf_3_contour.legend_elements()[0][0]
        ],
        [
            "$V(x)=1$",
            "$h_1(x)=0$",
            "$h_2(x)=0$",
            "$h_3(x)=0$",
        ],
        loc="upper left",
        prop={"size": 18, 
              "weight": "bold"}
    )
    fig.show()

def compare():
    x = sym.MakeVectorContinuousVariable(2, "x")
    x_set = sym.Variables(x)
    f = system_dynamics_forward()[0]
    g = system_dynamics_forward()[1]
    loading_path = get_pkl_file_path()
    data = load_clf_cbf(loading_path, x_set)
    V = data["V"]
    h = data["h"]
    kappa_V = data["kappa_V"]
    kappa_h = data["kappa_h"]

    unsafe_polys = np.array([
        sym.Polynomial(x[0] + 5),
        sym.Polynomial(-x[0] - 3),
        sym.Polynomial(x[1] + 1),
        sym.Polynomial(-x[1] + 1),
    ])

    # starting point:
    x_start = np.array([-11, 0])
    t_range = np.linspace(0, 15, 1000)
    t_delta = t_range[1] - t_range[0]
    x_traj1 = np.zeros((2, t_range.shape[0]))
    x_traj2 = np.zeros((2, t_range.shape[0]))
    x_traj1[:, 0] = x_start
    x_traj2[:, 0] = x_start
    for i in range(1, t_range.shape[0]):
        u_1 = compared_cbf_qp(
            x_value=x_traj1[:, i-1],
            x=x,
            f=f,
            g=g,
            h_polys=-unsafe_polys,
            kappa_h=1,
            K_desired_control = 0.5,
            smooth_k=2,
            buffer_b=np.log(unsafe_polys.shape[0])
        )
        x_delta = (f + g @ u_1) * t_delta
        x_traj1[:, i] = x_traj1[:, i-1] + x_delta
        
        u_2 = cbf_clf_qp(
            x_value=x_traj2[:, i-1],
            x=x,
            f=f,
            g=g,
            V=V,
            h=h,
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            relative_degrees=None,
            Au=None,
            bu=None,
            Q=np.eye(2)
        )
        x_delta = (f + g @ u_2) * t_delta
        x_traj2[:, i] = x_traj2[:, i-1] + x_delta
    
    # plot the trajectory:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18)
    ax.set_xticks([-12, -10, -8, -6, -4, -2, 0, 2])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels([r"-12", r"-10", r"-8", r"-6", r"-4", r"-2", r"0", r"2"], fontsize=18)
    ax.set_yticklabels([r"-3", r"-2", r"-1", r"0", r"1", r"2", r"3"], fontsize=18)
    ax.set_xlim(-12, 2)
    ax.set_ylim(-3, 3)
    # plot trajectory:
    (
        trajectory_safety_filter,
    ) = ax.plot(
        x_traj1[0, :], x_traj1[1, :], 
        label="Compared_Trajectory",
        color="red",
        linewidth=4
        )
    (
        trajecotry_CLF_CBF,
    ) = ax.plot(
        x_traj2[0, :], x_traj2[1, :],
        label="CLF-CBF-QP",
        color="blue",
        linewidth=4
    )
    # plot the unsafe region:
    plot_intersection_region(
        ax=ax,
        p=unsafe_polys,
        x=x,
        x_range=(-15, 5),
        y_range=(-5, 5),
        sampling_rate=1000
    )
    # # plot the safe region:
    # plot_union_region(
    #     ax=ax,
    #     p=-unsafe_polys,
    #     x=x,
    #     x_range=(-15, 5),
    #     y_range=(-5, 5),
    #     sampling_rate=1000
    # )
    # compute the h_value of syntheized h(x):
    # union_h_info = compute_compared_2D_function(
    #     x=x,
    #     h_polys=-unsafe_polys,
    #     smooth_k=2,
    #     buffer_b=np.log(unsafe_polys.shape[0]),
    #     x_range=(-15, 5),
    #     y_range=(-5, 5),
    #     sampling_rate=1000
    # )
    # plot the compared h(x):
    # plot_2D_function(
    #     ax=ax,
    #     f=union_h_info,
    #     x=x,
    #     x_range=(-15, 5),
    #     y_range=(-5, 5),
    #     sampling_rate=1000,
    #     color="blue",
    #     alpha=0.3
    # )

    # add the legend:
    ax.legend(
        [
            trajectory_safety_filter,
            trajecotry_CLF_CBF,
        ],
        [
            "Independent CLF/CBF",
            "CLF/maixmum CBF"
        ],
        loc="upper left",
        prop={"size": 18}
    )

    fig.show()


def main():
    simulate()
    compare()


if __name__ == "__main__":
    main()