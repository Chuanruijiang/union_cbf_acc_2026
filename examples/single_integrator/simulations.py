"""
This file simulates the synthesized CBF and CLF with CLF-CBF-QP
we use the single integrator dynamics as the example. 
"""
import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import os.path
import pickle
from typing import Optional
import numpy as np
import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt
import pydrake.symbolic as sym

from compatible_clf_union_cbf.utils import(
    deserialize_polynomial,
)
from dynamics import system_dynamics_forward
from compatible_clf_union_cbf.controller import cbf_clf_qp
from compatible_clf_union_cbf.plot import(
    plot_2D_function,
    plot_intersection_region
)


def get_pkl_file_path():
    filename = "single_integrator_synthesized_clf_cbf.pkl"
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

def main():
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
        [-8, 1],
        [-8, 2.5],
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
    # t_range = np.linspace(0, 15, 1000)
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
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    ax.set_xticks([-15, -10, -5, 0, 5])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels([r"-15", r"-10", r"-5", r"0", r"5"], fontsize=16)
    ax.set_yticklabels([r"-10", r"-5", r"0", r"5", r"10"], fontsize=16)
    ax.set_xlim(-15, 5)
    ax.set_ylim(-5, 5)
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
    # plot the safe region:
    for i in range(len(h)):
        plot_2D_function(
            ax=ax,
            f=h[i],
            x=x,
            x_range=(-15, 5),
            y_range=(-5, 5),
            sampling_rate=1000
        )
    fig.show()
    
def compaired_method():
    # use this function to define the method  to compare
    # 





if __name__ == "__main__":
    main()