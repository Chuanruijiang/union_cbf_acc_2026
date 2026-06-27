"""
We have saved the simulation results of the 2D quadrotor 
example in a pickle file. This script loads the results
and plots them. The generated plots are figure 4(a) and (b)
in the journal paper.
"""

import os
import pickle
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pydrake.symbolic as sym

import union_cbf_base.utils as utils
from union_cbf_base.plot import (
    plot_environement,
    plot_cbf_boundaries,
    plot_simulation_results,
)

def load_simulation_data(pickle_path: str, x_vars: np.ndarray) -> dict:
    read_data = {}
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    
    # load the trajectory data:
    if "state_data" in data.keys():
        read_data["state_data"] = data["state_data"]
    if "action_data" in data.keys():
        read_data["action_data"] = data["action_data"]
    if "time_data" in data.keys():
        read_data["time_data"] = data["time_data"]
    if "waypoints" in data.keys():
        read_data["waypoints"] = data["waypoints"]

    # load the cbf data:
    x_set = sym.Variables(x_vars)
    if "static_cbfs" in data.keys():
        read_data["static_cbfs"] = np.array([
            utils.deserialize_polynomial(h_i, x_set)
            for h_i in data["static_cbfs"]
            ])
    if "switching_cbfs" in data.keys():
        read_data["switching_cbfs"] = np.array([
            utils.deserialize_polynomial(h_i, x_set)
            for h_i in data["switching_cbfs"]
            ])
    
    return read_data

def show_trajectories(
    state_data: np.ndarray,
    x_vars: np.ndarray,
    static_cbfs: Optional[np.ndarray] = None,
    switching_cbfs: Optional[np.ndarray] = None,
):
    traj_linewidth = 2.5
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    x_low = -20.0
    x_high = 2.0
    y_low = -5.0
    y_high = 1.0
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(r"$p_y$", fontsize=16)
    ax.set_ylabel(r"$p_z$", fontsize=16)
    ax.set_xticks([-20, -16, -12, -8, -4, 0, 2])
    ax.set_xticklabels(
        [r"$-20$", r"$-16$", r"$-12$", r"$-8$", r"$-4$", r"$0$", r"$2$"],
        fontsize=14
    )
    ax.set_yticks([-5, -4, -3, -2, -1, 0, 1])
    ax.set_yticklabels(
        [r"$-5$", r"$-4$", r"$-3$", r"$-2$", r"$-1$", r"$0$", r"$1$"],
        fontsize=14
    )
    ax.tick_params(axis="both", which="major", labelsize=14)
    # plot the environment:
    plot_environement(
        ax=ax,
        x_states=x_vars,
        union_unsafe_regions=(
            [static_cbfs] if static_cbfs is not None 
            else None
        ),
        intersection_unsafe_regions=(
            [-switching_cbfs] if switching_cbfs is not None 
            else None
        ),
        x_range=(x_low, x_high),
        y_range=(y_low, y_high)
    )

    # plot the cbf boundaries:
    plot_cbf_boundaries(
        ax=ax,
        x_states=x_vars,
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
        x_range=(x_low, x_high),
        y_range=(y_low, y_high)
    )

    # plot the simulated trajectories:
    plot_simulation_results(
        ax=ax,
        state_data=state_data,
    )

    # Thicken all trajectory and boundary curves drawn on this axis.
    for line in ax.lines:
        line.set_linewidth(traj_linewidth)
    for collection in ax.collections:
        if hasattr(collection, "set_linewidth"):
            collection.set_linewidth(traj_linewidth)
        elif hasattr(collection, "set_linewidths"):
            collection.set_linewidths(traj_linewidth)


    # plot waypoints:
    if "waypoints" in data.keys():
        waypoints = data["waypoints"]
        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            color="red",
            marker="x",
            s=100,
        )

    # Replace helper legend (if any) with the requested custom legend entries.
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    legend_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=traj_linewidth,
            label=r"$b(x)=0$",
        ),
        Line2D(
            [0], [0],
            color="blue",
            linestyle="-",
            linewidth=traj_linewidth,
            label="Switch II",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=14, frameon=True)

def show_actions(
    action_data: np.ndarray,
    time_data: np.ndarray,
):
    action_linewidth = 2.5
    limit_linewidth = 2.0
    x_ticks = [0, 2, 4, 6, 8]
    y_ticks = [0, 3, 6]
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(
        time_data,
        action_data[0, :],
        linewidth=action_linewidth,
        label=r"thrust $u_1$",
        color="darkgreen",
    )
    ax[1].plot(
        time_data,
        action_data[1, :],
        linewidth=action_linewidth,
        label=r"thrust $u_2$",
        color="darkgreen",
    )
    ax[0].set_xlabel(r"time $t$ (s)", fontsize=16)
    ax[1].set_xlabel(r"time $t$ (s)", fontsize=16)
    ax[0].set_ylabel(r"$T_1$ (N)", fontsize=16)
    ax[1].set_ylabel(r"$T_2$ (N)", fontsize=16)
    ax[0].set_xlim(0.0, 8.0)
    ax[1].set_xlim(0.0, 8.0)
    ax[0].set_xticks(x_ticks)
    ax[1].set_xticks(x_ticks)
    ax[0].set_xticklabels([r"$0$", r"$2$", r"$4$", r"$6$", r"$8$"], fontsize=14)
    ax[1].set_xticklabels([r"$0$", r"$2$", r"$4$", r"$6$", r"$8$"], fontsize=14)
    ax[0].set_yticks(y_ticks)
    ax[1].set_yticks(y_ticks)
    ax[0].set_yticklabels([r"$0$", r"$3$", r"$6$"], fontsize=14)
    ax[1].set_yticklabels([r"$0$", r"$3$", r"$6$"], fontsize=14)

    # show the control input limits:
    u_max = 1.5 * 0.486 * 9.81
    u_min = 0.0
    ax[0].axhline(u_max, color="red", linestyle="--", linewidth=limit_linewidth)
    ax[0].axhline(u_min, color="red", linestyle="--", linewidth=limit_linewidth)
    ax[1].axhline(u_max, color="red", linestyle="--", linewidth=limit_linewidth)
    ax[1].axhline(u_min, color="red", linestyle="--", linewidth=limit_linewidth)

    # legend and formatting:
    legend_handles = [
        Line2D([0], [0], color="darkgreen", linewidth=action_linewidth, label="control input"),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=limit_linewidth,
            label="control limits",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=14,
        frameon=True,
    )


if __name__ == "__main__":
    # load data for plotting:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        current_dir, "../../data/quadrotor_2D_simulation_results.pkl"
    )
    x_vars = sym.MakeVectorContinuousVariable(7, "x")
    data = load_simulation_data(pickle_path=data_path, x_vars=x_vars)
    static_cbfs = (data["static_cbfs"] 
                   if "static_cbfs" in data.keys() 
                   else None)
    switching_cbfs = (data["switching_cbfs"] 
                      if "switching_cbfs" in data.keys() 
                      else None)
    state_data = (data["state_data"] 
                  if "state_data" in data.keys() 
                  else None)
    action_data = (data["action_data"] 
                   if "action_data" in data.keys() 
                   else None)
    time_data = (data["time_data"] 
                 if "time_data" in data.keys() 
                 else None)
    
    show_trajectories(
        state_data=state_data,
        x_vars=x_vars,
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
    )

    show_actions(
        action_data=action_data,
        time_data=time_data,
    )






