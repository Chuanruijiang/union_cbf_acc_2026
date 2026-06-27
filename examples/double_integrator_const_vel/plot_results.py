"""
This script plots the simulation results of the simiulation
of double integrator with constant velocity using switching
policy I. 
The plot is also the figure 2 in the journal paper
"""
import os
import pickle
import numpy as np
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

def show():
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot()
    x_low = -50.0
    x_high = 50.0
    y_low = -50.0
    y_high = 50.0
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(r"$p_x$", fontsize=16)
    ax.set_ylabel(r"$p_y$", fontsize=16)
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_xticklabels(
        [r"$-50$", r"$-25$", r"$0$", r"$25$", r"$50$"]
    )
    ax.set_yticks([-50, -25, 0, 25, 50])
    ax.set_yticklabels(
        [r"$-50$", r"$-25$", r"$0$", r"$25$", r"$50$"]
    )
    ax.tick_params(axis="both", which="major", labelsize=14)

    # load data for plotting:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        current_dir, "../../data/double_integrator_const_vel_simulation_data.pkl"
    )
    x_vars = sym.MakeVectorContinuousVariable(4, "x")
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

    # plot the environment:
    plot_environement(
        ax=ax,
        x_states=x_vars,
        union_unsafe_regions=[-static_cbfs],
        intersection_unsafe_regions=[-switching_cbfs],
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
        state_data=state_data
    )

    # plot reference trajectory x[1] = 0:
    ax.plot(
        [x_low, x_high], [0.0, 0.0],
        label="Reference Height",
        color="red",
        linestyle="--",
        linewidth=2.0
    )

    legend_handles = [
        Line2D([0], [0], color="red", linestyle="--", linewidth=2.0,
               label="Reference Height"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.0,
               label=r"$h(x) = 0$"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0,
               label=r"$b(x) = 0$"),
        Line2D([0], [0], color="blue", linestyle="-", linewidth=2.0,
               label="Switch I"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=14)

    plt.tight_layout()
    plt.show()

    
    


if __name__ == "__main__":
    show()
