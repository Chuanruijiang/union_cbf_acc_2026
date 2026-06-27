import os
import pickle
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pydrake.symbolic as sym

import union_cbf_base.utils as utils
from union_cbf_base.plot import (
    plot_environement,
    plot_cbf_boundaries,
    plot_union_switching_intersect_normal_region,
)
from examples.double_integrator.generate_unsafe_region import (
    generate_constrained_shapes
)

def load_simulation_data(pickle_paths: List[str], x_vars: np.ndarray) -> List[dict]:
    """
    Load simulation dictionaries from multiple pickle paths.
    """
    all_data = []
    x_set = sym.Variables(x_vars)
    for pickle_path in pickle_paths:
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

        all_data.append(read_data)

    return all_data

def show():
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    x_low = -8.0
    x_high = 2.0
    y_low = -8.0
    y_high = 2.0
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(r"$p_x$", fontsize=16)
    ax.set_ylabel(r"$p_y$", fontsize=16)
    ax.set_xticks([-8, -6, -4, -2, 0, 2])
    ax.set_xticklabels(
        [r"$-8$", r"$-6$", r"$-4$", r"$-2$", r"$0$", r"$2$"]
    )
    ax.set_yticks([-8, -6, -4, -2, 0, 2])
    ax.set_yticklabels(
        [r"$-8$", r"$-6$", r"$-4$", r"$-2$", r"$0$", r"$2$"]
    )
    ax.tick_params(axis="both", which="major", labelsize=14)

    # load data for plotting:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path_II = os.path.join(
        current_dir, "../../data/double_integrator_2D_simulation_data_II.pkl"
    )
    x_vars = sym.MakeVectorContinuousVariable(4, "x")
    data_list = load_simulation_data(
        pickle_paths=[data_path_II],
        x_vars=x_vars,
    )
    base_data = data_list[0]
    static_cbfs = (base_data["static_cbfs"]
                   if "static_cbfs" in base_data.keys()
                   else None)
    switching_cbfs = (base_data["switching_cbfs"]
                      if "switching_cbfs" in base_data.keys()
                      else None)

    # plot the environment:
    obstacles = generate_constrained_shapes(x_vars=x_vars)
    plot_environement(
        ax=ax,
        x_states=x_vars,
        union_unsafe_regions=[-static_cbfs],
        intersection_unsafe_regions=obstacles,
        x_range=(x_low, x_high),
        y_range=(y_low, y_high)
    )

    # plot the safe region (union of switching CBFs ∩ static CBFs) in green:
    if switching_cbfs is not None and static_cbfs is not None:
        plot_union_switching_intersect_normal_region(
            ax,
            switching_cbfs,
            static_cbfs,
            x_vars[0:2],
            (x_low, x_high),
            (y_low, y_high),
            sampling_rate=500,
            color='green',
            alpha=0.4
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

    # plot simulated trajectory:
    state_data_II = data_list[0].get("state_data", None)
    if state_data_II is not None:
        ax.plot(
            state_data_II[0, :],
            state_data_II[1, :],
            color="red",
            linewidth=2,
        )
    if "waypoints" in data_list[0].keys():
        waypoints = data_list[0]["waypoints"]
        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            color="purple",
            marker="x",
            s=100,
        )

    # add explicit legend entries for CBF boundaries and safe region
    legend_handles = [
        Line2D([], [], color="red", linewidth=2, label="Switch II"),
        Line2D([], [], color="black", linewidth=2, linestyle="-",
               label=r"$h(x)=0$"),
        Line2D([], [], color="black", linewidth=2, linestyle="--",
               label=r"$b(x)=0$")
    ]
    ax.legend(handles=legend_handles, fontsize=14, loc="lower right")
    


if __name__ == "__main__":
    show()
