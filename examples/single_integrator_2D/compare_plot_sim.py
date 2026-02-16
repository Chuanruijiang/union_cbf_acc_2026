import os
import pickle
import numpy as np
from typing import Tuple, List, Optional
from matplotlib import pyplot as plt

import pydrake.symbolic as sym
import matplotlib.lines as lines

from union_cbf_base.plot import (
    plot_2D_function,
    plot_unsafe_region,
    plot_union_cbfs,
    plot_simulation_results
)
from examples.single_integrator_2D.compare_cbfs import (
     load_unsafe_region_and_cbfs
)

def load_simulation_data(
    pickle_path: str
) -> Tuple[
    np.ndarray, # initial states
    np.ndarray, # waypoints
    List[np.ndarray], # state data list
    List[np.ndarray], # action data list
    np.ndarray, # time data
]:
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    initial_states = data["initial_states"]
    waypoints = data["waypoints"]
    state_data_list = data["state_data_list"]
    action_data_list = data["action_data_list"]
    time_data = data["time_data"]
    return (
        initial_states,
        waypoints,
        state_data_list,
        action_data_list,
        time_data
    )

def main():
    x = sym.MakeVectorContinuousVariable(2, "x")

    x_low, x_high = -4.5, 5.5
    y_low, y_high = -4.5, 4.5

    fig, ax = plt.subplots(
        #figsize=(6/1.2, 4.5/1.2)
    )
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

    # load and plot the unsafe region
    filename = "unsafe_region_and_cbfs.pkl"
    unsafe_region_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (unsafe_region, _, _, _, _) = load_unsafe_region_and_cbfs(
        x_vars=x,
        pickle_path=unsafe_region_path
    )
    plot_unsafe_region(
        ax=ax,
        x_states=x,
        unsafe_regions=[unsafe_region],
        x_range=(x_low, x_high),
        y_range=(y_low, y_high)
    )

    # load and plot the simulated trajectories for different cases:    
    case_name_list = [
        "bncbf",
        "composite_cbf",
        "switching_policy_I",
        "switching_policy_II",
    ]
    trajectory_color_list = [
        'purple',
        'lightblue',
        'orange',
        'green',
    ]
    trajectory_style_list = [
        '-',
        '-',
        '--',
        '-.',
    ]
    for case_name, color, style in zip(
        case_name_list, trajectory_color_list, trajectory_style_list
        ):
        filename = "sim_" + case_name + "_data.pkl"
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
        )
        (x0, waypoints, state_data_list,_, _) = load_simulation_data(pickle_path=data_path)

        for each_state_data in state_data_list:
            ax.plot(
                each_state_data[0, :],
                each_state_data[1, :],
                color=color,
                linestyle=style,
                linewidth=1.5
            )
    # plot the waypoints:
    ax.scatter(waypoints[:, 0], waypoints[:, 1], color = 'red', marker='x', s=100)
    # plot the initial states:
    ax.scatter(x0[:, 0], x0[:, 1], color = 'red', marker='o', s=20)

    ax.set_xlabel("x0", fontsize=16)
    ax.set_ylabel("x1", fontsize=16)
    ax.set_title("Simulation Trajectories", fontsize=16)

    legend_handles = [
        lines.Line2D([0], [0], color='purple', linestyle='-', label='BNCBF'),
        lines.Line2D([0], [0], color='lightblue', linestyle='-', label='CompCBF'),
        lines.Line2D([0], [0], color='orange', linestyle='--', label='Switching I'),
        lines.Line2D([0], [0], color='green', linestyle='-.', label='Switching II')
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=12,
    )




if __name__ == "__main__":
    main()