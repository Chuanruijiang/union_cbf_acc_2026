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
    reference_hight = data["reference_hight"]
    state_data_list = data["state_data_list"]
    action_data_list = data["action_data_list"]
    time_data = data["time_data"]
    return (
        initial_states,
        reference_hight,
        state_data_list,
        action_data_list,
        time_data
    )

def main():
    x = sym.MakeVectorContinuousVariable(2, "x")

    x_low, x_high = -3.0, 3.0
    y_low, y_high = -2.5, 3.5

    fig, ax = plt.subplots(
        figsize=(6/1.2, 4.5/1.2)
    )

    # plot the unsafe region
    unsafe_region = np.array([
        sym.Polynomial(-x[0] + x[1]),
        sym.Polynomial((x[0] + 2)**2 + (x[1] + 2)**2 - 8)
    ])
    plot_unsafe_region(
        ax=ax,
        x_states=x,
        unsafe_regions=[unsafe_region],
        x_range=(x_low, x_high),
        y_range=(y_low, y_high)
    )
    plot_union_cbfs(
        ax=ax,
        x_states=x,
        switching_cbfs=-unsafe_region,
        x_range=(x_low, x_high),
        y_range=(y_low, y_high),
        contour_line_width=2.0
    )

    # load and plot the simulated trajectories:    
    filename = "sim_const_vel_data.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (x0, reference_hight, state_data_list,_, _) = load_simulation_data(pickle_path=data_path)

    for each_state_data in state_data_list:
        ax.plot(
            each_state_data[0, :],
            each_state_data[1, :],
            color="blue",
            linestyle='-',
            linewidth=2
        )
    # plot the reference line:
    ax.plot(
        [x_low, x_high],
        [reference_hight, reference_hight],
        color="red",
        linestyle='--',
        linewidth=2
    )
    
    # plot the initial states:
    ax.scatter(x0[:, 0], x0[:, 1], color = 'red', marker='o', s=20)

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("x0", fontsize=16)
    ax.set_ylabel("x1", fontsize=16)
    ax.set_title("Simulation Trajectories", fontsize=16)

    legend_handles = [
        lines.Line2D([0], [0], color='black', lw=1.5, linestyle='--', label=r'$h_i(x)=0$'),
        lines.Line2D([0], [0], color='blue', lw=1.5, label='Trajectories'),
        lines.Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='Reference Hight')
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=10,
    )

if __name__ == "__main__":
    main()

