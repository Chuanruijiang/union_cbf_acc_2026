"""
This file defines the plot functions.
"""
import numpy as np
import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt
import pydrake.symbolic as sym
from typing import Tuple, Union, Optional, List

# basic blocks
def plot_2D_function(
    ax: matplotlib.axes.Axes,
    f: Union[sym.Polynomial, np.ndarray],
    x: Optional[np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    sampling_rate: Optional[int],
    with_contour: bool,
    color: str,
    contour_line_style='-',
    contour_line_width: float=3,
    with_region_filled: bool=False,
    alpha: Optional[float]=None,
):
    """
    Given a function f(x) defined over a 2D domain, plot the superlvel set
    f(x) >= 0. 
    f(x) can be a polynomial prevoiding the symbolic expression, or an
    array of values of f(x) over a grid: meaning that f[0] = grid_x,
    f[1] = grid_y, f[2] = gird_f_values.
    """
    if isinstance(f, sym.Polynomial):
        assert x is not None
        assert sampling_rate is not None
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], sampling_rate),
            np.linspace(y_range[0], y_range[1], sampling_rate)
            )
        grid_x_val = np.concatenate(
            [grid_x.reshape(1, -1), grid_y.reshape(1, -1)],
            axis=0
        )
        f_vals = f.EvaluateIndeterminates(x, grid_x_val)
        grid_f = f_vals.reshape(grid_x.shape)
    else:
        grid_x = f[0]
        grid_y = f[1]
        grid_f = f[2]
    
    f_contour = None
    if with_contour:
        f_contour = ax.contour(
            grid_x,
            grid_y,
            grid_f,
            levels=[0],
            colors=color,
            linestyles=contour_line_style,
            linewidths=contour_line_width
            )
    else:
        f_contour = None    
    if with_region_filled:
        f_positive_region = ax.contourf(
            grid_x,
            grid_y,
            grid_f,
            levels=[0, np.inf],
            colors=color,
            alpha=alpha
            )
    else:
        f_positive_region = None
    
    return f_contour, f_positive_region

def plot_intersection_region(
    ax: matplotlib.axes.Axes,
    p: np.ndarray,
    x: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    sampling_rate: int
):
    """
    In this function, we plot the intersection of p_i(x) >= 0 for all i.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_range[0], x_range[1], sampling_rate),
        np.linspace(y_range[0], y_range[1], sampling_rate)
        )
    grid_x_val = np.concatenate(
        [grid_x.reshape(1, -1), grid_y.reshape(1, -1)],
        axis=0
        )
    num_p = p.shape[0]
    p_grid_values = np.zeros(shape=(num_p, grid_x.shape[1], grid_x.shape[0]))
    for i in range(num_p):
        p_vals = p[i].EvaluateIndeterminates(x, grid_x_val)
        p_grid = p_vals.reshape(grid_x.shape)
        p_grid_values[i, :, :] = p_grid
    
    intersection = np.min(p_grid_values, axis=0)
    intersection_region = ax.contourf(
        grid_x,
        grid_y,
        intersection,
        levels=[0, np.inf],
        colors='black',
        alpha=0.5
    )
    return intersection_region

def plot_union_region(
    ax: matplotlib.axes.Axes,
    p: np.ndarray,
    x: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    sampling_rate: int,
    color: str,
    alpha: float
):
    """
    In this function, we plot the union of p_i(x) >= 0 for all i.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_range[0], x_range[1], sampling_rate),
        np.linspace(y_range[0], y_range[1], sampling_rate)
        )
    grid_x_val = np.concatenate(
        [grid_x.reshape(1, -1), grid_y.reshape(1, -1)],
        axis=0
        )
    num_p = p.shape[0]
    p_grid_values = np.zeros(shape=(num_p, grid_x.shape[1], grid_x.shape[0]))
    for i in range(num_p):
        p_vals = p[i].EvaluateIndeterminates(x, grid_x_val)
        p_grid = p_vals.reshape(grid_x.shape)
        p_grid_values[i, :, :] = p_grid
    
    grid_union = np.max(p_grid_values, axis=0)
    union_region = ax.contourf(
        grid_x,
        grid_y,
        grid_union,
        levels=[0, np.inf],
        colors=color,
        alpha=alpha
    )
    return union_region

def plot_compatible_region(
    ax: matplotlib.axes.Axes,
    h: np.ndarray,
    V: sym.Polynomial,
    x: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    sampling_rate: int,
    color: str,
    alpha: float
):
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_range[0], x_range[1], sampling_rate),
        np.linspace(y_range[0], y_range[1], sampling_rate)
        )
    grid_x_val = np.concatenate(
        [grid_x.reshape(1, -1), grid_y.reshape(1, -1)],
        axis=0
        )
    num_h = h.shape[0]
    h_grid_values = np.zeros(shape=(num_h, grid_x.shape[1], grid_x.shape[0]))
    for i in range(num_h):
        h_vals = h[i].EvaluateIndeterminates(x, grid_x_val)
        h_grid = h_vals.reshape(grid_x.shape)
        h_grid_values[i, :, :] = h_grid
    V_grid_values = V.EvaluateIndeterminates(x, grid_x_val)
    V_grid = V_grid_values.reshape(grid_x.shape)
    grid_compatible = np.min(np.array([np.max(h_grid_values, axis=0), V_grid]), axis=0)
    compatible_region = ax.contourf(
        grid_x,
        grid_y,
        grid_compatible,
        levels=[0, np.inf],
        colors=color,
        alpha=alpha
    )

    return compatible_region

# plotting functions for union of CBFs
def plot_unsafe_region(
    ax: plt.Axes,
    x_states: np.ndarray,
    unsafe_regions: List[np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
):
    assert isinstance(unsafe_regions, List)
    for each_intersection_unsafe_region in unsafe_regions:
        plot_intersection_region(
            ax=ax,
            p=each_intersection_unsafe_region,
            x=x_states[0:2],
        x_range=x_range,
        y_range=y_range,
        sampling_rate=500,
        color="grey",
        alpha=1.0
    )

def plot_cbf_boundaries(
    ax: plt.Axes,
    x_states: np.ndarray,
    switching_cbfs: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float]
):
    for h_i in switching_cbfs:
        plot_2D_function(
            ax=ax,
            f=h_i,
            x=x_states[0:2],
            x_range=x_range,
            y_range=y_range,
            with_contour=True,
            color="black",
            contour_line_style="--",
            contour_line_width=1,
            sampling_rate=500
        )

def plot_simulation_results(
    ax: plt.Axes,
    state_data: np.ndarray,
):
    """
    Plot the simulated trajectories.
    """
    ax.plot(
        state_data[0, :], state_data[1, :],
        label="Simulation Positional Trajectory",
        color="blue"
    )

def plot_signal_time_records(
    ax: plt.Axes,
    signal_data: np.ndarray,
    time_data: np.ndarray
):
    """
    Plot the action and time records.
    """
    ax.plot(
        time_data,
        signal_data,
        color="blue"
    )


