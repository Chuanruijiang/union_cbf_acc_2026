"""
This file defines the plot functions.
"""
import numpy as np
import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt
import pydrake.symbolic as sym
from typing import Tuple, Union, Optional


def plot_2D_function(
    ax: matplotlib.axes.Axes,
    f: Union[sym.Polynomial, np.ndarray],
    x: Optional[np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    sampling_rate: int
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
    
    f_contour = ax.contour(grid_x, grid_y, grid_f, levels=[0], colors='blue')
    f_positive_region = ax.contourf(
        grid_x,
        grid_y,
        grid_f,
        levels=[0, np.inf],
        colors='blue',
        alpha=0.4
        )
    
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

    



# def main():
#     x = sym.MakeVectorContinuousVariable(2, "x")
#     unsafe_polys = np.array([
#         sym.Polynomial(x[0] + 5),
#         sym.Polynomial(-x[0] - 3),
#         sym.Polynomial(x[1] + 1),
#         sym.Polynomial(-x[1] + 1),
#     ])
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.set_xlabel(r"$x_1$", fontsize=16)
#     ax.set_ylabel(r"$x_2$", fontsize=16)
#     ax.set_xticks([-15, -10, -5, 0, 5])
#     ax.set_yticks([-10, -5, 0, 5, 10])
#     ax.set_xticklabels([r"-15", r"-10", r"-5", r"0", r"5"], fontsize=16)
#     ax.set_yticklabels([r"-10", r"-5", r"0", r"5", r"10"], fontsize=16)
#     ax.set_xlim(-15, 5)
#     ax.set_ylim(-5, 5)
#     # plot the unsafe region:
#     unsafe_region = plot_intersection_region(
#         ax=ax,
#         p=unsafe_polys,
#         x=x,
#         x_range=(-15, 5),
#         y_range=(-5, 5),
#         sampling_rate=1000
#     )
#     fig.show()

# if __name__ == "__main__":
#     main()