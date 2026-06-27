"""
This script generates a complex unsafe region for the double integrator system,
and samples points from this unsafe region. The sampled unsafe points are then
saved in a pickle file for the switching CBFs validity checking.
"""
import os
import pickle
import numpy as np
import pydrake.symbolic as sym
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple

from union_cbf_base.utils import serialize_polynomial
from union_cbf_base.plot import(
    plot_environement,
    plot_cbf_boundaries
)

# The following function generates a rectangular region
def rotated_rectangle(
        x: np.ndarray,
        center: np.ndarray,
        width: float,
        height: float,
        theta: float,
    ) -> np.ndarray:
    """
    Creates a set of 4 linear polynomials such that 
    p(x) >= 0 defines the interior of a rotated 
    rectangle.
    
    Args:
        x: np.ndarray of symbolic variables [x0, x1]
        center: np.array [c0, c1]
        width: float (size along the local u-axis)
        height: float (size along the local v-axis)
        theta: float (rotation angle in radians)
        
    Returns:
        np.array of sym.Polynomial
    """
    c = np.array(center)
    
    # Unit vectors for the local coordinate system
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])
    
    # Polynomials represent the distance from the edges pointing inward.
    # The general form is: (width/2) - (x - c).dot(u) >= 0
    
    # 1. Width-side boundaries
    p1 = sym.Polynomial(
        (width / 2.0) - ( (x[0] - c[0]) * u[0] + (x[1] - c[1]) * u[1] )
        )
    p2 = sym.Polynomial(
        (width / 2.0) + ( (x[0] - c[0]) * u[0] + (x[1] - c[1]) * u[1] )
        )
    
    # 2. Height-side boundaries
    p3 = sym.Polynomial(
        (height / 2.0) - ( (x[0] - c[0]) * v[0] + (x[1] - c[1]) * v[1] )
        )
    p4 = sym.Polynomial(
        (height / 2.0) + ( (x[0] - c[0]) * v[0] + (x[1] - c[1]) * v[1] )
        )
    
    return np.array([p1, p2, p3, p4])

# using the rectangle function above, we define the complex
# unsafe region that will be excluded by swiching CBFs.
def generate_constrained_shapes(
    x_vars: np.ndarray
    ) -> List[np.ndarray]:
    shapes_polynomials = []

    # obstacle 1: 
    shape_1 = rotated_rectangle(
        x=x_vars,
        center=np.array([1.5, -6.0]),
        width=1.5,
        height=4.0,
        theta=0.0
    )
    shapes_polynomials.append(shape_1)

    # obstacle 2: a rectangle
    shape_2 = rotated_rectangle(
        x=x_vars,
        center=np.array([-1.5, -7.0]),
        width=2.0,
        height=4.0,
        theta=0.0
    )
    shapes_polynomials.append(shape_2)

    # obstacle 3: a rectangle, width spans from -3 to -1, 
    # height spans from -5 to -2:  
    shape_3 = rotated_rectangle(
        x=x_vars,
        center=np.array([-2.0, -3.5]),
        width=2.0,
        height=3.0,
        theta=0.0
    )
    shapes_polynomials.append(shape_3)

    # obstacle 4: a rectangle, width spans from -2 to 0.5,
    # height spans from -3.5 to -1:
    shape_4 = rotated_rectangle(
        x=x_vars,
        center=np.array([-0.7, -2.1]),
        width=2.5,
        height=2.5,
        theta=0.0
    )
    shapes_polynomials.append(shape_4)

    # obstacle 5: a rotated rectangle
    shape_5 = rotated_rectangle(
        x=x_vars,
        center=np.array([0.0, -5.0]),
        width=4.0,
        height=1.5,
        theta=np.pi/2.5
    )
    shapes_polynomials.append(shape_5)

    # obstacle 6: a rotated rectangle
    shape_6 = rotated_rectangle(
        x=x_vars,
        center=np.array([-3.0, -7.0]),
        width=1.5,
        height=1.5,
        theta=-np.pi/4
    )
    shapes_polynomials.append(shape_6)

    # obstacle 7: a rotated rectangle
    shape_7 = rotated_rectangle(
        x=x_vars,
        center=np.array([-4.0, -2.5]),
        width=2.5,
        height=1.3,
        theta=np.pi/4
    )
    shapes_polynomials.append(shape_7)

    # obstacle 8: a rotated rectangle
    shape_8 = rotated_rectangle(
        x=x_vars,
        center=np.array([-4.5, -7.0]),
        width=1.5,
        height=4.0,
        theta=0.0
    )
    shapes_polynomials.append(shape_8)
            
    return shapes_polynomials

# This function samples points from the complex unsafe region.
def sample_unsafe_points(
        x_vars: np.ndarray,
        obstacle_list: List[np.ndarray],
        num_samples: int,
        x0_range: Tuple[float, float],
        x1_range: Tuple[float, float],
    ) -> np.ndarray:
    """
    Samples points and collect those that fall inside the unsafe regions
    The unsafe region is defined by the constrainted shapes in the 
    function above.
    
    Args:
        x_vars: 
            np.ndarray of symbolic variables corresponding to the state variables.
        obstacle_list: 
            List of np.arrays, where each array contains sym.Polynomials.
            A point is in the obstacle if ALL polynomials in the array
            are >= 0.
        num_samples: 
            Total number of samples in each dimension.
        x0_range, x1_range: 
            The bounding box for sampling.
    """
    # 1. Generate samples across the domain
    x0_samples = np.linspace(x0_range[0], x0_range[1], num_samples)
    x1_samples = np.linspace(x1_range[0], x1_range[1], num_samples)
    points = np.array(np.meshgrid(x0_samples, x1_samples)).T.reshape(-1, 2)
    
    # 2. Check each point
    inside_points = []
    for pt in points:
        env = {x_vars[0]: pt[0], x_vars[1]: pt[1]}
        is_inside_any = False
        for obstacle_polys in obstacle_list:
            # Check if point satisfies ALL polynomials for THIS obstacle
            # If a point is in an obstacle, then it's also in the unsafe
            # region.
            if all(p.Evaluate(env) >= -1e-9 for p in obstacle_polys):
                is_inside_any = True
                break 
        if is_inside_any:
            inside_points.append(pt)
    
    return np.array(inside_points)

# this function saves the unsafe points for the switching CBFs
# validity checking.
def save_unsafe_region(
    unsafe_points: np.ndarray,
    x_vars: np.ndarray,
    static_unsafe: np.ndarray,
    pickle_path: str
): 
    """
    Save the unsafe region for switching CBFs and the 
    static unsafe bound for static CBFs in a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"

    data = {
        "unsafe_points": unsafe_points,
        "static_unsafe": [
            serialize_polynomial(h_i, sym.Variables(x_vars)) 
            for h_i in static_unsafe
        ]
    }

    if os.path.exists(pickle_path):
        overwrite_cmd = input(
            f"File {pickle_path} already exists. Overwrite the file? Press [Y/n]:"
        )
        if overwrite_cmd in ("Y", "y"):
            save_cmd = True
        else:
            save_cmd = False
    else:
        save_cmd = True

    if save_cmd:
        with open(pickle_path, "wb") as handle:
            pickle.dump(data, handle)

# this function plots the boundaries of the normal CBFs and a
# circular single high-degree CBF.
def show_circular_cbfs():
    x = sym.MakeVectorContinuousVariable(4, "x")
    
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

    static_unsafe_bound = np.array([
        -sym.Polynomial(x[0] + 6.5),
        -sym.Polynomial(-x[1] + 0.5)
    ])
    
    obstacles = generate_constrained_shapes(x_vars=x)

    plot_environement(
        ax,
        x_states=x,
        union_unsafe_regions=[static_unsafe_bound],
        intersection_unsafe_regions=obstacles,
        x_range=(x_low, x_high),
        y_range=(y_low, y_high),
    )

    plot_cbf_boundaries(
        ax,
        x_states=x,
        static_cbfs=-static_unsafe_bound,
        switching_cbfs=np.array([
            sym.Polynomial((x[0] + 1.4)**2 + (x[1] + 5.2)**2 - 4.8**2)
        ]),
        x_range=(x_low, x_high),
        y_range=(y_low, y_high),
    )

    legend_handles = [
        Line2D([], [], color="black", linewidth=2, linestyle="-",
               label=r"$h(x) = 0$"),
        Line2D([], [], color="black", linewidth=2, linestyle="--",
               label=r"$b(x) = 0$"),
    ]
    ax.legend(handles=legend_handles, fontsize=14, loc="lower right")
    

# finally, we can use this function to see how the unsafe region
# looks like, and also to save the unsafe points for synthesis.
def show_environment_unsafe_points():
    x = sym.MakeVectorContinuousVariable(4, "x")
    
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

    static_unsafe_bound = np.array([
        -sym.Polynomial(x[0] + 6.5),
        -sym.Polynomial(-x[1] + 0.5)
    ])
    
    obstacles = generate_constrained_shapes(x_vars=x)

    plot_environement(
        ax,
        x_states=x,
        union_unsafe_regions=[static_unsafe_bound],
        intersection_unsafe_regions=obstacles,
        x_range=(x_low, x_high),
        y_range=(y_low, y_high),
    )

    plot_cbf_boundaries(
        ax,
        x_states=x,
        static_cbfs=-static_unsafe_bound,
        switching_cbfs=None,
        x_range=(x_low, x_high),
        y_range=(y_low, y_high),
    )

    unsafe_points = sample_unsafe_points(
        x_vars=x,
        obstacle_list=obstacles,
        num_samples=100,
        x0_range=(x_low, x_high),
        x1_range=(y_low, y_high)
    )
    
    # after getting the unsafe points, show these points
    ax.scatter(
        unsafe_points[:, 0],
        unsafe_points[:, 1],
        color="red",
        s=10,
        alpha=0.5,
        label="Sampled Unsafe Points"
    )

    # save the unsafe points:
    filename = "double_integrator_2D_unsafe_region.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    save_unsafe_region(
        unsafe_points=unsafe_points,
        x_vars=x,
        static_unsafe=static_unsafe_bound,
        pickle_path=data_path
    )




if __name__ == "__main__":
    show_circular_cbfs()
    show_environment_unsafe_points()



