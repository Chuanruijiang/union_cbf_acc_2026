import os
import pickle
import numpy as np
import pydrake.symbolic as sym
from typing import Tuple

from union_cbf_base.utils import (
    deserialize_polynomial,
    Degree as DegreeII
)
from union_cbf_base.union_cbf_II import (
    CbfFeasibilityLagrangianDegrees
)
from union_cbf_base.synthesis_base import (
    find_2d_linear_hocbfs,
    save_synthesized_hocbfs
)
from examples.double_integrator.dynamics import (
    DoubleIntegratorPlant2D
)

def load_unsafe_region(
    x_vars: np.ndarray,
    pickle_path: str
) -> Tuple[
    np.ndarray, # unsafe points for switching CBFs
    np.ndarray  # unsafe polys for normal CBFs.
]:
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)

    unsafe_points = None
    static_unsafe = None
    # load the cbf data:
    x_set = sym.Variables(x_vars)
    assert "unsafe_points" in data.keys()
    unsafe_points = data["unsafe_points"]
    if "static_unsafe" in data.keys():
        static_unsafe = np.array([
            deserialize_polynomial(h_i, x_set)
            for h_i in data["static_unsafe"]
            ])
    
    return unsafe_points, static_unsafe

def main(save_results: bool):
    x = sym.MakeVectorContinuousVariable(4, "x")
    filename = "double_integrator_2D_unsafe_region.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (   unsafe_points,
        static_unsafe_polys
    ) = load_unsafe_region(
        x_vars=x,
        pickle_path=data_path
    )
    double_integrator = DoubleIntegratorPlant2D()
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    static_cbfs = -static_unsafe_polys
    relative_degree = [2, 2, 2]
    alphas = [
        [0.1, 1.0],
        [0.1, 1.0],
        [0.1, 1.0]
    ]
    intersection_lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis=[
            [DegreeII(x=2, y=2, c=0) for _ in range(relative_degree[i])]
            for i in range(len(relative_degree))
        ],
        lambda_y=[
            DegreeII(x=2, y=0, c=0),
            DegreeII(x=2, y=0, c=0)
            ],
        xi_y=DegreeII(x=2, y=0, c=0),
        state_eq=None,
    )

    initial_position = np.array([-6, -6])
    waypoint = np.array([0, 0])
    interested_states = np.vstack((initial_position, waypoint))
    switching_cbfs = find_2d_linear_hocbfs(
        x_vars=x,
        f=f,
        g=g,
        control_limits=(A, c),
        static_cbfs=static_cbfs,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        unsafe_points=unsafe_points,
        unsafe_polys=None,
        initial_angle=np.pi/2,
        angle_diff=np.pi/4,
        iter_num=3,
        interested_states=interested_states,
        lagrangian_degrees=intersection_lagrangian_degrees,
        unsafe_poly_lagrangian_x_degrees=None,
        cbf_lagrangian_x_degree=None,
    )

    if save_results:
        filename = "double_integrator_2D_switching_cbf_synthesis.pkl"
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
        )
        save_synthesized_hocbfs(
            x_vars=x,
            switching_cbfs=switching_cbfs,
            static_cbfs=static_cbfs,
            pickle_path=save_path
        )


if __name__ == "__main__":
    main(save_results=True)
    