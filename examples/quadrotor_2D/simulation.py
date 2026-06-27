import os
import pickle
import numpy as np
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt

import pydrake.symbolic as sym
import pydrake.systems.analysis
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.systems.primitives import (
    LogVectorOutput, VectorLogSink
)
import union_cbf_base.utils as utils
from examples.quadrotor_2D.dynamics import (
    QuadrotorDynamics2D,
    NominalController,
    StateConverter
)
from union_cbf_base.controller import SwitchingCBFController
from union_cbf_base.plot import (
    plot_simulation_results,
    plot_environement
)


def build_diagram() -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs the waypoints
    np.ndarray, # outputs the state variable x
    np.ndarray, # outputs the static cbfs
    np.ndarray, # outputs the switching cbfs
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(6, "x")
    z = sym.MakeVectorContinuousVariable(7, "z")
    
    # the plant block:
    quadrotor = builder.AddSystem(QuadrotorDynamics2D())
    state_logger = LogVectorOutput(quadrotor.get_output_port(), builder)
    
    # the nominal controller block (LQR):
    Q = np.eye(6)
    Q[2,2] = 1.0
    Q[3,3] = 20
    Q[4,4] = 20
    Q[5,5] = 50
    R = np.eye(2)
    nominal_controller = builder.AddSystem(
        NominalController(
            Q=Q,
            R=R
        )
    )
    
    # the state converter block:
    state_converter = builder.AddSystem(StateConverter())

    # specify the CBFs:
    static_cbfs = np.array([
        sym.Polynomial(z[3] + (1 - np.cos(np.pi / 3)) - (z[6])**2 + (np.pi/2)**2)
    ])
    switching_cbfs = np.array([
        sym.Polynomial(-z[0] - z[1] - 15),
        sym.Polynomial(-z[1] - 3),
        sym.Polynomial(z[0] - z[1] + 3),
    ])
    alpha = [
        [1.0],
        [1.0, 10.0]
    ]
    relative_degree = [1, 2]

    # the switching CBF-QP controller block:
    quadrotor_poly_system = QuadrotorDynamics2D()
    f, g = quadrotor_poly_system.trig_poly_affine_dynamics(z)
    (A, c) = quadrotor_poly_system.control_limits(output_polys=False)
    cbf_qp_controller = builder.AddSystem(
        SwitchingCBFController(
            x=z,
            f=f,
            g=g,
            static_cbfs=static_cbfs,
            switching_cbfs=switching_cbfs,
            alpha=alpha,
            relative_degree=relative_degree,
            control_limits=(A, c),
            switching_policy_id=2,
            switching_policy_param=(0.5, 0.6),
            solver_id=None,
            solver_options=None
        )
    )
    action_logger = LogVectorOutput(
        cbf_qp_controller.action_output_port(), builder
        )

    # connect the blocks:
    builder.Connect(
        quadrotor.get_output_port(0),
        nominal_controller.get_input_port(0)
        )
    builder.Connect(
        quadrotor.get_output_port(0),
        state_converter.get_input_port(0)
        )
    builder.Connect(
        state_converter.get_output_port(0),
        cbf_qp_controller.get_input_port(0)
        )
    builder.Connect(
        nominal_controller.get_output_port(0),
        cbf_qp_controller.get_input_port(1)
        )
    builder.Connect(
        cbf_qp_controller.action_output_port(),
        quadrotor.get_input_port(0)
        )
            
    # finialize the diagram:
    diagram = builder.Build()
    return (
        diagram,
        state_logger,
        action_logger,
        np.array([[0.0, 0.0]]), # waypoints since the nominal control is LQR
        z,
        None, # we won't plot the static CBF in this example since it is not in positiion domain
        switching_cbfs
    )

def build_diagram_nominal_only() -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs the waypoints
    np.ndarray, # outputs the state variable x
    np.ndarray, # outputs the static cbfs
    np.ndarray, # outputs the switching cbfs
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(6, "x")
    
    # the plant block:
    quadrotor = builder.AddSystem(QuadrotorDynamics2D())
    state_logger = LogVectorOutput(quadrotor.get_output_port(), builder)
    
    Q = np.eye(6)
    Q[3,3] = 50
    Q[4,4] = 50
    Q[5,5] = 30
    R = np.eye(2)
    nominal_controller = builder.AddSystem(
        NominalController(
            Q=Q,
            R=R
        )
    )
    action_logger = LogVectorOutput(nominal_controller.get_output_port(), builder)
    
    # connect the blocks:
    builder.Connect(
        quadrotor.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        nominal_controller.get_output_port(0),
        quadrotor.get_input_port(0)
    )
    
    # finialize the diagram:
    diagram = builder.Build()
    return (
        diagram,
        state_logger,
        action_logger,
        np.array([[0.0, 0.0]]), # waypoints since the nominal control is LQR
        x,
        None,
        None
    )

def simulate(
    total_states: np.ndarray, duration: float
) -> Tuple[
    np.ndarray, # state data
    np.ndarray, # action data
    np.ndarray, # time data
    np.ndarray, # waypoints
    np.ndarray, # plant states variables
    np.ndarray, # static cbfs
    np.ndarray, # switching cbfs
    ]:
    # the state logger and action logger still refers
    # to the plant states and the plant input actions.
    (diagram,
     state_logger,
     action_logger,
     waypoints,
     plant_state_variables,
     static_cbfs,
     switching_cbfs) = build_diagram()

    # (diagram, 
    #  state_logger,
    #  action_logger,
    #  waypoints,
    #  plant_state_variables,
    #  static_cbfs,
    #  switching_cbfs,
    #  ) = build_diagram_nominal_only()
    
    simulator = pydrake.systems.analysis.Simulator(diagram)
    context = simulator.get_mutable_context()
    # total state refers to all the states in the diagram
    # these not only include the plant states, but also
    # the integrator states in the PID controllers.
    context.SetContinuousState(total_states)
    # configure the simulator
    simulator_config = pydrake.systems.analysis.SimulatorConfig(
        integration_scheme="runge_kutta3"
    )
    pydrake.systems.analysis.ApplySimulatorConfig(simulator_config, simulator)
    # run the simulation
    simulator.AdvanceTo(duration)
    # collect the data
    state_data = state_logger.FindLog(simulator.get_context()).data()
    action_data = action_logger.FindLog(simulator.get_context()).data()
    time_data = state_logger.FindLog(simulator.get_context()).sample_times()
    return (
        state_data,
        action_data,
        time_data,
        waypoints,
        plant_state_variables,
        static_cbfs,
        switching_cbfs
    )

def plot_trajectories(
    state_data: np.ndarray,
    waypoints: np.ndarray,
    static_cbfs: Optional[np.ndarray]=None,
    switching_cbfs: Optional[np.ndarray]=None,
    x_extended: Optional[np.ndarray]=None,
    with_unsafe_regions: bool = True,
):
    fig = plt.figure()
    ax = fig.add_subplot()
    x_low = -20.0
    x_high = 5.0
    y_low = -10.0
    y_high = 5.0
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(r"position x0", fontsize=16)
    ax.set_ylabel(r"position x1", fontsize=16)
    ax.set_xticks([-20, -15, -10, -5, 0, 5])
    ax.set_xticklabels(
        [r"$-20$", r"$-15$", r"$-10$", r"$-5$", r"$0$", r"$5$"]
    )
    ax.set_yticks([-10, -5, 0, 5])
    ax.set_yticklabels(
        [r"$-10$", r"$-5$", r"$0$", r"$5$"]
    )
    ax.tick_params(axis="both", which="major", labelsize=14)

    # plot the simulated trajectories:
    plot_simulation_results(
        ax=ax,
        state_data=state_data
    )
    # mark the waypoints:
    ax.scatter(
        waypoints[0, 0], waypoints[0, 1],
        color="red",
        marker="*",
        s=200,
        label="waypoint"
    )
    if with_unsafe_regions:
        plot_environement(
            ax=ax,
            x_states=x_extended,
            union_unsafe_regions=(-static_cbfs 
                                  if static_cbfs is not None 
                                  else None),
            intersection_unsafe_regions=([-switching_cbfs]
                                        if switching_cbfs is not None
                                        else None),
            x_range=[x_low, x_high],
            y_range=[y_low, y_high],
        )

def plot_control_inputs(
    num_control: int,
    action_data: np.ndarray,
    time_data: np.ndarray,
    control_limits: Tuple[float, float],
    time_limit:float,
    figsize: Tuple[float, float] = (5, 5)
):
    fig, axs = plt.subplots(num_control, 1, figsize=figsize)
    for i in range(num_control):
        axs[i].plot(
            time_data,
            action_data[i,:],
            color="blue"
        )
        # plot the control limits:
        axs[i].axhline(
            y=control_limits[0],
            color="red",
            linestyle="--",
            label="control lower bound"
        )
        axs[i].axhline(
            y=control_limits[1],
            color="red",
            linestyle="-.",
            label="control upper bound"
        )
        axs[i].set_xlim(time_data[0], time_limit)
        axs[i].set_ylim(control_limits[0] - 0.5, control_limits[1] + 0.5)
        axs[i].set_xlabel(r"time (s)", fontsize=16)
        axs[i].set_ylabel(r"action $u_{}$".format(i), fontsize=16)
        axs[i].tick_params(axis="both", which="major", labelsize=14)

def save_data(
    pickle_path: str,
    state_data: np.ndarray,
    action_data: np.ndarray,
    time_data: np.ndarray,
    waypoints: np.ndarray,
    static_cbfs: Optional[np.ndarray],
    switching_cbfs: Optional[np.ndarray],
    x_vars: np.ndarray,
):
    """
    Save the switching and static CBFs in a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    
    # save the trajectory data:
    data = {
        "state_data": state_data,
        "action_data": action_data,
        "time_data": time_data,
        "waypoints": waypoints
    }
    # save the cbf data:
    x_set = sym.Variables(x_vars)
    if static_cbfs is not None:
        data["static_cbfs"] = [
            utils.serialize_polynomial(h_i, x_set) 
            for h_i in static_cbfs
            ]
    if switching_cbfs is not None:
        data["switching_cbfs"] = [
            utils.serialize_polynomial(h_i, x_set)
            for h_i in switching_cbfs
            ]
    
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

def run_simulation_and_plot():
    # initial states for the whole diagram
    total_states = np.array([-17.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    duration = 30.0
    (state_data,
     action_data,
     time_data,
     waypoints,
     z_vars,
     static_cbfs,
     switching_cbfs) = simulate(
        total_states=total_states,
        duration=duration
    )
    # save the simulation data:
    filename = "quadrotor_2D_simulation_results.pkl"
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data",
        filename
    )
    save_data(
        pickle_path=save_path,
        state_data=state_data,
        action_data=action_data,
        time_data=time_data,
        waypoints=waypoints,
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
        x_vars=z_vars
    )

if __name__ == "__main__":
    run_simulation_and_plot()

