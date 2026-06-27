"""
In this simulation, we use a double integrator model with limited control
inputs to track a target while avoiding unsafe regions using a switching
HOCBFs. The setup is the following:

The map shows a 2D plane L shapeed corridor, we start from the position 
(-6, -6) the target is to get tothe origin, but we always need stay in 
the corridor and avoid the unsafe region. The outter bounds of the corridor
are defined by static CBFs:
    h1(x) = x0 + 7 >= 0
    h2(x) = -x1 + 1 >= 0
the inner bounds of the corridor are defined by union CBFs:
    h3_1(x) = -x0 - 5 >= 0
    h3_2(x) = x1 + 1 >= 0
"""
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
from examples.double_integrator.dynamics import (
    DoubleIntegratorPlant2D,
    LQRNominalController2D
)
from union_cbf_base.controller import SwitchingCBFController

# The static CBFs, switching CBFs, system dynamics, and 
# alpha, relative degree parameters are set in function
# build_diagram().
# The initial states and the simulation duration are set
# in function run_simulation().

def load_cbfs(
    x_vars: np.ndarray,
    pickle_path: str
) -> Tuple[
    Optional[np.ndarray], # static cbfs
    np.ndarray # switching cbfs
]:
    """
    Load the switching and static CBFs in a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    
    static_cbfs = None
    switching_cbfs = None
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    
    # load the cbf data:
    x_set = sym.Variables(x_vars)
    if "static_cbfs" in data.keys():
        static_cbfs = np.array([
            utils.deserialize_polynomial(h_i, x_set)
            for h_i in data["static_cbfs"]
            ])
    switching_cbfs = np.array([
        utils.deserialize_polynomial(h_i, x_set)
        for h_i in data["switching_cbfs"]
        ])
    
    return static_cbfs, switching_cbfs

def build_diagram() -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs waypoints
    np.ndarray, # outputs the state variable x
    np.ndarray, # outputs the static cbfs
    np.ndarray # outputs the switching cbfs
]:
    # we don't need to set the waypoints for double integrator
    # since the nominal controller is a LQR controller and
    # it directly tracks the origin.
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(4, "x")
    poly_plant = DoubleIntegratorPlant2D()
    f, g = poly_plant.affine_dynamics(x)
    (A, c) = poly_plant.control_input_limits(output_poly=False)
    # load the static CBFs and switching CBFs:
    filename = "double_integrator_2D_switching_cbf_synthesis.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (   static_cbfs,
        switching_cbfs
    ) = load_cbfs(
        x_vars=x,
        pickle_path=data_path
    )

    #!!!ATENTION: The loaded switching CBFs does not following the order
    # of the safety filter switching policy.

    switching_cbfs = switching_cbfs[::-1]

    relative_degree=np.array([2, 2, 2])
    alpha = [
        [0.01, 0.1],
        [0.01, 0.1],
        [0.01, 0.1],
    ]
    
    # the plant block:
    double_integrator = builder.AddSystem(DoubleIntegratorPlant2D())
    state_logger = LogVectorOutput(double_integrator.get_output_port(), builder)
    
    # the nominal controller block:
    Q = np.zeros((4, 4))
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    Q[2, 2] = 50.0
    Q[3, 3] = 50.0
    R = np.eye(2)
    nominal_controller = builder.AddSystem(
        LQRNominalController2D(Q=Q, R=R)
    )
    
    # the switching CBF-QP controller block:
    switching_cbf_qp_controller = builder.AddSystem(
        SwitchingCBFController(
            x=x,
            f=f,
            g=g,
            static_cbfs=static_cbfs,
            switching_cbfs=switching_cbfs,
            relative_degree=relative_degree,
            alpha = alpha,
            control_limits=(A, c),
            switching_policy_id=2,
            switching_policy_param=(0.5, 0.6),
            solver_id=None,
            solver_options=None
        )
    )
    action_logger = LogVectorOutput(switching_cbf_qp_controller.action_output_port(), builder)
    
    # connect the blocks:
    builder.Connect(
        double_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        double_integrator.get_output_port(0),
        switching_cbf_qp_controller.get_input_port(0)
    )
    builder.Connect(
        nominal_controller.get_output_port(0),
        switching_cbf_qp_controller.get_input_port(1)
    )
    builder.Connect(
        switching_cbf_qp_controller.get_output_port(0),
        double_integrator.get_input_port(0)
    )
    
    # finialize the diagram:
    diagram = builder.Build()
    return (
        diagram,
        state_logger,
        action_logger,
        np.array([[0.0, 0.0]]), # waypoints since the nominal control is LQR
        x,
        static_cbfs,
        switching_cbfs
    )

def build_diagram_nominal_only() -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs the waypoints
    np.ndarray # outputs the state variable x
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(4, "x")
    
    # the plant block:
    double_integrator = builder.AddSystem(DoubleIntegratorPlant2D())
    state_logger = LogVectorOutput(double_integrator.get_output_port(), builder)
    
    nominal_controller = builder.AddSystem(
        LQRNominalController2D(
            Q=np.eye(4),
            R=np.eye(2),
            x_states=x
        )
    )
    action_logger = LogVectorOutput(nominal_controller.get_output_port(), builder)
    
    # connect the blocks:
    builder.Connect(
        double_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        nominal_controller.get_output_port(0),
        double_integrator.get_input_port(0)
    )
    
    # finialize the diagram:
    diagram = builder.Build()
    return (
        diagram,
        state_logger,
        action_logger,
        np.array([[0.0, 0.0]]), # waypoints since the nominal control is LQR
        x,
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
     switching_cbfs
     ) = build_diagram()
    
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

def run_simulation():
    x0 = np.array([-6, -6, 0.0, 0.0])
    # x0 = np.array([-5.40731557e+00, -3.64643078e-02,  5.33116460e-04,  5.15343240e-03])
    # x0 = np.array([-1.97453901e+00,  4.82097586e-01,  1.96900565e-04,  1.79020754e-04])
    # x0 = np.array([-3.0298902,  -0.47742158,  0.01075037,  0.00977421])
    total_states = x0
    duration = 300.0
    (state_data, 
     action_data, 
     time_data,
     waypoints,
     plant_state_variables,
     static_cbfs,
     switching_cbfs
    ) = simulate(total_states, duration)
    
    filename = "double_integrator_2D_simulation_data_II.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    save_data(
        pickle_path=data_path,
        state_data=state_data,
        action_data=action_data,
        time_data=time_data,
        waypoints=waypoints,
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
        x_vars=plant_state_variables
    )
    



if __name__ == "__main__":
    run_simulation()

