# """
# This script compares the simulation results using different CBfs.
# We load the unsafe region and the union of CBFs, as well as the 
# compositional CBF and the high degree poynomial CBF.
# We choose 3 different initial conditions, and then simulate the
# closed loop system with the union CBF switching policy I and II,
# the closed loop system with the compositional CBF, and the closed
# loop system with the high degree polynomial CBF. We save the data
# of the simulation and then plot these results in the same plot.
# """

import os
import pickle
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

import pydrake.symbolic as sym
import pydrake.systems.analysis
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.systems.primitives import (
    LogVectorOutput, VectorLogSink
)

from union_cbf_base.controller import SwitchingCBFController
from examples.single_integrator_2D.dynamics import (
    SingleIntegrator2D,
    NominalController
)
from examples.single_integrator_2D.compared_controllers import (
    BncbfController,
    CompositeCbfController
)
from examples.single_integrator_2D.compare_cbfs import (
    load_unsafe_region_and_cbfs
)

def build_diagram_nominal() -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs waypoints
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = SingleIntegrator2D()
    (f, g) = plant.affine_dynamics(x)
    (A, c) = plant.control_limits(output_poly=False)
    waypoints = np.array([[4.5, 0.0]])
    
    # the plant block:
    single_integrator = builder.AddSystem(SingleIntegrator2D())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        NominalController(
            waypoints=waypoints,
            gain=1.0
        )
    )
    action_logger = LogVectorOutput(nominal_controller.get_output_port(), builder)

    builder.Connect(
        nominal_controller.get_output_port(0),
        single_integrator.get_input_port(0)
    )

    builder.Connect(
        single_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    diagram = builder.Build()

    return(
        diagram,
        state_logger,
        action_logger,
        waypoints
    )
       
def build_diagram_switching_policy_I(
    init_cbf_index: int=0
) -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs waypoints
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = SingleIntegrator2D()
    (f, g) = plant.affine_dynamics(x)
    (A, c) = plant.control_limits(output_poly=False)
    alpha = 1.0
    waypoints = np.array([[4.5, 0.0]])

    # the plant block:
    single_integrator = builder.AddSystem(SingleIntegrator2D())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        NominalController(
            waypoints=waypoints,
            gain=1.0
        )
    )

    # load the union of CBFs:
    filename = "unsafe_region_and_cbfs.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (_, union_cbfs, _, _, _) = load_unsafe_region_and_cbfs(
        x_vars=x,
        pickle_path=data_path
    )
    # the switching CBF controller block:
    switching_cbf_controller = builder.AddSystem(
        SwitchingCBFController(
            x=x,
            f=f,
            g=g,
            cbfs=union_cbfs,
            alpha=alpha,
            control_limits=(A, c),
            switching_policy_id=1,
            switching_threshold=(2.1, 2.3),
            initial_cbf_index=init_cbf_index
        )
    )
    action_logger = LogVectorOutput(switching_cbf_controller.get_output_port(), builder)
    
    builder.Connect(
        nominal_controller.get_output_port(0),
        switching_cbf_controller.get_input_port(1)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        switching_cbf_controller.get_input_port(0)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        switching_cbf_controller.get_output_port(0),
        single_integrator.get_input_port(0)
    )
    diagram = builder.Build()

    return(
        diagram,
        state_logger,
        action_logger,
        waypoints
    )
    
def build_diagram_switching_policy_II(
    init_cbf_index: int=0
) -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs waypoints
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = SingleIntegrator2D()
    (f, g) = plant.affine_dynamics(x)
    (A, c) = plant.control_limits(output_poly=False)
    alpha = 1.0
    waypoints = np.array([[4.5, 0.0]])

    # the plant block:
    single_integrator = builder.AddSystem(SingleIntegrator2D())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        NominalController(
            waypoints=waypoints,
            gain=1.0
        )
    )

    # load the union of CBFs:
    filename = "unsafe_region_and_cbfs.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (_, union_cbfs, _, _, _) = load_unsafe_region_and_cbfs(
        x_vars=x,
        pickle_path=data_path
    )
    # the switching CBF controller block:
    switching_cbf_controller = builder.AddSystem(
        SwitchingCBFController(
            x=x,
            f=f,
            g=g,
            cbfs=union_cbfs,
            alpha=alpha,
            control_limits=(A, c),
            switching_policy_id=2,
            switching_threshold=(0.05, 0.15),
            initial_cbf_index=init_cbf_index
        )
    )
    action_logger = LogVectorOutput(switching_cbf_controller.get_output_port(), builder)
    
    builder.Connect(
        nominal_controller.get_output_port(0),
        switching_cbf_controller.get_input_port(1)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        switching_cbf_controller.get_input_port(0)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        switching_cbf_controller.get_output_port(0),
        single_integrator.get_input_port(0)
    )
    diagram = builder.Build()

    return(
        diagram,
        state_logger,
        action_logger,
        waypoints
    )

def build_diagram_bncbf()-> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs waypoints
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = SingleIntegrator2D()
    (f, g) = plant.affine_dynamics(x)
    (A, c) = plant.control_limits(output_poly=False)
    alpha = 1.0
    waypoints = np.array([[4.5, 0.0]])

    # the plant block:
    single_integrator = builder.AddSystem(SingleIntegrator2D())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        NominalController(
            waypoints=waypoints,
            gain=1.0
        )
    )

    # load the union of CBFs:
    filename = "unsafe_region_and_cbfs.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (_, union_cbfs, _, _, _) = load_unsafe_region_and_cbfs(
        x_vars=x,
        pickle_path=data_path
    )
    # the switching CBF controller block:
    bncbf_cbf_controller = builder.AddSystem(
        BncbfController(
            x=x,
            f=f,
            g=g,
            cbfs=union_cbfs,
            alpha=alpha,
            control_limits=(A, c)
        )
    )
    action_logger = LogVectorOutput(
        bncbf_cbf_controller.get_output_port(), builder
        )

    builder.Connect(
        nominal_controller.get_output_port(0),
        bncbf_cbf_controller.get_input_port(1)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        bncbf_cbf_controller.get_input_port(0)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        bncbf_cbf_controller.get_output_port(0),
        single_integrator.get_input_port(0)
    )
    diagram = builder.Build()

    return(
        diagram,
        state_logger,
        action_logger,
        waypoints
    )

def build_diagram_composite_cbf()-> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs waypoints
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = SingleIntegrator2D()
    (f, g) = plant.affine_dynamics(x)
    (A, c) = plant.control_limits(output_poly=False)
    alpha = 1.0
    waypoints = np.array([[4.5, 0.0]])

    # the plant block:
    single_integrator = builder.AddSystem(SingleIntegrator2D())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        NominalController(
            waypoints=waypoints,
            gain=1.0
        )
    )

    # load the union of CBFs:
    filename = "unsafe_region_and_cbfs.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    (_, union_cbfs, _, composite_param, _) = load_unsafe_region_and_cbfs(
        x_vars=x,
        pickle_path=data_path
    )
    # the switching CBF controller block:
    composite_cbf_controller = builder.AddSystem(
        CompositeCbfController(
            x=x,
            f=f,
            g=g,
            cbfs=union_cbfs,
            alpha=alpha,
            control_limits=(A, c),
            composite_cbf_param=composite_param
        )
    )
    action_logger = LogVectorOutput(
        composite_cbf_controller.get_output_port(), builder
        )

    builder.Connect(
        nominal_controller.get_output_port(0),
        composite_cbf_controller.get_input_port(1)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        composite_cbf_controller.get_input_port(0)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        composite_cbf_controller.get_output_port(0),
        single_integrator.get_input_port(0)
    )
    diagram = builder.Build()

    return(
        diagram,
        state_logger,
        action_logger,
        waypoints
    )

def simulate(
    total_states_init: np.ndarray, 
    init_cbf_index: Optional[int],
    case_name: str,
    duration: float,
) -> Tuple[
    np.ndarray, # state data
    np.ndarray, # action data
    np.ndarray, # time data
    np.ndarray, # waypoints
    ]:
    # the state logger and action logger still refers
    # to the plant states and the plant input actions.
    if case_name == "nominal":
        (diagram, 
        state_logger,
        action_logger,
        waypoints
        ) = build_diagram_nominal()
    elif case_name == "switching_policy_I":
        (diagram, 
        state_logger,
        action_logger,
        waypoints
        ) = build_diagram_switching_policy_I(
            init_cbf_index=init_cbf_index
        )
    elif case_name == "switching_policy_II":
        (diagram, 
        state_logger,
        action_logger,
        waypoints,
        ) = build_diagram_switching_policy_II(
            init_cbf_index=init_cbf_index
            )
    elif case_name == "bncbf":
        (diagram, 
        state_logger,
        action_logger,
        waypoints,
        ) = build_diagram_bncbf()
    elif case_name == "composite_cbf":
        (diagram, 
        state_logger,
        action_logger,
        waypoints,
        ) = build_diagram_composite_cbf()
    else:
        raise ValueError(f"Invalid case name: {case_name}")
    
    simulator = pydrake.systems.analysis.Simulator(diagram)
    context = simulator.get_mutable_context()
    # total state refers to all the states in the diagram
    # these not only include the plant states, but also
    # the integrator states in the PID controllers.
    context.SetContinuousState(total_states_init)
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
        waypoints
    )

def run_simulation(
    case_name: str
):
    # initial states:
    x0 = np.array([
        [-4, 1.5],
        [-4, 0],
        [-4, -1.5]
    ])
    # initial_cbf_indecies:
    if case_name == "switching_policy_I" \
        or case_name == "switching_policy_II":
        init_cbf_indices = [0, 0, 3]
    else:
        init_cbf_indices = [None]*x0.shape[0]
    
    state_data_list = []
    action_data_list = []
    time_data = None
    for i in range(x0.shape[0]):
        total_states_init = x0[i, :]
        duration = 100.0
        (state_data, 
        action_data, 
        time_data,
        waypoints
        ) = simulate(
            total_states_init=total_states_init,
            init_cbf_index=init_cbf_indices[i],
            case_name=case_name,
            duration=duration
        )
        state_data_list.append(state_data)
        action_data_list.append(action_data)

    # save the trajectory data:
    filename = "sim_" + case_name + "_data.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    save_simulation_data(
        initial_states=x0,
        waypoints=waypoints,
        state_data_list=state_data_list,
        action_data_list=action_data_list,
        time_data=time_data,
        pickle_path=data_path
    )

def save_simulation_data(
    initial_states: np.ndarray,
    waypoints: np.ndarray,
    state_data_list: List[np.ndarray],
    action_data_list: List[np.ndarray],
    time_data: np.ndarray,
    pickle_path: str
):
    data = {
        "initial_states": initial_states,
        "waypoints": waypoints,
        "state_data_list": state_data_list,
        "action_data_list": action_data_list,
        "time_data": time_data
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


if __name__ == "__main__":
    run_simulation(case_name="switching_policy_I")
    run_simulation(case_name="switching_policy_II")
    run_simulation(case_name="bncbf")
    run_simulation(case_name="composite_cbf")
















