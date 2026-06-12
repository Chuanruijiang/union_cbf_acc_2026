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
from examples.single_integrator_2D_const_vel.dynamics import (
    SingleIntegrator,
    NominalController
)

def build_diagram_nominal() -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    float, # output_ref_hight
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = SingleIntegrator()
    (f, g) = plant.affine_dynamics(x)
    (A, c) = plant.control_limits(output_poly=False)
    reference_hight = 1.5
    
    # the plant block:
    single_integrator = builder.AddSystem(SingleIntegrator())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        NominalController(
            target_height=reference_hight,
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
        reference_hight
    )

def build_diagram_switching_I() -> Tuple[
    Diagram, # outputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    float, # output_ref_hight
]:
    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = SingleIntegrator()
    (f, g) = plant.affine_dynamics(x)
    (A, c) = plant.control_limits(output_poly=False)
    reference_hight = 1.5
    cbfs = np.array([
        sym.Polynomial(8 - (x[0] + 2)**2 - (x[1] + 2)**2),
        sym.Polynomial(x[0] - x[1])
    ])
    alpha = 1.0
    
    # the plant block:
    single_integrator = builder.AddSystem(SingleIntegrator())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        NominalController(
            target_height=reference_hight,
            gain=2.0
        )
    )

    # the switching CBF controller block:
    switching_controller = builder.AddSystem(
        SwitchingCBFController(
            x=x,
            f=f,
            g=g,
            normal_cbfs=None,
            switching_cbfs=cbfs,
            relative_degree=[1],
            alpha=[[alpha]],
            control_limits=(A, c),
            switching_policy_id=1,
            switching_policy_param=(0.3, 0.4),
        )
    )
    action_logger = LogVectorOutput(switching_controller.get_output_port(), builder)

    builder.Connect(
        nominal_controller.get_output_port(0),
        switching_controller.get_input_port(1)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        switching_controller.get_input_port(0)
    )
    builder.Connect(
        switching_controller.get_output_port(0),
        single_integrator.get_input_port(0)
    )

    diagram = builder.Build()

    return(
        diagram,
        state_logger,
        action_logger,
        reference_hight
    )

def simulate(
    total_states_init: np.ndarray,
    duration: float,
) -> Tuple[
    np.ndarray, # state data
    np.ndarray, # action data
    np.ndarray, # time data
    float, # reference height
    ]:
    # the state logger and action logger still refers
    # to the plant states and the plant input actions.
    (diagram, 
    state_logger,
    action_logger,
    reference_hight
    ) = build_diagram_switching_I()
    
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
        reference_hight
    )

def run_simulation():
    # initial states:
    x0 = np.array([
        [-2, -1],
        [-2, -1.5],
        [-2, -2]
    ])
    # initial_cbf_indecies:
    state_data_list = []
    action_data_list = []
    time_data = None
    for i in range(x0.shape[0]):
        total_states_init = x0[i, :]
        duration = 50.0
        (state_data, 
        action_data, 
        time_data,
        reference_hight
        ) = simulate(
            total_states_init=total_states_init,
            duration=duration
        )
        state_data_list.append(state_data)
        action_data_list.append(action_data)

    # save the trajectory data:
    filename = "sim_const_vel_data.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    save_simulation_data(
        initial_states=x0,
        reference_hight=reference_hight,
        state_data_list=state_data_list,
        action_data_list=action_data_list,
        time_data=time_data,
        pickle_path=data_path
    )

def save_simulation_data(
    initial_states: np.ndarray,
    reference_hight: float,
    state_data_list: List[np.ndarray],
    action_data_list: List[np.ndarray],
    time_data: np.ndarray,
    pickle_path: str
):
    data = {
        "initial_states": initial_states,
        "reference_hight": reference_hight,
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
    run_simulation()
