import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
from typing import Optional, Tuple, List, Union

import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt

import pydrake.symbolic as sym
import pydrake.systems.analysis
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, VectorLogSink

from plant import SingleIntegratorPlant
from union_cbf_base.controller import SwitchingCBFController
from experiment_setup import ExperimentSetup

def build_diagram()->Tuple[
    Diagram,
    SingleIntegratorPlant,
    SwitchingCBFController,
    VectorLogSink,
    VectorLogSink
]:
    builder = DiagramBuilder()
    
    # create the plant
    single_integrator = builder.AddSystem(SingleIntegratorPlant())
    state_logger = LogVectorOutput(single_integrator.get_output_port(), builder)
    plant_obj = SingleIntegratorPlant()
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, g = plant_obj.affine_dynamics(x)

    # create the environement setup obejct:
    env_setup = ExperimentSetup(x=x)

    # create the controller
    # noted that we should load the evironment setup when creating the controller
    switching_controller = builder.AddSystem(SwitchingCBFController(
        x=x,
        f=f,
        g=g,
        cbfs=env_setup.cbfs,
        alpha=0.5,
        control_limits=plant_obj.control_limits(),
        waypoints=env_setup.waypoints,
        switching_policy_id=1,
        solver_id=None,
        solver_options=None
    ))

    # with the blockes ready, now lets connect them:
    builder.Connect(
        switching_controller.action_output_port(),
        single_integrator.get_input_port(0)
    )
    builder.Connect(
        single_integrator.get_output_port(0),
        switching_controller.get_input_port(0)
    )
    action_logger = LogVectorOutput(switching_controller.action_output_port(), builder)
    diagram = builder.Build()

    return(
        diagram,
        single_integrator,
        switching_controller,
        state_logger,
        action_logger
    )

def simulate(x0: np.ndarray, duration: float):
    # initialize block diagram:
    (
        diagram,
        single_integrator,
        switching_controller,
        state_logger,
        action_logger
    ) = build_diagram()

    # set the initial state
    simulator = pydrake.systems.analysis.Simulator(diagram)
    simulator.get_mutable_context().SetContinuousState(x0)

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
    return state_data, action_data, time_data

def run_simulation():
    # initial condition:
    x0 = np.array([1, 7])
    # duration = 90.0 # duration for switching policy 1
    duration = 260.0 # duration for switching policy 2
    
    fig = plt.figure()
    ax = fig.add_subplot()

    (state_data, action_data, time_data) = simulate(x0=x0, duration=duration)
    ax.plot(state_data[0, :], state_data[1, :], label="trajectory", color="blue")
    
    
def main():
    print(os.path.dirname(__file__))
    run_simulation()

if __name__ == "__main__":
    main()




