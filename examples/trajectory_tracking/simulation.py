import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
from typing import Optional, Tuple, List, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pydrake.symbolic as sym
import pydrake.systems.analysis
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, VectorLogSink

from plant import SingleIntegratorPlant
from union_cbf_base.controller import SwitchingCBFController
from experiment_setup import ExperimentSetup

def build_diagram(
    environment_setup: ExperimentSetup,
    switching_policy_idx: int
)->Tuple[
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

    # create the controller
    # noted that we should load the evironment setup when creating the controller
    switching_controller = builder.AddSystem(SwitchingCBFController(
        x=x,
        f=f,
        g=g,
        cbfs=environment_setup.get_cbfs(x),
        alpha=0.5,
        control_limits=plant_obj.control_limits(),
        waypoints=environment_setup.waypoints,
        switching_policy_id=switching_policy_idx,
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

def simulate(
        experiment_setup: ExperimentSetup,
        switching_policy_idx: int,
        x0: np.ndarray,
        duration: float
):
    # initialize block diagram:
    (
        diagram,
        single_integrator,
        switching_controller,
        state_logger,
        action_logger
    ) = build_diagram(
        environment_setup=experiment_setup,
        switching_policy_idx=switching_policy_idx
        )

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
    x0 = np.array([3, 2])
    duration_1 = 100.0 # duration for switching policy 1
    duration_2 = 300.0 # duration for switching policy 2

    fig = plt.figure()
    ax = fig.add_subplot()
    env_setup = ExperimentSetup()
    env_setup.plot_setup(ax=ax)
    
    # we first run the simulation with switching policy 1:
    (
        state_data_1,
        action_data_1,
        time_data_1
    ) = simulate(
        experiment_setup=env_setup,
        switching_policy_idx=1,
        x0=x0,
        duration=duration_1
        )
    # then we run the simulation with switching policy 2:
    (
        state_data_2,
        action_data_2,
        time_data_2
    ) = simulate(
        experiment_setup=env_setup,
        switching_policy_idx=2,
        x0=x0,
        duration=duration_2
        )
    
    # trajectory plot:
    ax.plot(
        state_data_1[0, :],
        state_data_1[1, :],
        linestyle="--",
        color="blue",
        linewidth=3,
        )
    ax.plot(
        state_data_2[0, :],
        state_data_2[1, :],
        linestyle="-",
        color="orange",
        linewidth=3.5,
        alpha=0.8
        )

    ax.set_title("Trajectory Tracking with Different Switching Policies")
    ax.set_xlabel("x1", fontsize=18)
    ax.set_ylabel("x2", fontsize=18)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 33)
    legend_elements = [
        plt.Line2D([0], [0],
            color='blue', lw=3, linestyle="--", label='Switching Policy 1'
            ),
        plt.Line2D([0], [0],
            color='orange', lw=3.5, linestyle="-", label='Switching Policy 2'
            ),
        patches.Patch(facecolor='black', edgecolor='black', alpha=0.5, label='Obstacles'),
        patches.Patch(facecolor='green', alpha=0.3, label='CBF regions')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=12
        )
    
    
    
def main():
    run_simulation()

if __name__ == "__main__":
    main()




