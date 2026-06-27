"""
This simulation shows that a double integrator system with
constant x-axis velocity can keep safe by switching policy
I. 
"""
import os
import pickle
import numpy as np
from typing import Optional, Tuple

import pydrake.symbolic as sym
import pydrake.systems.analysis
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.systems.primitives import (
    LogVectorOutput, VectorLogSink, ConstantVectorSource
)
import union_cbf_base.utils as utils
from examples.double_integrator_const_vel.dynamics import (
    DoubleIntegratorPlantConstantVel,
    LQRNominalController1D
)
from union_cbf_base.controller import SwitchingCBFController


def build_diagram() -> Tuple[
    Diagram, # ourputs the whole diagram
    VectorLogSink, # outputs the state logger
    VectorLogSink, # outputs the action logger
    np.ndarray, # outputs the state variable x
    np.ndarray, # outputs the static cbfs
    np.ndarray # outputs the switching cbfs
]:
    # we don't need to set the waypoints for this double integrator
    # since we always has a constant velocity in x-axis direction to
    # drive our system forward. The safe control is just to avoid
    # obstacles.

    builder = DiagramBuilder()
    x = sym.MakeVectorContinuousVariable(3, "x")
    poly_plant = DoubleIntegratorPlantConstantVel(vx_const=1.0)
    f, g = poly_plant.affine_dynamics(x)
    (A, c) = poly_plant.control_input_limits(output_poly=False)
    # specify the static CBFs and switching CBFs:
    static_cbfs = np.array([
        sym.Polynomial(25 - x[1]),
    ])
    switching_cbfs = np.array([
        sym.Polynomial(x[0] + x[1] - 10),
        sym.Polynomial(-x[0] + x[1] - 10),
    ])  
    relative_degree = [2, 2]
    alphas = [[0.1, 1.0], [0.1, 1.0]]
    
    # the plant block:
    double_integrator = builder.AddSystem(
        DoubleIntegratorPlantConstantVel(vx_const=1.0)
        )
    state_logger = LogVectorOutput(double_integrator.get_output_port(), builder)
    
    # the switching CBF-QP controller block:
    # the swtihcong CBF-QP should have two input ports:
    # the first input port is for the feedback state,
    # the second input port is for the nominal control action.
    switching_cbf_qp_controller = builder.AddSystem(
        SwitchingCBFController(
            x=x,
            f=f,
            g=g,
            static_cbfs=static_cbfs,
            switching_cbfs=switching_cbfs,
            relative_degree=relative_degree,
            alpha = alphas,
            control_limits=(A, c),
            switching_policy_id=1,
            switching_policy_param=(1e-2, 2e-2),
            solver_id=None,
            solver_options=None
        )
    )
    action_logger = LogVectorOutput(switching_cbf_qp_controller.action_output_port(), builder)

    # the nominal controller block:
    nominal_controller = builder.AddSystem(
        LQRNominalController1D(
            Q = np.array([
                [1, 0],
                [0, 10]
            ]),
            R= np.array([[1]])
        )
    )
    
    # connect the blocks:
    builder.Connect(
        double_integrator.get_output_port(0),
        switching_cbf_qp_controller.get_input_port(0)
    )
    builder.Connect(
        double_integrator.get_output_port(0),
        nominal_controller.get_input_port(0)
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
        x,
        static_cbfs,
        switching_cbfs
    )

def simulate(
    total_states: np.ndarray, 
    duration: float
) -> Tuple[
    np.ndarray, # state data
    np.ndarray, # action data
    np.ndarray, # time data
    np.ndarray, # plant states variables
    np.ndarray,  # static cbfs
    np.ndarray   # switching cbfs
    ]:
    # The reason why we have "total states" here is because the
    # controller may also have internal states, hence the cloesd
    # loop system states is concatenation of plant states and 
    # controller states. 
    # the state logger and action logger still refers to the total
    # states output and the plant input actions.
    (diagram, 
     state_logger,
     action_logger,
     plant_states_variables,
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
        plant_states_variables,
        static_cbfs,
        switching_cbfs
    )

def save_data(
    pickle_path: str,
    state_data: np.ndarray,
    action_data: np.ndarray,
    time_data: np.ndarray,
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
        "time_data": time_data
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
    x0 = np.array([-30, -6, 0.0])
    total_states = x0
    duration = 70.0
    (state_data, 
     action_data, 
     time_data,
     plant_states_vars,
     static_cbfs,
     switching_cbfs
    ) = simulate(total_states, duration)
    
    filename = "double_integrator_const_vel_simulation_data.pkl"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    save_data(
        pickle_path=data_path,
        state_data=state_data,
        action_data=action_data,
        time_data=time_data,
        static_cbfs=static_cbfs,
        switching_cbfs=switching_cbfs,
        x_vars=plant_states_vars
    )


if __name__ == "__main__":
    run_simulation()

