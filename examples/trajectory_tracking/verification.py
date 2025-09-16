"""
This script compaires the verification time for theorem 2 and theorem 3
with different number of CBFs. We use the CBFs used for trajectory tracking
and let the verification algorithm run for the first CBF, then union of the
first two CBFs, and then the union of the first three CBFs and so on.

We finally wanted to plot the verification time against the number of CBFs.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import numpy as np
from typing import Tuple, Optional, List
import pickle

import pydrake.symbolic as sym

import matplotlib.pyplot as plt

from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf import UnionCbf
from plant import SingleIntegratorPlant
from experiment_setup import ExperimentSetup

def verify_union_cbf_thm3(
    setup: ExperimentSetup,
    number_of_cbfs: int
) -> float:
    """
    This function returns the verification time for theorem 3 
    given the number of CBFs.
    """
    assert number_of_cbfs >= 1
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegratorPlant()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    cbfs = setup.get_cbfs(x)
    union_object = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs[:number_of_cbfs],
        alpha=0.5,
        control_limits=(A, c),
    )

    start_time = time.time()
    verfication_thm3 = union_object.verification_of_theorem_3(
        cbf_lagrangian_x_degree=2,
        cbf_lagrangian_y_degree=2,
        lambda_y_lagrangian_x_degree=2,
        lambda_y_lagrangian_y_degree=0,
        xi_y_lagrangian_x_degree=2,
        xi_y_lagrangian_y_degree=0,
        eta=1e-3,
        epsilon=0.01,
    )
    end_time = time.time()
    assert verfication_thm3 == True
    thm3_time = end_time - start_time
    print(f"The verification of therorem 3 for {number_of_cbfs} CBFs is successful!")
    print(f"Time taken: {thm3_time} seconds")

    return thm3_time

def non_empty_subsets_for_traj_tracking(
    x: np.ndarray,
    setup: ExperimentSetup,
    number_of_cbfs: int
) -> List[Subset]:
    """
    Since we already know that all
    the CBFs in the trajectory tracking example only has two
    consecutive overlapping, hence we can directly set the
    activation indices for all the non-empty subsets given
    the number of CBFs.
    """
    # Set the non-empty subset indices:
    one_active_cbf = np.eye(number_of_cbfs, dtype=int)
    two_active_cbf = np.eye(number_of_cbfs, dtype=int)
    for i in range(number_of_cbfs-1):
        two_active_cbf[i, i+1] = 1
    active_index_list = np.vstack((one_active_cbf, two_active_cbf[:-1, :]))
    print(f"The active index list is:\n{active_index_list}")
    assert active_index_list.shape[0] == number_of_cbfs + (number_of_cbfs - 1)
    assert active_index_list.shape[1] == number_of_cbfs

    all_cbfs = setup.get_cbfs(x)
    used_cbfs = all_cbfs[:number_of_cbfs]
    non_empty_subsets = [
        Subset(
            x=x,
            cbfs=used_cbfs,
            activation_index=active_index
        )
        for active_index in active_index_list
    ]

    return non_empty_subsets

def verify_union_cbf_thm2(
    setup: ExperimentSetup,
    number_of_cbfs: int
) -> float:
    """
    This function returns the verification time for theorem 2 
    given the number of CBFs.
    """
    assert number_of_cbfs >= 1
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegratorPlant()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    cbfs = setup.get_cbfs(x)
    union_object = UnionCbf(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs[:number_of_cbfs],
        alpha=0.5,
        control_limits=(A, c),
    )
    non_empty_subsets = non_empty_subsets_for_traj_tracking(
        x=x,
        setup=setup,
        number_of_cbfs=number_of_cbfs
    )
    start_time = time.time()
    for each_subset in non_empty_subsets:
        print(
            f"Checking subset with activation index: \
            {each_subset.activation_index}"
            )
        single_start_time = time.time()
        union_object.check_feasibility_in_subset(
            subset=each_subset,
            cbf_lagrangian_x_degree=2,
            cbf_lagrangian_y_degree=2,
            lambda_y_lagrangian_x_degree=2,
            lambda_y_lagrangian_y_degree=0,
            xi_y_lagrangian_x_degree=2,
            xi_y_lagrangian_y_degree=0,
            eta=1e-3,
            epsilon=0.01
        )
        single_end_time = time.time()
        print(
            f"Time taken for this subset: \
            {single_end_time - single_start_time} seconds"
            )
    end_time = time.time()
    thm2_time = end_time - start_time
    print(f"The verification of therorem 2 for {number_of_cbfs} CBFs is successful!")
    print(f"Time taken: {thm2_time} seconds")    

    return thm2_time

def verify_all(
    save_data: bool
) -> Optional[Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
    ]]:
    setup = ExperimentSetup()
    number_of_all_cbfs = setup.waypoints.shape[0]
    cbf_num_axis = np.arange(1, number_of_all_cbfs+1)
    thm2_times = np.zeros((number_of_all_cbfs,))
    thm3_times = np.zeros((number_of_all_cbfs,))
    for idx in range(number_of_all_cbfs):
        thm2_time = verify_union_cbf_thm2(setup, idx+1)
        thm3_time = verify_union_cbf_thm3(setup, idx+1)
        thm2_times[idx] = thm2_time
        thm3_times[idx] = thm3_time
    if save_data:
        # Save with pickle
        with open('examples/trajectory_tracking/arrays.pkl', 'wb') as f:
            pickle.dump({
                'thm2_times': thm2_times,
                'thm3_times': thm3_times,
                'cbf_num_axis': cbf_num_axis
                }, f)

    return cbf_num_axis, thm2_times, thm3_times

def plot_computation_time_comparison():
    # Load with pickle
    with open('examples/trajectory_tracking/arrays.pkl', 'rb') as f:
        data = pickle.load(f)

    thm2_computation_times = data['thm2_times']
    thm3_computation_times = data['thm3_times']
    cbf_num_axis = data['cbf_num_axis']

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(
        cbf_num_axis,
        thm2_computation_times,
        marker='o',
        linestyle='--',
        color='blue',
        label='Verification Time Theorem 2'
    )
    ax.plot(
        cbf_num_axis,
        thm3_computation_times,
        marker='s',
        linestyle='-',
        color='orange',
        label='Verification Time Theorem 3'
    )
    ax.set_xlabel('Number of CBFs')
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Verification Time vs Number of CBFs')
    ax.legend()


def main():
    plot_computation_time_comparison()
    


if __name__ == "__main__":
    main()