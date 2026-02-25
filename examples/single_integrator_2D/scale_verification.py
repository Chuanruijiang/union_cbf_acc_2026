# """
# This script conduct the verification comparison for the union of
# 11 circular CBFs defined in scale_cbfs.py. The comparison is 
# conducted following these steps:
# 1. We create the setup Class defined in scale_cbfs.py,
#     and get the CBFs as array of sym.Polynomials
# 2. We use the verif-I and verif-II methods to verify these CBFs
#     respectively. And record the verification time as the number
#     of CBF increases from 1 to 11.
# 3. We compare the results of these two methods, and plot the
#     verification time against the number of CBFs.
# """

import os
import pickle

import pydrake.symbolic as sym
import matplotlib.pyplot as plt

from union_cbf_base.union_cbf_I import (
    UnionCbfI,
    SubsetGeneralLagrangianDegrees,
)
from union_cbf_base.union_cbf_II import (
    UnionCbfII,
    CbfFeasibilityLagrangianDegrees,
    Degree as DegreeII,
)
from examples.single_integrator_2D.dynamics import SingleIntegrator2D
from examples.single_integrator_2D.scale_cbfs import ExperimentSetup

def verification_with_verif_I(num_cbf: int) -> float:
    """
    This function returns the verification time for verif-I method.
    """
    assert num_cbf >= 1
    setup = ExperimentSetup()
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator2D()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    cbfs = setup.get_cbfs(x)[:num_cbf]
    alpha = 1.0

    lagrangian_degrees = SubsetGeneralLagrangianDegrees(
        num_control_inputs=2,
        cbfs_lagrangian_x_degree=2,
        cbfs_lagrangian_y_degree=2,
        lambda_lagrangian_x_degree=2,
        lambda_lagrangian_y_degree=0,
        xi_lagrangian_x_degree=2,
        xi_lagrangian_y_degree=0,
    )
    union_object = UnionCbfI(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        control_limits=(A, c),
    )
    (verification_flag, 
     verif_time
    ) = union_object.verification_feasibility_condition_I(
        union_cbfs=cbfs,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_verification_time=True,
    )
    assert verification_flag == True
    return verif_time

def verification_with_verif_II(num_cbf: int) -> float:
    """
    This function returns the verification time for verif-II method.
    """
    assert num_cbf >= 1
    setup = ExperimentSetup()
    x = sym.MakeVectorContinuousVariable(2, "x")
    single_integrator = SingleIntegrator2D()
    f, g = single_integrator.affine_dynamics(x)
    A, c = single_integrator.control_limits()
    num_controls = A.shape[1]
    cbfs = setup.get_cbfs(x)[:num_cbf]
    alpha = 1.0

    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        cbf = DegreeII(x=2, y=2, c=0),
        lambda_y=[DegreeII(x=2, y=0, c=0)]*num_controls,
        xi_y=DegreeII(x=2, y=0, c=0)
    )
    union_object = UnionCbfII(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        control_limits=(A, c),
    )
    (verification_flag, 
     verif_time
    ) = union_object.verification_feasibility_condition_II(
        union_cbfs=cbfs,
        lagrangian_degrees=lagrangian_degrees,
        eta=1e-4,
        eps=1e-4,
        show_output=True,
        show_verification_time=True,
    )
    assert verification_flag == True
    return verif_time

def comparison(save_path: str):
    num_cbfs_list = list(range(1, 12))
    verif_I_times = []
    verif_II_times = []
    for num_cbfs in num_cbfs_list:
        print("====================================")
        print(f"Verifying with {num_cbfs} CBFs...")
        print("------------Verif-I:--------------")
        verif_I_time = verification_with_verif_I(num_cbfs)
        print("------------Verif-II:--------------")
        verif_II_time = verification_with_verif_II(num_cbfs)
        verif_I_times.append(verif_I_time)
        verif_II_times.append(verif_II_time)
    
    # save the recorded verification times:
    data = {
        'num_cbfs_list': num_cbfs_list,
        'verif_I_times': verif_I_times,
        'verif_II_times': verif_II_times,
    }
    if os.path.exists(save_path):
        overwrite_cmd = input(
            f"File {save_path} already exists. Overwrite the file? Press [Y/n]:"
        )
        if overwrite_cmd in ("Y", "y"):
            save_cmd = True
        else:
            save_cmd = False
    else:
        save_cmd = True

    if save_cmd:
        with open(save_path, "wb") as handle:
            pickle.dump(data, handle)
    
def plot_comparison(data_path: str):
    # load the recorded verification times:
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    num_cbfs_list = data['num_cbfs_list']
    verif_I_times = data['verif_I_times']
    verif_II_times = data['verif_II_times']
    # plot the comparison results:
    plt.figure()
    plt.plot(num_cbfs_list, verif_I_times, label='Verif-I')
    plt.plot(num_cbfs_list, verif_II_times, label='Verif-II')
    plt.xlabel('Number of CBFs')
    plt.ylabel('Verification Time (s)')
    plt.title('Verification Time vs Number of CBFs')
    plt.legend()

if __name__ == "__main__":
    filename = "verification_time_comparison_data.pkl"
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    comparison(save_path=file_path)
    # plot_comparison(data_path=file_path)