"""
This script compares verification efficiencey between:
1. Direct verification of a high-degree polynomial HOCBF.
2. Verification of a union of linear HOCBFs constructed
   to approximate the coverage of the high-degree polynomial HOCBF.

We consider an square obstacle in a 2D position domain presented by
ranges: x0 in [-1, 1] and x1 in [-1, 1].
The unsafe region is defined by four linear inequalites:
l1(x) <= 0, l1 = x0 - 1
l2(x) <= 0, l2 = -x0 - 1
l3(x) <= 0, l3 = x1 - 1
l4(x) <= 0, l4 = -x1 - 1
The high-degree polynomial HOCBF is constructed as:
h(x) = x_0^n + x_1^n - 2*1^n, where n is an even integer.
while the union of linear HOCBFs are constructed
as:
h1(x) = x0 - 1.05
h2(x) = -x0 - 1.05
h3(x) = x1 - 1.05
h4(x) = -x1 - 1.05
The verification time of both methods are recorded and compared.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from typing_extensions import Self
import time

import pydrake.symbolic as sym
from examples.double_integrator import dynamics
import union_cbf_base.verification_base as verification
from union_cbf_base.union_cbf_II import (
    CbfFeasibilityLagrangianDegrees,
    Degree
)

import multiprocessing
from concurrent.futures import (
    ProcessPoolExecutor, 
    as_completed
)


def verification_of_union_hocbfs_cascaded():
    double_integrator = dynamics.DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    num_control = A.shape[1]

    # Define linear HOCBFs
    hocbfs = [
        sym.Polynomial(x[0] - 1.05),
        sym.Polynomial(-x[0] - 1.05),
        sym.Polynomial(x[1] - 1.05),
        sym.Polynomial(-x[1] - 1.05)
    ]
    relative_degree = [2]
    alphas = [[0.01, 0.1]]

    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [
            [Degree(x=2, y=2, c=0)] * relative_degree[0]
        ],
        lambda_y=[Degree(x=2, y=0, c=0)]*num_control,
        xi_y=Degree(x=2, y=0, c=0),
        state_eq=None,
    )

    start_time = time.time()
    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=np.array(hocbfs),
        static_cbfs=None,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees,
    )
    end_time = time.time()
    assert verification_flag is True, "Verification failed!"
    print(f"Union of linear HOCBFs verification time: {end_time - start_time:.4f} seconds")

# the following functions are used for parallel verification
def create_missions(hocbfs_coefficients: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
    """Create verification missions"""
    assert len(hocbfs_coefficients.shape) == 2
    num_missions = hocbfs_coefficients.shape[0]
    missions = []
    for i in range(num_missions):
        each_hocbf_coefficients = hocbfs_coefficients[i, :]
        # Mission configuration
        config = {
            "mission_id": i,
            "HOCBF_name": f"HOCBF_{i}",
            "verification_type": "feasibility",
            "timeout": 30
        }
        missions.append((each_hocbf_coefficients, config))
    return missions

def single_verification_mission(
    hocbf_coefficients: np.ndarray,
    config: dict
) -> Tuple[int, dict]:
    assert len(hocbf_coefficients.shape) == 1
    assert hocbf_coefficients.shape[0] == 3 # the coefficent should be for x[0], x[1], bias
    
    mission_id = config.get("mission_id", 0)
    
    double_integrator = dynamics.DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    num_control = A.shape[1]

    # Define linear HOCBFs
    hocbf = sym.Polynomial(
        hocbf_coefficients.dot(np.array([x[0]**1, x[1]**1, 1.0]))
        )
    relative_degree = [2]
    alphas = [[0.01, 0.1]]

    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [
            [Degree(x=2, y=2, c=0)] * relative_degree[0]
        ],
        lambda_y=[Degree(x=2, y=0, c=0)]*num_control,
        xi_y=Degree(x=2, y=0, c=0),
        state_eq=None,
    )
    verification_flag = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=np.array([hocbf]),
        static_cbfs=None,
        relative_degree=relative_degree,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees,
    )
    return mission_id, {
        "is_feasible": verification_flag,
        "mission_id": mission_id,
        "config": config   
    }

def run_parallel_verification(
    missions: List[Tuple[np.ndarray, dict]],
    max_workers: int = None,
    timeout: int = 30
) -> dict:
    """
    Run multiple verification missions in parallel.
    
    Args:
        missions: List of (verified_mat, config_dict) tuples
        max_workers: Number of parallel processes (default: CPU count)
        timeout: Maximum time per mission in seconds
    
    Returns:
        Dictionary of results keyed by mission_id
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    results = {}
    
    # IMPORTANT: Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all missions
        future_to_mission = {}
        for hocbf_coefficients, config in missions:
            future = executor.submit(single_verification_mission, hocbf_coefficients, config)
            future_to_mission[future] = config.get("mission_id", len(future_to_mission))
        
        # Collect results as they complete
        completed = 0
        total = len(missions)
        
        for future in as_completed(future_to_mission, timeout=timeout):
            try:
                mission_id, result = future.result()
                results[mission_id] = result
                completed += 1
                print(f"✓ Mission {mission_id} completed ({completed}/{total})")
                
            except Exception as e:
                mission_id = future_to_mission[future]
                print(f"✗ Mission {mission_id} failed: {e}")
                results[mission_id] = {"error": str(e), "mission_id": mission_id}
    
    return results

def verification_of_union_hocbfs_parallel():
    # hocbfs coefficients for multiple missions
    hocbfs_coefficients = np.array([
        [1.0, 0.0, -1.05],
        [-1.0, 0.0, -1.05],
        [0.0, 1.0, -1.05],
        [0.0, -1.0, -1.05],
    ])
    missions = create_missions(hocbfs_coefficients)
    start_time = time.time()
    results = run_parallel_verification(missions)
    end_time = time.time()
    # show the final results
    print(f"\n{'='*60}")
    print(f"Verification completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(results)} missions")
    
    print("\nSummary of results:")
    for mission_id, result in sorted(results.items()):
        if "error" in result:
            print(f"  Mission {mission_id}: ERROR - {result['error']}")
        else:
            status = "FEASIBLE" if result["is_feasible"] else "INFEASIBLE"
            print(f"  Mission {mission_id}: {status}")

def main():
    print("\nVerifying union of linear HOCBFs...")
    verification_of_union_hocbfs_cascaded()
    print("\nVerifying union of linear HOCBFs in parallel...")
    verification_of_union_hocbfs_parallel()

if __name__ == "__main__":
    main()