"""
This script verifies a union of CBFs for double integrator system in 
multi-threading way. Since the Pydrake symbolic variables class does not 
support multi-threading, then the multi-thread is implemented at the whole
verification sos program level. Meaning that, for each CBF in the union, 
we create a separate process of verification. Each process creates its own
symbolic variables, its own system dynamics, and its own verification sos program.
"""
import time
import numpy as np
from typing import List, Optional, Tuple, Union
import pydrake.symbolic as sym

from examples.double_integrator import dynamics
from union_cbf_base.union_cbf_II import (
    CbfFeasibilityLagrangianDegrees,
    Degree
)
import union_cbf_base.verification_base as verification

import multiprocessing
from concurrent.futures import ( 
    ProcessPoolExecutor, 
    as_completed
)

"""
The following functions are used for multi-thread verification of union CBFs
using the union formulation II. Assume we have a union of N CBFs, and this 
union of CBF is intersected with the other M-1 number of CBFs. According
to the union formulation II, we need to verify the following intersections:
 [h_1, h_2, ..., h_{M-1}, h_{M1}],
 [h_1, h_2, ..., h_{M-1}, h_{M2}],
 ... 
 [h_1, h_2, ..., h_{M-1}, h_{MN}],
and the whole verification can be parallelized into N processes, each
process verifies one of the above intersections.
Noted that sym.variables can not be shared across multiple threads, thus
each process creates its own symbolic variables and its own verification
sos program. Hence, all the CBFs in the union (h_{M1} to h_{MN}) above 
should be specified as groups of coefficients of polynomials, instead of 
symbolic polynomials. However, since the other normal CBFs (h_1 to h_{M-1})
are shared across multiple processes, they can be specified as symbolic
polynomials in single process, rather than being specified as coefficients
at the mission creation phase.
"""

def create_missions(
        union_cbfs_coefficients: np.ndarray
    ) -> List[Tuple[np.ndarray, dict]]:
    """
    create the missions for multi-thread verification
    Each mission corresponds to the verification of one CBF in the union
    intersected with the other normal CBFs. We assume all the CBFs in the
    union shares the same set of nonomial basis.
    Args:
        union_cbfs_coefficients: np.ndarray
        Shape should be (num_cbfs_in_union, num_monomial_basis)
    Returns:
        List of missions, each mission is a tuple of
        (cbf_coefficients: np.ndarray, config: dict)
    """
    num_missions = union_cbfs_coefficients.shape[0]
    missions = []
    for i in range(num_missions):
        each_polynomial_coeffs = union_cbfs_coefficients[i]
        # Mission configuration
        config = {
            "mission_id": i,
            "cbf_name": f"CBF_{i}",
            "verification_type": "feasibility",
            "timeout": 30
        }
        missions.append((each_polynomial_coeffs, config))
    return missions

def single_verification_mission(
    cbf_coeffs: np.ndarray,
    config: dict
) -> Tuple[int, dict]:
    """
    This function verifies the feasibility of the CBF-QP in an intersection
    of CBFs, the normal CBFs (h_1 to h_{M-1}) are hard-coded as sym.Polynomials
    in this function, meanwhile the CBF from the union (h_{Mi}) is specified by
    the input coefficients. The monomial basis should be specified in advance
    and also hard-coded in this function.
    """
    mission_id = config.get("mission_id", 0)
    double_integrator = dynamics.DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    num_control = A.shape[1]
    
    static_cbfs = np.array([
        sym.Polynomial(-x[1] + 2.0),
        sym.Polynomial(x[0] + 2.0)
    ])
    monomial_basis = np.array([x[0], x[1], 1.0])
    cbf_from_union = sym.Polynomial(cbf_coeffs.dot(monomial_basis))
    relative_degrees = [2, 2, 2]
    alphas = [
        [0.01, 0.1],
        [0.01, 0.1],
        [0.01, 0.1]
    ]
    lagrangian_degrees = CbfFeasibilityLagrangianDegrees(
        phis = [[Degree(x=2, y=2, c=0)
            for _ in range(relative_degrees[i])]
            for i in range(len(relative_degrees))
        ],
        lambda_y=[Degree(x=2, y=0, c=0)
            for _ in range(num_control)],
        xi_y=Degree(x=2, y=0, c=0),
        state_eq=None,
    )
    is_feasible = verification.verify_sufficient_condition_II(
        x=x,
        f=f,
        g=g,
        control_limits=(A, c),
        switching_cbfs=np.array([cbf_from_union]),
        static_cbfs=static_cbfs,
        relative_degree=relative_degrees,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees
    )
    return mission_id, {
        "feasibility": is_feasible,
        "mission_id": mission_id,
        "config": config
        }

def run_parallel_verification(
    missions: List[Tuple[np.ndarray, dict]],
    max_workers: int = None,
    timeout: int = None
) -> dict:
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    results = {}
    
    # IMPORTANT: Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all missions
        future_to_mission = {}
        for cbf_coeffs, config in missions:
            future = executor.submit(single_verification_mission, cbf_coeffs, config)
            future_to_mission[future] = config.get("mission_id", len(future_to_mission))
        
        # Collect results as they complete
        completed = 0
        total = len(missions)
        
        for future in as_completed(future_to_mission, timeout=timeout):
            try:
                mission_id, result = future.result()
                results[mission_id] = result
                completed += 1
                print(f"Mission {mission_id} completed ({completed}/{total})")
                
            except Exception as e:
                mission_id = future_to_mission[future]
                print(f"Mission {mission_id} failed: {e}")
                results[mission_id] = {"error": str(e), "mission_id": mission_id}
    
    return results

def main():
    """
    The multi-thread verification verifies the following feasibility problems:
    [h_1 = -x1 + 2, h_2 = x0 + 2, h_31 = -x0],
    [h_1 = -x1 + 2, h_2 = x0 + 2, h_32 = x1 ],
    where h_31 and h_32 are two CBFs in the union, and h_1 and h_2 are normal CBFs.
    """
    # hocbfs coefficients for multiple missions
    x = sym.MakeVectorContinuousVariable(2, "x")
    switching_cbf_coeffs = np.array([
        [-1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0]
    ])
    missions = create_missions(union_cbfs_coefficients=switching_cbf_coeffs)
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
            status = "Feasible" if result["feasibility"] else "Not Feasible"
            print(f"  Mission {mission_id}: {status}")

if __name__ == "__main__":
    main()