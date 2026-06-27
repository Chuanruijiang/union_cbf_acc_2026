"""
This script defines the synthesis method for union of linear HOCBFs in the 2D 
positional domain with multi-threading support. We decoupled the synthesis of
each HOCBF with its feasibility requirements, meaning that the synthesis part
only gives some sampled parameters for each HOCBF, while the feasibility will
be verified in a separate verification process. All the sampled parameters that
does not satisfy the verification will be discarded, and we only keep the ones
that are verified to be feasible.
"""
import time
import numpy as np
from typing import List, Optional, Tuple, Union
import pydrake.symbolic as sym

import multiprocessing
from concurrent.futures import ( 
    ProcessPoolExecutor, 
    as_completed
)

from examples.double_integrator import dynamics
import union_cbf_base.verification_base as verification
from union_cbf_base.union_cbf_II import (
    CbfFeasibilityLagrangianDegrees,
    Degree
)
"""
The overall process of synthesis:
1. sample normal vector for linear HOCBF:
    since each HOCBF is just defined in the 2D positional domain (x_0, x_1) and 
    it is linear, then each h(x) should just be h(x)=a*x_0 + b*x_1 + c. 
    In order to shrink the parameter sample space, we fix the length of the normal
    vector of the hyperplane to be 1, meaning that we have the following constraint
    on the parameters: a^2 + b^2 = 1. Thus, we can just sample a and b by sampling
    an angle theta in a range [anchor_theta + 0, anchor_theta + pi/2), where 
    anchor_theta is a given anchor angle for the current HOCBF. We sample theta and 
    compute a and b by a=cos(theta), b=sin(theta).
    Note that we should always modulo theta to be within [-pi, pi)], and anchor 
    theta should also be within [-pi, pi].
2. computing c for each sampled (a, b):
    If we are given some points from the unsafe region or and polynomial expression
    of the unsafe region, then we can search c such that the hyperplane 
    h(x)=a*x_0 + b*x_1 + c separates the safe region and the unsafe region and also
    the boundary of the safe region is as close as possible to the unsafe region.
    This can be done by a simple line search of c within some specified number of
    iterations.
3. verify the feasibility of each sampled HOCBF:
    After step 1 and 2, we now get a set of sampled HOCBFs. Now, we use multi-
    threading to verify the feasibility of each sampled HOCBF. If we are searching
    for the first HOCBF in the union, then we only need to verify the feasibility. 
    But if we are searching for subsequent HOCBFs in the union, we also need to make
    sure that the forward invariant set of the new HOCBF has non-empty intersection
    with the forward invariant set of the former HOCBF. This can also be done in the
    by verifying that an subset with only activated polynomials is non-empty, which
    is already implemented in the "non_empty_subset.py"
4. After getting the feasible HOCBFs, we pick the one with normal vector theta that
    closest to the anchor angle.
"""

def sample_normal_vector(
    anchor_theta: float,
    num_samples: int = 10,
    theta_range: float = np.pi/2
) -> np.ndarray:

    thetas = np.linspace(
        anchor_theta,
        anchor_theta + theta_range,
        num_samples
    )
    normal_vectors = np.zeros((num_samples, 2))
    normal_vectors[:, 0] = np.cos(thetas)
    normal_vectors[:, 1] = np.sin(thetas)
    normal_vectors[np.abs(normal_vectors) < 1e-10] = 0.0

    return normal_vectors


def line_search_2D_hyperplane_bias(
    positional_states: np.ndarray,
    normal_vector: np.ndarray,
    unsafe_polys: Optional[np.ndarray],
    unsafe_points: Optional[np.ndarray],
    unsafe_poly_lagrangian_x_degrees: Optional[List[int]],
    cbf_lagrangian_x_degree: Optional[int],
) -> Tuple[sym.Polynomial, float]:
    """
    This function performs a line search to find the maximum bias
    term for the linear HOCBF defined by the given normal vector.
    Since we are focusing on linear 2D HOCBFs in the position domain,
    then the HOCBF should be in the form of:
        h(x) = a*x[0] + b*x[1] + c
    where the x[0] and x[1] are the position in the 2D space.
    This function will search for suitable c (we does not change
    the a, b in this function) such that the region {x|h(x) >= 0} excludes
    all the unsafe points, or the unsafe regions presented by polynomials.
    The input normal_vector is a candidate (a, b) pair,
    and we will perform line search c for this candidate. The return should
    be a sym.Polynomial representing the HOCBF with the found bias c. 
    """
    assert normal_vector.shape[0] == 2
    a = normal_vector[0]
    b = normal_vector[1]
    c_lower = -10.0
    c_upper = 10.0
    tolerance = 1e-2
    max_iterations = 20
    best_c = None

    for _ in range(max_iterations):
        mid_c = (c_lower + c_upper) / 2.0
        h_poly = sym.Polynomial(
            a*positional_states[0] + b*positional_states[1] + mid_c
        )
        cbf_valid = verification.verify_switching_cbfs_validity(
            x=positional_states,
            switching_cbfs=np.array([h_poly]),
            unsafe_polys=unsafe_polys,
            unsafe_points=unsafe_points,
            unsafe_poly_lagrangian_x_degrees=unsafe_poly_lagrangian_x_degrees,
            cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
        )
        if cbf_valid:
            best_c = mid_c
            c_lower = mid_c
        else:
            c_upper = mid_c
        if c_upper - c_lower < tolerance:
            break
    if best_c is not None:
        final_cbf = sym.Polynomial(
            a*positional_states[0] + b*positional_states[1] + best_c
        )
        return final_cbf, best_c
    else:
        return None, None


def check_union_covered_points(
    intrest_points: np.ndarray,
    hocbf_coefficients_list: List[List[float]]
) -> bool:
    """
    This function checks whether the given union of HOCBFs covers
    all the given intrest points.
    """
    for i in range(intrest_points.shape[0]):
        point = intrest_points[i, :]
        covered = False
        for hocbf_coeffs in hocbf_coefficients_list:
            a = hocbf_coeffs[0]
            b = hocbf_coeffs[1]
            c = hocbf_coeffs[2]
            h_value = a*point[0] + b*point[1] + c
            if h_value >= 0:
                covered = True
                break
        if not covered:
            return False
    return True


def create_missions(normal_vectors: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
    """Create verification missions for each sampled normal vector"""
    num_missions = normal_vectors.shape[0]
    missions = []
    for i in range(num_missions):
        normal_vector = normal_vectors[i, :]
        config = {
            "mission_id": i,
            "system_name": "DoubleIntegrator2D",
            "mission_type": "feasibility_check",
            "timeout": 30
        }
        missions.append((normal_vector, config))
    return missions


def single_mission(
    normal_vector: np.ndarray,
    config: dict,
    first_hocbf: bool = True,
    former_hocbf_coefficients: Optional[np.ndarray] = None
) -> Tuple[int, dict]:
    """
    This function performs a single verification mission for the given
    normal vector. It first computes the bias term for the linear HOCBF
    defined by the normal vector, then verifies the feasibility of the
    resulting HOCBF. If the currently searched HOCBF is not the first one
    in the union, it also verifies the non-empty intersection with the
    former HOCBF.
    Note that the following messages of the verification is sepcified
    in this function:
    - system dynamics,
    - control input limits,
    - normal CBFs (if any),
    - unsafe region points or polynomials (if any),
    - relative degrees, alphas, lagrangian degrees.
    """
    mission_id = config.get("mission_id", 0)
    double_integrator = dynamics.DoubleIntegratorPlant2D()
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = double_integrator.affine_dynamics(x)
    (A, c) = double_integrator.control_input_limits()
    num_control = A.shape[1]
    positional_states = x[0:2]
    
    # unsafe points sampled from the unsafe region
    # to test the verification method, we generate 100 
    # random points inside the region:
    # x_0 ∈ [-2, -1], x_1 ∈ [-1, 1]
    unsafe_points = np.random.uniform(
        low=[-2, -1],
        high=[-1, 1],
        size=(1000, 2)
        )
    # Step 1: Line search for bias term
    (hocbf_poly, hocbf_bias) = line_search_2D_hyperplane_bias(
        positional_states=positional_states,
        normal_vector=normal_vector,
        unsafe_polys=None,
        unsafe_points=unsafe_points,
        unsafe_poly_lagrangian_x_degrees=None,
        cbf_lagrangian_x_degree=None,
    )
    if hocbf_poly is None:
        return mission_id, {
            "feasibility": False,
            "reason": "No valid bias term found",
            "mission_id": mission_id
        }
    # Step 2: Verify feasibility of the resulting HOCBF
    relative_degrees = [2]
    alphas = [
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
        switching_cbfs=np.array([hocbf_poly]),
        static_cbfs=None,
        relative_degree=relative_degrees,
        alpha=alphas,
        state_eq_constr=None,
        lagrangian_degrees=lagrangian_degrees
    )
    if first_hocbf:
        return mission_id, {
            "feasibility": is_feasible,
            "mission_id": mission_id,
            "hocbf_coefficients": [normal_vector[0],
                                   normal_vector[1],
                                   hocbf_bias],
            "config": config
            }
    else:
        former_hocbf = sym.Polynomial(
            former_hocbf_coefficients[0]*x[0] +
            former_hocbf_coefficients[1]*x[1] +
            former_hocbf_coefficients[2]
        )
        overlap_with_former = verification.verify_hocbfs_overlap(
            x=x,
            f=f,
            g=g,
            state_eq_constr=None,
            cbfs=np.array([former_hocbf, hocbf_poly]),
            relative_degree=[2, 2],
            alphas=[[0.01, 0.1], [0.01, 0.1]]
            )
        return mission_id, {
            "feasibility": is_feasible,
            "overlap_with_former": overlap_with_former,
            "mission_id": mission_id,
            "hocbf_coefficients": [normal_vector[0],
                                   normal_vector[1],
                                   hocbf_bias],
            "config": config
            }


def run_parallel_verification_for_synthesis(
    missions: List[Tuple[np.ndarray, dict]],
    first_hocbf: bool = True,
    former_hocbf_coefficients: Optional[np.ndarray] = None,
    max_workers: int = None,
    timeout: int = None
) -> dict:
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1
    
    results = {}
    
    # IMPORTANT: Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all missions
        future_to_mission = {}
        for normal_vector, config in missions:
            future = executor.submit(
                single_mission,
                normal_vector,
                config,
                first_hocbf,
                former_hocbf_coefficients
                )
            future_to_mission[future] = config.get("mission_id")
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


def search_single_hocbf(
    anchor_theta: float,
    num_normal_vector_samples: int = 10,
    range_theta: float = np.pi/2,
    first_hocbf: bool = True,
    former_hocbf_coefficients: Optional[np.ndarray] = None
) -> Optional[List[float]]:
    normal_vectors = sample_normal_vector(
        anchor_theta=anchor_theta,
        num_samples=num_normal_vector_samples,
        theta_range=range_theta
    )
    missions = create_missions(normal_vectors)
    results = run_parallel_verification_for_synthesis(
        missions=missions,
        first_hocbf=first_hocbf,
        former_hocbf_coefficients=former_hocbf_coefficients
    )
    # Find the best feasible HOCBF
    best_hocbf = None
    best_theta_diff = float('inf')
    for id in range(len(missions)):
        result = results.get(id)
        if result is None:
            continue
        elif result.get("feasibility") is True:
            if not first_hocbf and result.get("overlap_with_former") is True:
                hocbf_coeffs = result.get("hocbf_coefficients")
                a = hocbf_coeffs[0]
                b = hocbf_coeffs[1]
                theta = np.arctan2(b, a)
                theta_diff = abs(((theta - anchor_theta + np.pi) % (2 * np.pi)) - np.pi)
                if theta_diff < best_theta_diff:
                    best_theta_diff = theta_diff
                    best_hocbf = hocbf_coeffs
            elif first_hocbf:
                hocbf_coeffs = result.get("hocbf_coefficients")
                a = hocbf_coeffs[0]
                b = hocbf_coeffs[1]
                theta = np.arctan2(b, a)
                theta_diff = abs(((theta - anchor_theta + np.pi) % (2 * np.pi)) - np.pi)
                if theta_diff < best_theta_diff:
                    best_theta_diff = theta_diff
                    best_hocbf = hocbf_coeffs                
    return best_hocbf        


def search_union_hocbfs() -> List[List[float]]:
    """
    This function searches for a union of linear HOCBFs in the 2D positional
    domain. It sequentially searches for each HOCBF in the union by specifying
    an anchor angle for each HOCBF's normal vector. We initially set the anchor
    angle to be 0 for the first HOCBF, then every subsequent HOCBF's anchor angle
    is the former best HOCBF's normal vector angle plus a fixed angle increment.
    We provide two types of stopping criteria:
    1. maximum number of HOCBFs in the union is reached,
    2. a set of possible initial positions are all covered by the current union
       of HOCBFs. 
    """
    init_angle = 0.0
    angle_increment = np.pi / 2
    max_hocbfs = 4
    # specify some initial states for the system:
    # the states are sampled from a square region:
    # x_0 ∈ [-2.4, -2.1], x_1 ∈ [-1, -0.5]
    covered_points = np.random.uniform(
        low=[-2.4, -1.0],
        high=[-2.1, -0.5],
        size=(50, 2)
    )
    searched_hocbfs = []
    former_hocbf_coefficients = None
    for i in range(max_hocbfs):
        anchor_theta = init_angle + i * angle_increment
        print(f"Searching for HOCBF {i} with anchor theta {anchor_theta:.2f} rad")
        best_hocbf = search_single_hocbf(
            anchor_theta=anchor_theta,
            num_normal_vector_samples=10,
            range_theta=np.pi/2,
            first_hocbf=(i==0),
            former_hocbf_coefficients=former_hocbf_coefficients
        )
        if best_hocbf is not None:
            print(f"Found feasible HOCBF {i}: {best_hocbf}")
            searched_hocbfs.append(best_hocbf)
            former_hocbf_coefficients = np.array(best_hocbf)
            if check_union_covered_points(
                intrest_points=covered_points,
                hocbf_coefficients_list=searched_hocbfs
            ):
                print("All intrest points are covered by the current union of HOCBFs.")
                break
        else:
            print(f"No feasible HOCBF found for index {i}, stopping search.")
            break
    return searched_hocbfs



if __name__ == "__main__":
    union_hocbfs = search_union_hocbfs()
        
    

