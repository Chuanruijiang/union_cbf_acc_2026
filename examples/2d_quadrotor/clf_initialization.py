"""
Since now we are working on a quadrotor nonlinear system,
then the CLF should not be simply initialized as
circular function. In stead, we will intialize it
using LQR. If the system also has a state-equation
constraints, we should then use the projected LQR,
and then use the cost-to-go function as the CLF.

After getting the CLF, we will continue to find bias
for the CLF expression so that the stabile region
{x| V(x) <= 1} includes the ball B(0, epsilon_0).
where the epsilon_0 is the specified radius.
"""

import os
import sys
import os.path
import pickle
import numpy as np
from typing import Optional, Tuple

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from union_cbf_base.utils import serialize_polynomial
from union_cbf_base import clf
from union_cbf_base.inclusion import BallInclusion
from dynamics import (Quadrotor2dPlant, Quadrotor2dTrigPlant)


sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../.."))


def get_pkl_file_path(filename: str):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path


def save_clf_init(
    V: Optional[sym.Polynomial],
    x_set: sym.Variables,
    pickle_path: str,
):
    """
    Save the CLF and CBF to a pickle file.
    """
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}

    data["V"] = serialize_polynomial(V, x_set)

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


def lqr(quadrotor: Quadrotor2dTrigPlant) -> Tuple[np.ndarray, np.ndarray]:
    x_des = np.zeros((7,))
    u_des = np.full((2,), quadrotor.m * quadrotor.g / 2)
    xdot_des = quadrotor.dynamics(x_des, u_des)
    np.testing.assert_allclose(xdot_des, np.zeros((7,)), atol=1e-10)

    A, B = quadrotor.linearize_dynamics(x_des, u_des)
    Q = np.diag([1, 1, 10, 10, 10, 10, 10.0])
    R = np.diag([10.0, 10])
    # Gradient of the constraint sin^2 + cos^2 = 1
    F = np.array([[0, 0, 0, 2, 0, 0, 0]])
    K, S = controllers.LinearQuadraticRegulator(
        A, B, Q, R, N=np.empty((0, 2)), F=F
    )
    return K, S


def find_trig_regional_clf(V_degree: int, x: np.ndarray) -> sym.Polynomial:
    quadrotor = Quadrotor2dTrigPlant()
    K_lqr, _ = lqr(quadrotor)
    u_lqr = -K_lqr @ x + np.full((2,), quadrotor.m * quadrotor.g / 2)
    dynamics_expr = quadrotor.dynamics(x, u_lqr)
    dynamics = np.array([sym.Polynomial(dynamics_expr[i]) for i in range(7)])
    positivity_eps = 0.01
    d = int(V_degree / 2)
    kappa = 1e-4
    state_eq_constraints = quadrotor.equality_constraint(x)
    positivity_ceq_lagrangian_degrees = [V_degree - 2]
    derivative_ceq_lagrangian_degrees = [int(np.ceil((V_degree + 1) / 2) * 2 - 2)]
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 1e-4)])
    positivity_cin_lagrangian_degrees = [V_degree - 2]
    derivative_cin_lagrangian_degrees = derivative_ceq_lagrangian_degrees

    prog, V = clf.find_candidate_regional_lyapunov(
        x,
        dynamics,
        V_degree,
        positivity_eps,
        d,
        kappa,
        state_eq_constraints,
        positivity_ceq_lagrangian_degrees,
        derivative_ceq_lagrangian_degrees,
        state_ineq_constraints,
        positivity_cin_lagrangian_degrees,
        derivative_cin_lagrangian_degrees,
    )
    result = solvers.Solve(prog)
    assert result.is_success()
    V_sol = result.GetSolution(V)
    return V_sol


def taylor_expansion_clf_initialization(
) -> Tuple[sym.Polynomial, np.ndarray]:
    x = sym.MakeVectorContinuousVariable(6, "x")
    quadrotor = Quadrotor2dPlant()
    x_eq = np.zeros(6)
    u_eq = np.array([
        0.5 * quadrotor.m * quadrotor.g,
        0.5 * quadrotor.m * quadrotor.g
        ])

    A, B = quadrotor.linearize_dynamics(x_des=x_eq, u_des=u_eq)

    # We want to put more emphasis on the postion of the turtle
    # bot instead of all the position and orientation. Therefore,
    # we will only penalize the first two states. Also in the
    # control cost, we put more weight on the velocity than the
    # angular velocity.
    Q = np.diag([1, 1, 1, 10, 10, 10])
    R = np.diag([10.0, 10.0])

    _, S_lqr = controllers.LinearQuadraticRegulator(A, B, Q, R)
    V_init = sym.Polynomial(x.dot(S_lqr @ x) / 0.01)
    V_init = V_init.RemoveTermsWithSmallCoefficients(1e-10)

    return (V_init, x)


def trig_poly_initialization(
) -> Tuple[sym.Polynomial, np.ndarray]:
    x = sym.MakeVectorContinuousVariable(7, "x")
    V_degree = 2
    V_init = find_trig_regional_clf(V_degree, x)

    return (V_init, x)


def main(trig_poly: bool, save_clf: bool):
    if trig_poly:
        (V_init, x) = trig_poly_initialization()
        filename = "2d_quadrotor_clf_init_trig.pkl"
    else:
        (V_init, x) = taylor_expansion_clf_initialization()
        filename = "2d_quadrotor_clf_init_taylor.pkl"

    # Now we have the initial CLF, we will continue to find the
    # bias for the CLF expression so that the stabile region
    # {x| bias - V(x) ≥ 0} includes the ball B(0, epsilon_0).
    epsilon_0 = 0.1
    bias_start = 1
    bias_end = 1e6
    bias = bias_start
    while bias <= bias_end:
        ball_inclusion = BallInclusion(radius=epsilon_0, h=(bias - V_init), x=x)
        if ball_inclusion.verify_ball_inclusion(ball_x_degree=2, h_x_degree=2):
            break
        bias *= 10

    print(bias)

    # Since we would like the stability region to be {x | v(x) ≤ 1}
    # during the synthesis, then, the initlized CLF should be V_lqr/bias:
    V_init = (1 / bias) * V_init
    if save_clf:
        save_clf_init(
            V=V_init,
            x_set=sym.Variables(x),
            pickle_path=get_pkl_file_path(filename=filename)
        )


if __name__ == "__main__":
    main(
        trig_poly=True,
        save_clf=True,
    )
