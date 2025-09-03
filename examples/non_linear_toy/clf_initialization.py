"""
Since now we are working on a nonlinear dynmaics,
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
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers
from typing import Optional
from union_cbf_base.utils import system_linearization, serialize_polynomial
from union_cbf_base.inclusion import BallInclusion
from dynamics import system_dynamics

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../.."))


def get_pkl_file_path():
    filename = "non_linear_toy_clf_init.pkl"
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


def main():
    x = sym.MakeVectorContinuousVariable(3, "x")
    f, g = system_dynamics(x)

    x_eq = np.array([0, 0, 0])
    u_eq = np.array([0, 0])

    eq_point = (x_eq, u_eq)

    A, B = system_linearization(f=f, g=g, states=x, eq_point=eq_point)

    # We want to put more emphasis on the postion of the turtle
    # bot instead of all the position and orientation. Therefore,
    # we will only penalize the first two states. Also in the
    # control cost, we put more weight on the velocity than the
    # angular velocity.
    R = np.eye(2)
    Q = np.eye(3)
    F = np.array([[0, 0, 2]])  # linearization of the state eq-const

    _, S_lqr = controllers.LinearQuadraticRegulator(A, B, Q, R, F=F)

    V_lqr = sym.Polynomial(np.dot(x, np.dot(S_lqr, x)))

    # Now we have the initial CLF, we will continue to find the
    # bias for the CLF expression so that the stabile region
    # {x| bias - V(x) ≥ 0} includes the ball B(0, epsilon_0).
    epsilon_0 = 0.1
    bias_start = 1
    bias_end = 1e6
    bias = bias_start
    while bias <= bias_end:
        ball_inclusion = BallInclusion(radius=epsilon_0, h=(bias - V_lqr), x=x)
        if ball_inclusion.verify_ball_inclusion(ball_x_degree=2, h_x_degree=2):
            break
        bias *= 10

    print(bias)

    # Since we would like the stability region to be {x | v(x) ≤ 1}
    # during the synthesis, then, the initlized CLF should be V_lqr/bias:
    V_init = (1 / bias) * V_lqr
    save_clf_init(V=V_init, x_set=sym.Variables(x), pickle_path=get_pkl_file_path())


if __name__ == "__main__":
    main()
