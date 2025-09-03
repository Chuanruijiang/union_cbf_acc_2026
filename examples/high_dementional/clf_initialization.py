"""
This script initializes the CLF for the high dimensional systsm
The initilization strategy is to use the LQR and use the cost-to-go
function as the initial CLF.
We initialize the CLF for systems with 2,3,4,5,6,7,8 dimensions.
and then save them to a pickle file.
"""

import os
import sys
import os.path
import pickle
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../.."))

import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers

from union_cbf_base.utils import serialize_polynomial
from union_cbf_base.inclusion import BallInclusion
from dynamics import HighDementionalDyanmics


def get_pkl_file_path(filename: str):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/", filename
    )
    return path


def save_clfs_init(
    V: list[sym.Polynomial],
    x_sets: list[sym.Variables],
    pickle_path: str,
):
    _, file_extension = os.path.splitext(pickle_path)
    assert file_extension in (".pkl", ".pickle"), f"File extension is {file_extension}"
    data = {}
    for i in range(len(V)):
        data["V"+str(i)] = serialize_polynomial(V[i], x_sets[i])

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


def main(save_clf: bool):
    x = sym.MakeVectorContinuousVariable(8, "x")
    V_inits = [None] * 7
    x_sets = [None] * 7
    for i in range(2, 9):
        x_vars = x[0:i]
        x_sets[i-2] = sym.Variables(x_vars)
        high_dim_system = HighDementionalDyanmics(i)
        (A, B) = (high_dim_system.A, high_dim_system.B)
        Q = np.eye(i)
        R = np.eye(i)
        _, S_lqr = controllers.LinearQuadraticRegulator(A, B, Q, R)
        V_init = sym.Polynomial(x_vars.dot(S_lqr @ x_vars))
        V_init = V_init.RemoveTermsWithSmallCoefficients(1e-10)
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
        # print(bias)
        # Since we would like the stability region to be {x | v(x) ≤ 1}
        # during the synthesis, then, the initlized CLF should be V_lqr/bias:
        V_init = (1 / bias) * V_init
        V_inits[i-2] = V_init
    if save_clf:
        pickle_path = get_pkl_file_path("high_dementional_clf_init.pkl")
        save_clfs_init(V_inits, x_sets, pickle_path)


if __name__ == "__main__":
    main(save_clf=True)
