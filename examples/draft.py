import numpy as np
import pydrake.symbolic as sym
import union_cbf_base.utils as utils
from union_cbf_base.union_cbf_II import (
    Degree,
    UnionCbfII,
    CbfFeasibilityLagrangianDegrees,
)
from union_cbf_base.non_empty_subset import Subset


def main():
    # x = sym.MakeVectorContinuousVariable(3, "x")
    # h = (x[0] - 2.5) ** 2 + (x[1] - 3.5) ** 2 - (1.6)**2
    # V = 0.75
    # alpha1 = 1.0
    # alpha2 = 10.0
    # f = np.array([
    #     V * np.cos(x[2]),
    #     V * np.sin(x[2]),
    #     sym.Expression(0.0),
    # ])
    # g = np.array([[sym.Expression(0.0)], [sym.Expression(0.0)], [sym.Expression(1.0)]])
    # Lfh = utils.lie_derivative(
    #     poly=h, vector_field=f, variables=x, pow=1
    # )
    # phi_1 = Lfh + alpha1*h
    # L2fh = utils.lie_derivative(
    #     poly=phi_1, vector_field=f, variables=x, pow=1
    # )
    # LgLfh = utils.lie_derivative(
    #     poly=phi_1, vector_field=g, variables=x, pow=1  
    # )
    # phi_2_xi = L2fh + (alpha1 + alpha2)*Lfh + alpha1*alpha2*h
    # phi_2_lambda = -LgLfh

    # x0_val, x1_val = -0.1, 3.5
    # env = {x[0]: x0_val, x[1]: x1_val, x[2]: np.arctan2(3.5 - x1_val, 2.5 - x0_val)}
    # phi_0_val = h.Evaluate(env)
    # phi_1_val = phi_1.Evaluate(env)
    # phi_2_xi_val = phi_2_xi.Evaluate(env)
    # phi_2_lambda_val = np.array([
    #     phi_2_lambda[i].Evaluate(env) 
    #     for i in range(phi_2_lambda.shape[0])
    # ])
    # print(f"checking at x0: {x0_val}, x1: {x1_val}, x2: {env[x[2]]}")
    # print(f"phi_0_val: {phi_0_val}")
    # print(f"phi_1_val: {phi_1_val}")
    # print(f"phi_2_xi_val: {phi_2_xi_val}")
    # print(f"phi_2_lambda_val: {phi_2_lambda_val}")
    pass






if __name__ == "__main__":
    main()