from union_cbf_base.union_cbf import UnionCbf
from union_cbf_base.non_empty_subset import Subset
import union_cbf_base.utils as utils
import pydrake.symbolic as sym
import numpy as np
import pydrake.solvers as solvers

def main():
    
    prog = solvers.MathematicalProgram()
    x = prog.NewContinuousVariables(2, "x")
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([0.5, 0.5, 0.5, 0.5])
    cost = np.zeros(shape=(2,)).dot(x) + 2.0
    prog.AddLinearCost(e=cost)
    prog.AddLinearConstraint(
        A=A, lb=np.full_like(b, -np.inf), ub=b, vars=x
        )
    result = solvers.Solve(prog)
    assert result.is_success()
    print("optimal cost:", result.get_optimal_cost())


if __name__ == "__main__":
    main()
