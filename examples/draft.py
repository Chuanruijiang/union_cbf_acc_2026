import os
import sys
import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.union_cbf as mut
from compatible_clf_union_cbf.clf_cbf import XYDegree
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))


def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    r = 0.5
    h = np.array([
        sym.Polynomial(
            -((x[0] - 0.5)**2 + x[1]**2 - r**2)
            ),
        sym.Polynomial(
            -((x[0] + 0.5)**2 + x[1]**2 - r**2)
            ),
    ])
    subset = mut.UnionSubset(
        activated=h,
        deactivated=None,
        variables=x
    )

    emptiness_lagragian_degrees = mut.EmptinessLagrangianDegrees(
        activated=[XYDegree(x=2, y=0), XYDegree(x=2, y=0)],
        deactivated=None
    )
    result = subset.is_empty(emptiness_lagrangian_degrees=emptiness_lagragian_degrees)
    assert result


if __name__ == "__main__":
    main()
