import os
import sys
import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.union_cbf as mut
from compatible_clf_union_cbf.clf_cbf import XYDegree
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))


def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(x[1]),
        -sym.Polynomial(0),
    ])
    g = np.array([
        [0],
        [1]
    ])
    h = sym.Polynomial(1 - x[0])
    relative_degree = 2
    kappa_h = [0.1, 0.2]
    cbf_const = mut.CbfConstraint(
        h=h,
        f=f,
        g=g,
        x=x,
        kappa=kappa_h,
        relative_degree=relative_degree
    )
    linear_const = cbf_const.add_to_prog(
        
    )


if __name__ == "__main__":
    main()
