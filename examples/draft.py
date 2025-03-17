import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import numpy as np
import pydrake.symbolic as sym
from compatible_clf_union_cbf.inclusion import(
    BallInclusion
)

def main():
    x = sym.MakeVectorContinuousVariable(3, "x")
    h = sym.Polynomial(x[1] - np.sin(-np.pi/6))
    ball_inclusion = BallInclusion(
        radius=0.1,
        h=h,
        x=x)
    check_result = ball_inclusion.verify_ball_inclusion(
        ball_x_degree=2,
        h_x_degree=2
        )
    
    assert check_result == True


    

if __name__ == "__main__":
    main()