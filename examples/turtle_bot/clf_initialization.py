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
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../.."))

import numpy as np
import pydrake.symbolic as sym
import pydrake.systems.controllers as controllers
from compatible_clf_union_cbf.utils import(
    system_linearization
)
from compatible_clf_union_cbf.clf_union_cbf import(
    CompatibleClfUnionCbfs
)
from dynamics import system_dynamics

def main():
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = system_dynamics(x)

    x_eq = np.array([0, 0, 0, 0])
    u_eq = np.array([0, 0])

    eq_point = (x_eq, u_eq)

    A, B = system_linearization(
        f=f,
        g=g, 
        states=x,
        eq_point=eq_point
        )
    
    # We want to put more emphasis on the postion of the turtle
    # bot instead of all the position and orientation. Therefore,
    # we will only penalize the first two states. Also in the
    # control cost, we put more weight on the velocity than the
    # angular velocity.
    R = np.eye(2)
    Q = np.eye(4)
    F = np.array([[0, 0, 0, 2]]) # linearization of the state eq-const

    _, S_lqr = controllers.LinearQuadraticRegulator(A, B, Q, R, F=F)

    V_init = sym.Polynomial(np.dot(x, np.dot(S_lqr, x)))

if __name__ == "__main__":
    main()

    


