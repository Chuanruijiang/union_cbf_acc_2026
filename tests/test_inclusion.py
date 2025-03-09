import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.inclusion as icl


def test_ball_inclusion():
    x = sym.MakeVectorContinuousVariable(2, "x")
    radius = 1
    h_center = np.array([3, 0])
    h_radius = 5
    h = sym.Polynomial(h_radius**2 - (x - h_center).dot(x - h_center))

    check_object = icl.BallInclusion(
        radius=radius,
        h=h,
        x=x
    )

    check_result = check_object.verify_ball_inclusion(
        ball_x_degree=2,
        h_x_degree=2
    )

    assert check_result is True
