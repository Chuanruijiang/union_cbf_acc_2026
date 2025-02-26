import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.ball_inclusion as mut


def checking_ball_inclusion():
    x = sym.MakeVectorContinuousVariable(2, "x")
    # case 1:
    center = np.array([10, 0])
    h = np.array([
        sym.Polynomial(5 - (x - center).dot(x - center)),
    ])
    dut = mut.BallInclusion(
        r_start=10,
        r_lower_bound=0.01,
        polys=h,
        x=x,
    )
    degrees = np.array([
        [2, 2]
    ])
    checking_result = dut.ball_inclusion_verification(
        degrees=degrees
    )
    expeccted_result = False
    assert checking_result == expeccted_result

    # case 2:
    center = np.array([1, 0])
    h = np.array([
        sym.Polynomial(5 - (x - center).dot(x - center)),
    ])
    dut = mut.BallInclusion(
        r_start=10,
        r_lower_bound=0.01,
        polys=h,
        x=x,
    )
    degrees = np.array([
        [2, 2]
    ])
    checking_result = dut.ball_inclusion_verification(
        degrees=degrees
    )
    expeccted_result = True
    assert checking_result == expeccted_result