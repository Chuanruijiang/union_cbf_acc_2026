import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.ball_inclusion as ball_inclusion


def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    center = np.array([10, 0])
    h = np.array([
        sym.Polynomial(5 - (x - center).dot(x - center)),
    ])
    ball_inclustion_checking = ball_inclusion.BallInclusion(
        r_start=10,
        r_lower_bound=0.01,
        polys=h,
        x=x,
    )
    degrees = np.array([
        [2, 2]
    ])
    checking_result = ball_inclustion_checking.ball_inclusion_verification(
        degrees=degrees
    )
    assert checking_result


if __name__ == "__main__":
    main()
