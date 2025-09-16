import numpy as np
import pydrake.symbolic as sym
import union_cbf_base.inclusion as icl


def test_unsafe_exclusion():
    x = sym.MakeVectorContinuousVariable(2, "x")
    obstacle = np.array([
        sym.Polynomial(x[0] - 3),
        sym.Polynomial(-x[0] - 1),
        sym.Polynomial(x[1] - 1),
        sym.Polynomial(-x[1] - 1)
    ])
    h_center = np.array([-2, 2])
    h = sym.Polynomial(0.5**2 - (x - h_center).dot(x - h_center))

    check_object = icl.UnsafeExclusion(
        unsafe_polys=obstacle,
        h=h,
        x=x
    )
    
    check_result = check_object.verify_unsafe_exclusion(
        unsafe_poly_x_degrees=[2,2,2,2],
        h_x_degree=0
    )

    assert check_result is True