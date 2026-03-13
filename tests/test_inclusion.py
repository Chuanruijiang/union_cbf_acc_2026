import numpy as np
import pydrake.symbolic as sym
import union_cbf_base.inclusion as inclusion


def test_unsafe_exclusion_poly():
    x = sym.MakeVectorContinuousVariable(2, "x")
    # the unsafe region is the folling rectangle: 
    # x0 in [ -1, 3], x1 in [-1, 1]
    obstacle = np.array([
        sym.Polynomial(x[0] - 3),
        sym.Polynomial(-x[0] - 1),
        sym.Polynomial(x[1] - 1),
        sym.Polynomial(-x[1] - 1)
    ])
    h_center = np.array([-2, 2])
    h = sym.Polynomial(0.5**2 - (x - h_center).dot(x - h_center))

    check_object = inclusion.UnsafeExclusion(
        h=h,
        x=x,
        unsafe_polys=obstacle,
        unsafe_points=None,
    )
    
    check_result = check_object.verify_unsafe_exclusion(
        unsafe_poly_x_degrees=[2,2,2,2],
        h_x_degree=0
    )

    assert check_result is True

def test_unsafe_exclusion_points():
    x = sym.MakeVectorContinuousVariable(2, "x")
    # the unsafe region is the folling rectangle: 
    # x0 in [ -1, 3], x1 in [-1, 1]
    # we sample 1000 points in the unsafe region
    num_points = 1000
    x0_samples = np.random.uniform(-1, 3, num_points)
    x1_samples = np.random.uniform(-1, 1, num_points)
    unsafe_points = np.vstack((x0_samples, x1_samples)).T
    h_center = np.array([-2, 2])
    h = sym.Polynomial(0.5**2 - (x - h_center).dot(x - h_center))
    check_object = inclusion.UnsafeExclusion(
        h=h,
        x=x,
        unsafe_polys=None,
        unsafe_points=unsafe_points,
    )
    check_result = check_object.verify_unsafe_point_exclusion()
    assert check_result == True
    assert False == True
    assert False