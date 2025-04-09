import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.non_empty_subset as nes

def test_subset_emptiness_check():
    x = sym.MakeVectorContinuousVariable(2, "x")
    rho_minus_V = sym.Polynomial(49 - x.dot(x))
    ball_rad = 0.1
    r = np.array([0.1, 0.3, 0.49, 0.5, 0.51, 0.6])

    # test case 1
    expected_results = [True, True, True, True, False, False]
    for i in range(r.shape[0]):
        h = np.array(
            [
                sym.Polynomial(-((x[0] - 0.5) ** 2 + x[1] ** 2 - r[i] ** 2)),
                sym.Polynomial(-((x[0] + 0.5) ** 2 + x[1] ** 2 - r[i] ** 2)),
            ]
        )
        subset = nes.Subset(
            x=x,
            cbfs=h,
            rho_minus_clf=rho_minus_V,
            ball_radius=ball_rad,
            activation_index=np.array([1, 1])
        )
        result = subset.is_empty(
            lagrangian_x_degree=2,
            lagrangian_c_degree=2
        )
        assert result == expected_results[i]
    
    # test case 2
    expected_results = [False, False, False, False, False, False]
    for i in range(r.shape[0]):
        h = np.array(
            [
                sym.Polynomial(-((x[0] - 0.5) ** 2 + x[1] ** 2 - r[i] ** 2)),
                sym.Polynomial(-((x[0] + 0.5) ** 2 + x[1] ** 2 - r[i] ** 2)),
            ]
        )
        subset = nes.Subset(
            x=x,
            cbfs=h,
            rho_minus_clf=rho_minus_V,
            ball_radius=ball_rad,
            activation_index=np.array([1, 0])
        )
        result = subset.is_empty(
            lagrangian_x_degree=2,
            lagrangian_c_degree=2
        )
        assert result == expected_results[i]