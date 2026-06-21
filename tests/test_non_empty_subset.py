from union_cbf_base.non_empty_subset import Subset
import numpy as np
import pydrake.symbolic as sym

def test_subset_emptiness_check():
    x = sym.MakeVectorContinuousVariable(2, "x")
    polys = np.array([
        sym.Polynomial((0.8)**2 - (x[0]-0.5)**2 - (x[1] - 0)**2),
        sym.Polynomial((0.8)**2 - (x[0]+0.5)**2 - (x[1] - 0)**2),
        sym.Polynomial((0.4)**2 - (x[0]-0)**2 - (x[1] - 0.5)**2),
        sym.Polynomial((0.4)**2 - (x[0]-0)**2 - (x[1] + 0.5)**2)
    ])
    subset_1 = Subset(
        x=x,
        activated_poly_groups = [polys[0:1], polys[1:2]],
        deactivated_polys = polys[2:4],
        equation_constraints = np.array([sym.Polynomial(x[1] - 2)]),
        num_avaliable_switching_cbfs=None,
        mask_avaliable_switching_cbfs=None,
        subset_component_indicator=None,
    )
    emptiness_check_result = subset_1.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=2,
        lagrangian_deact_c_degree=2
    )
    assert emptiness_check_result == True

    subset_2 = Subset(
        x=x,
        activated_poly_groups = [
            np.array([polys[0], polys[2]]),
            np.array([polys[1], polys[3]])
        ],
        deactivated_polys=None,
        equation_constraints=None,
        num_avaliable_switching_cbfs=None,
        mask_avaliable_switching_cbfs=None,
        subset_component_indicator=None,
    )
    emptiness_check_result = subset_2.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=0,
        lagrangian_deact_c_degree=0
    )
    assert emptiness_check_result == True
