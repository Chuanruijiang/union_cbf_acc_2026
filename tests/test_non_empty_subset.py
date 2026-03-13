from union_cbf_base.non_empty_subset import Subset
import numpy as np
import pydrake.symbolic as sym

def test_subset_emptiness_check():
    x = sym.MakeVectorContinuousVariable(2, "x")
    # big test case 1:
    all_polys = np.array([
        sym.Polynomial((0.8)**2 - (x[0]-0.5)**2 - (x[1] - 0)**2),
        sym.Polynomial((0.8)**2 - (x[0]+0.5)**2 - (x[1] - 0)**2)
    ])
    # subcase 1:
    subset_1 = Subset(
        x = x,
        all_polys = all_polys,
        activation_index=np.array([1, 1])
    )
    emptiness_check_result = subset_1.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=0,
        lagrangian_deact_c_degree=0
    )
    assert emptiness_check_result == False
    # subcase 2:
    subset_1.activation_index = np.array([1, 0])
    emptiness_check_result = subset_1.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=2,
        lagrangian_deact_c_degree=0
    )
    assert emptiness_check_result == False
    # subcase 3:
    subset_1.activation_index = np.array([0, 1])
    emptiness_check_result = subset_1.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=2,
        lagrangian_deact_c_degree=0
    )
    assert emptiness_check_result == False

    # big test case 2:
    all_polys = np.array([
        sym.Polynomial((0.4)**2 - (x[0]-0.5)**2 - (x[1] - 0)**2),
        sym.Polynomial((0.4)**2 - (x[0]+0.5)**2 - (x[1] - 0)**2)
    ])
    # subcase 1:
    subset_2 = Subset(
        x = x,
        all_polys = all_polys,
        activation_index=np.array([1, 1])
    )
    emptiness_check_result = subset_2.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=0,
        lagrangian_deact_c_degree=0
    )
    assert emptiness_check_result == True
    # subcase 2:
    subset_2.activation_index = np.array([1, 0])
    emptiness_check_result = subset_2.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=2,
        lagrangian_deact_c_degree=0
    )
    assert emptiness_check_result == False
    # subcase 3:
    subset_2.activation_index = np.array([0, 1])
    emptiness_check_result = subset_2.is_empty(
        lagrangian_x_degree=2,
        lagrangian_act_c_degree=2,
        lagrangian_deact_c_degree=0
    )
    assert emptiness_check_result == False
