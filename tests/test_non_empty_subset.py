import numpy as np
import pydrake.symbolic as sym
import union_cbf_base.non_empty_subset as nes

def test_subset_emptiness_check():
    x = sym.MakeVectorContinuousVariable(2, "x")
    cbfs = np.array([
        (0.8)**2 - (x[0]-0.5)**2 - (x[1] - 0)**2,
        (0.8)**2 - (x[0]+0.5)**2 - (x[1] - 0)**2
    ])
    subset_1 = nes.Subset(
        x = x,
        cbfs = cbfs,
        activation_index=np.array([1, 1])
    )
    emptiness_check_result = subset_1.is_empty()
    assert emptiness_check_result == False

    subset_1.activation_index = np.array([1, 0])
    emptiness_check_result = subset_1.is_empty()
    assert emptiness_check_result == False

    subset_1.activation_index = np.array([0, 1])
    emptiness_check_result = subset_1.is_empty()
    assert emptiness_check_result == False

    subset_1.activation_index = np.array([0, 0])
    emptiness_check_result = subset_1.is_empty()
    assert emptiness_check_result == False

    cbfs = np.array([
        (0.4)**2 - (x[0]-0.5)**2 - (x[1] - 0)**2,
        (0.4)**2 - (x[0]+0.5)**2 - (x[1] - 0)**2
    ])
    subset_2 = nes.Subset(
        x = x,
        cbfs = cbfs,
        activation_index=np.array([1, 1])
    )
    emptiness_check_result = subset_2.is_empty()
    assert emptiness_check_result == True

    subset_2.activation_index = np.array([1, 0])
    emptiness_check_result = subset_2.is_empty()
    assert emptiness_check_result == False

    subset_2.activation_index = np.array([0, 1])
    emptiness_check_result = subset_2.is_empty()
    assert emptiness_check_result == False

    subset_2.activation_index = np.array([0, 0])
    emptiness_check_result = subset_2.is_empty()
    assert emptiness_check_result == False
