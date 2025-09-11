import numpy as np
import pydrake.symbolic as sym
from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_high_ord_cbf import UnionHighOrderCBF
from dynamics import (
    system_dynamics,
    equation_constraint,
    control_input_limits
)

def main():
    x = sym.MakeVectorContinuousVariable(4, "x")
    f, g = system_dynamics(x)
    eq_constriant = equation_constraint(x)
    (A, c) = control_input_limits()
    cbfs = np.array([
        sym.Polynomial(1.0 - (x[0])**2 - (x[1])**2),
    ])
    union_object = UnionHighOrderCBF(
        x=x,
        f=f,
        g=g,
        cbfs=cbfs,
        alpha1=0.1,
        alpha2=0.1,
        control_limits=(A, c),
        state_eq_constraint=eq_constriant
    )
    verification_flag = union_object.check_single_cbf(
        cbf_index=0,
        lagrangian_cbf_x_degree=2,
        lagrangian_cbf_y_degree=2,
        lagrangian_cbf_dot_x_degree=2,
        lagrangian_cbf_dot_y_degree=2,
        lagrangian_s_x_degree=2,
        lagrangian_s_y_degree=0,
        lagrangian_q_x_degree=2,
        lagrangian_q_y_degree=0,
        lagrangian_state_eq_x_degree=2,
        lagrangian_state_eq_y_degree=2,
        eta=0,
        epsilon=0.01,
    )

    assert verification_flag == True



if __name__ == "__main__":
    main()