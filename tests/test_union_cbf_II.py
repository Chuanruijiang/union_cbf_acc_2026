import numpy as np
import pydrake.symbolic as sym

from union_cbf_base.utils import(
    Degree,
    check_polynomial_arrays_equal
)
from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf_II import(
    CbfFeasibilityLagrangianDegrees,
    UnionCbfII
)
from examples.single_integrator_2D import dynamics
def test_all_subsets_to_verify():
    x = sym.MakeVectorContinuousVariable(2, "x")
    plant = dynamics.SingleIntegrator2D()
    (f, g) = plant.affine_dynamics(x)
    control_limits = plant.control_limits() 

    # 2 static CBFs:
    static_cbfs = np.array([
        sym.Polynomial(x[0] + x[1] + 4),
        sym.Polynomial(x[0]**2 + x[1]**2 + 9)
    ])
    # 3 switching CBFs:
    switching_cbfs = np.array([
        sym.Polynomial(x[0] - x[1] + 1),
        sym.Polynomial(x[0] + 1),
        sym.Polynomial(x[0] + x[1] + 1)
    ])
    relative_degrees = [1,1,1]
    alphas = [
        [0.5],
        [0.3],
        [2.1]
    ]

    test_obj = UnionCbfII(
        x=x,
        f=f,
        g=g,
        alpha=alphas,
        relative_degree=relative_degrees,
        control_limits=control_limits,
        state_eq_constr=None
    )

    all_subsets = test_obj.all_subsets_to_verify(
        switching_cbfs=switching_cbfs,
        static_cbfs=static_cbfs
    )

    # test subsets:
    assert len(all_subsets) == switching_cbfs.shape[0]
    for i in range(len(all_subsets)):
        assert all_subsets[i].deactivated_polys == None
        assert all_subsets[i].subset_component_indicator == None
        assert len(all_subsets[i].activated_poly_groups) == \
            static_cbfs.shape[0] + 1
        expected_first_group = test_obj._all_phi_polys_given_h(
            cbf = switching_cbfs[i],
            relative_degree=relative_degrees[0],
            alphas=alphas[0]
        )
        check_polynomial_arrays_equal(
            p = expected_first_group,
            q = all_subsets[i].activated_poly_groups[0],
            tol=1e-8
        )
        for j in range(static_cbfs.shape[0]):
            expected_static_group = test_obj._all_phi_polys_given_h(
                cbf = static_cbfs[j],
                relative_degree=relative_degrees[j+1],
                alphas=alphas[j+1]
            )
            check_polynomial_arrays_equal(
                p = expected_static_group,
                q = all_subsets[i].activated_poly_groups[j+1],
                tol=1e-8
            )

def test_compute_xi_lambda():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([sym.Polynomial(0), sym.Polynomial(0)])
    g = np.array(
        [[sym.Polynomial(1), sym.Polynomial(0)], [sym.Polynomial(0), sym.Polynomial(1)]]
    )
    A = np.array(
        [
            [sym.Polynomial(1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(1)],
            [sym.Polynomial(-1), sym.Polynomial(0)],
            [sym.Polynomial(0), sym.Polynomial(-1)],
        ]
    )
    c = np.array(
        [sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1), sym.Polynomial(1)]
    )
    cbfs = np.array(
        [
            sym.Polynomial((0.6) ** 2 - (x[0] - 0.5) ** 2 - (x[1] - 0) ** 2),
            sym.Polynomial((0.6) ** 2 - (x[0] + 0.5) ** 2 - (x[1] - 0) ** 2),
        ]
    )
    alpha = [[0.1]]
    relative_degree = [1]
    eta = 0.1
    epsilon = 0.01
    test_obj = UnionCbfII(
        x=x,
        f=f,
        g=g,
        alpha=alpha,
        relative_degree=relative_degree,
        control_limits=(A, c),
    )
    # test case 1: first cbf
    expected_lambda = np.array([
        [sym.Polynomial(2*(x[0]-0.5)), sym.Polynomial(2*(x[1]-0))],
        [sym.Polynomial(1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(1)],
        [sym.Polynomial(-1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(-1)],
    ])
    expected_xi = np.array([
        sym.Polynomial(-2*(x[0]-0.5)*f[0] - 2*(x[1]-0)*f[1] 
                       + alpha[0][0]*((0.6)**2 - (x[0]-0.5)**2 - (x[1]-0)**2) 
                       - eta),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
    ])
    subset = Subset(
        x=x,
        activated_poly_groups=[np.array([cbfs[0]], dtype=object)],
        deactivated_polys=None,
        equation_constraints=None,
        num_avaliable_switching_cbfs=1,
        mask_avaliable_switching_cbfs=np.array([1, 0]),
        subset_component_indicator=None,
    )
    computed_lambda, computed_xi = test_obj._xi_lambda_(
        subset=subset,
        eta=eta,
        eps=epsilon,
    )
    check_polynomial_arrays_equal(computed_lambda, expected_lambda, tol=1e-8)
    check_polynomial_arrays_equal(computed_xi, expected_xi, tol=1e-8)

    # test case 2: second cbf
    expected_lambda = np.array([
        [sym.Polynomial(2*(x[0]+0.5)), sym.Polynomial(2*(x[1]-0))],
        [sym.Polynomial(1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(1)],
        [sym.Polynomial(-1), sym.Polynomial(0)],
        [sym.Polynomial(0), sym.Polynomial(-1)],
    ])
    expected_xi = np.array([
        sym.Polynomial(-2*(x[0]+0.5)*f[0] - 2*(x[1]-0)*f[1] 
                       + alpha[0][0]*((0.6)**2 - (x[0]+0.5)**2 - (x[1]-0)**2) 
                       - eta),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
        sym.Polynomial(1 - epsilon),
    ])
    subset = Subset(
        x=x,
        activated_poly_groups=[np.array([cbfs[1]], dtype=object)],
        deactivated_polys=None,
        equation_constraints=None,
        num_avaliable_switching_cbfs=1,
        mask_avaliable_switching_cbfs=np.array([0, 1]),
        subset_component_indicator=None,
    )
    computed_lambda, computed_xi = test_obj._xi_lambda_(
        subset=subset,
        eta=eta,
        eps=epsilon,
    )
    check_polynomial_arrays_equal(computed_lambda, expected_lambda, tol=1e-8)
    check_polynomial_arrays_equal(computed_xi, expected_xi, tol=1e-8)
