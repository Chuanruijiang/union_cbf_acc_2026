import numpy as np
import pydrake.symbolic as sym
import compatible_clf_union_cbf.clf_union_cbf as mut
from compatible_clf_union_cbf.utils import Degree
from compatible_clf_union_cbf.union_cbf import UnionSubset


def test_calc_xi_lambda():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(x[1]),
        sym.Polynomial(0)
    ])
    g = np.array([
        [sym.Polynomial(0)],
        [sym.Polynomial(1)]
    ])
    V = sym.Polynomial(x.dot(x))
    h = sym.Polynomial(1 - x.dot(x))
    range_non_negative = np.array([sym.Polynomial(5 - x.dot(x))])
    Au = np.array([
        [1], [-1]
    ])
    bu = np.array([1, 1])
    kappa_V = 1
    kappa_h = 1
    epsilon = 0.1
    test_object = mut.CompatibleClfCbfInRange(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        V=V,
        h=h,
        range_non_negative=range_non_negative,
        range_strictly_positive=None,
        state_eq_const=None,
        Au=Au,
        bu=bu
    )
    expected_lambda = np.array([
        [sym.Polynomial(2*x[1])],
        [sym.Polynomial(2*x[1])],
        [1],
        [-1]
    ])
    expected_xi = np.array([
        sym.Polynomial(-2*x[0]*x[1] + 1 - x.dot(x) - epsilon),
        sym.Polynomial(-2*x[0]*x[1] - x.dot(x) - epsilon),
        1 - epsilon,
        1 - epsilon
    ])
    xi, lambda_ = test_object._calc_xi_lambda(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        epsilon=epsilon
    )
    assert xi.shape == expected_xi.shape
    assert lambda_.shape == expected_lambda.shape
    for i in range(xi.shape[0]):
        if isinstance(xi[i], sym.Polynomial):
            assert xi[i].EqualTo(expected_xi[i])
        else:
            assert xi[i] == expected_xi[i]
    for i in range(lambda_.shape[0]):
        for j in range(lambda_.shape[1]):
            if isinstance(lambda_[i][j], sym.Polynomial):
                assert lambda_[i][j].EqualTo(expected_lambda[i][j])
            else:
                assert lambda_[i][j] == expected_lambda[i][j]

def test_compatibility_in_range():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(),
        sym.Polynomial()
    ])
    g = np.array([
        [sym.Polynomial(1), sym.Polynomial()],
        [sym.Polynomial(), sym.Polynomial(1)],
    ])
    V = sym.Polynomial(x[0]**2 + x[1]**2)
    h = 1 - V
    kappa_V = 0.1
    kappa_h = 0.1
    # case 1:
    range_strictly_positive = np.array([
        sym.Polynomial(1 - x[0]**2 - x[1]**2)
    ])
    range_non_negative = None
    state_eq_constraints = None
    Au = None
    bu = None

    check_compatibility = mut.CompatibleClfCbfInRange(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        V=V,
        h=h,
        range_non_negative=range_non_negative,
        range_strictly_positive=range_strictly_positive,
        state_eq_const=state_eq_constraints,
        Au=Au,
        bu=bu
    )

    # specify lagragian degrees:
    compatibility_lagrangian_degrees = mut.CompatibilityInRangeLagragianDegrees(
        lambda_y=[
            Degree(x=2, y=0, c=2),
            Degree(x=2, y=0, c=2)
        ],
        xi_y=Degree(x=2, y=0, c=2),
        range_non_negative=None,
        range_strictly_positive=[
            Degree(x=2, y=2, c=0)
        ],
        state_eq_const=None
    )

    # solve the compatibility problem:
    lagrangians = check_compatibility.verify_compatibility(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        epsilon=0,
        lagrangian_degrees=compatibility_lagrangian_degrees
    )

    assert lagrangians is not None

    # case 2:
    range_non_negative = np.array([
        sym.Polynomial(1 - x[0]**2 - x[1]**2)
    ])
    range_strictly_positive = None
    state_eq_constraints = None
    Au = None
    bu = None

    check_compatibility = mut.CompatibleClfCbfInRange(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        V=V,
        h=h,
        range_non_negative=range_non_negative,
        range_strictly_positive=range_strictly_positive,
        state_eq_const=state_eq_constraints,
        Au=Au,
        bu=bu
    )

    # specify lagragian degrees:
    compatibility_lagrangian_degrees = mut.CompatibilityInRangeLagragianDegrees(
        lambda_y=[
            Degree(x=2, y=0, c=0),
            Degree(x=2, y=0, c=0)
        ],
        xi_y=Degree(x=2, y=0, c=0),
        range_non_negative=[
            Degree(x=2, y=2, c=0)
        ],
        range_strictly_positive=None,
        state_eq_const=None
    )

    # solve the compatibility problem:
    lagrangians = check_compatibility.verify_compatibility(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        epsilon=0,
        lagrangian_degrees=compatibility_lagrangian_degrees
    )

    assert lagrangians is not None

def test_step_one_verification():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(),
        sym.Polynomial()
    ])
    g = np.array([
        [sym.Polynomial(1), sym.Polynomial()],
        [sym.Polynomial(), sym.Polynomial(1)],
    ])
    V = sym.Polynomial(x[0]**2 + x[1]**2)
    cbf_center = np.array([
        [-5, 0],
        [3, 0]
    ])
    cbf_radiuses = np.array([5, 5])
    h = np.array([
        sym.Polynomial(cbf_radiuses[i]**2 - (x - cbf_center[i]).dot(x - cbf_center[i]))
        for i in range(cbf_center.shape[0])
    ])
    kappa_V = 0.1
    kappa_h = [0.1, 0.1]

    # specify the step one:
    check = mut.StepOne(
        clf=V,
        cbfs=h,
        sys_dyn_f=f,
        sys_dyn_g=g,
        x=x,
        r_start=3,
        r_lower_bound=0.1,
        state_eq_constraints=None
    )
    
    eps_0 = check.step_one_verification(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        ball_inclusion_ball_x_degrees=[2, 2],
        ball_inclusion_cbf_x_degrees=[2, 2],
        qp_feasible_in_ball_lambda_y_x_degrees=[2, 2],
        qp_feasible_in_ball_xi_y_x_degrees=[2, 2],
        qp_feasible_in_ball_ball_x_degrees=[2, 2],
        qp_feasible_in_ball_state_eq_x_degrees=None,
    )

    expected_ball_exist = True
    assert (expected_ball_exist) == (eps_0 is not None)

def test_compatibility_verification_in_subset():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(),
        sym.Polynomial()
    ])
    g = np.array([
        [sym.Polynomial(1), sym.Polynomial()],
        [sym.Polynomial(), sym.Polynomial(1)],
    ])
    V = sym.Polynomial(x[0]**2 + x[1]**2)
    rho = 49
    cbf_center = np.array([
        [-5, 0],
        [3, 0]
    ])
    cbf_radiuses = np.array([5, 5])
    h = np.array([
        sym.Polynomial(cbf_radiuses[i]**2 - (x - cbf_center[i]).dot(x - cbf_center[i]))
        for i in range(cbf_center.shape[0])
    ])
    ball = sym.Polynomial(1 - x.dot(x))
    check_subset = UnionSubset(
        activated=np.array([rho-V, h[1]]),
        deactivated=np.array([h[0], ball]),
        variables=x
    )
    
    kappa_V = 0.1
    kappa_h = [1, 1]
    
    compatibility_in_subset_lagrangian_degrees = mut.SubsetVerifyLagrangianDegree(
        lambda_y=[
            [Degree(x=2, y=0, c=2), Degree(x=2, y=0, c=2)],
        ],
        xi_y=[
            Degree(x=2, y=0, c=2),
        ],
        activated_cbfs=[
            Degree(x=2, y=2, c=2),
        ],
        clf = Degree(x=2, y=2, c=2),
        deactivated_cbfs=[
            Degree(x=2, y=2, c=2)
        ],
        ball = Degree(x=2, y=2, c=2),
        state_eq_constraints=None
    )

    compatibility_object = mut.CompatibleClfCbfInSubset(
        x=x,
        sys_dyn_f=f,
        sys_dyn_g=g,
        subset=check_subset,
        Au = None,
        bu = None,
        state_eq_constraints=None,
    )

    (
        compatibility_in_subset_lagrangians
    ) = compatibility_object.verify_compatibility_in_subset(
        kappa_V=kappa_V,
        kappa_h=kappa_h,
        epsilon = 0.1,
        lagrangian_degrees=compatibility_in_subset_lagrangian_degrees
    )

    assert compatibility_in_subset_lagrangians is not None

def test_all_non_empty_subsets():
    x = sym.MakeVectorContinuousVariable(2, "x")
    f = np.array([
        sym.Polynomial(),
        sym.Polynomial()
    ])
    g = np.array([
        [sym.Polynomial(1), sym.Polynomial()],
        [sym.Polynomial(), sym.Polynomial(1)],
    ])
    V = sym.Polynomial(x[0]**2 + x[1]**2)
    rho = 49
    cbf_center = np.array([
        [-5, 0],
        [3, 0]
    ])
    cbf_radiuses = np.array([5, 5])
    h = np.array([
        sym.Polynomial(cbf_radiuses[i]**2 - (x - cbf_center[i]).dot(x - cbf_center[i]))
        for i in range(cbf_center.shape[0])
    ])
    radius = 1
    step_two_check = mut.StepTwo(
        clf = rho - V,
        cbfs=h,
        ball=sym.Polynomial(radius**2 - x.dot(x)),
        sys_dyn_f=f,
        sys_dyn_g=g,
        x=x,
        r_start=1,
        r_lower_bound=0.1,
        state_eq_constraints=None
    )
    non_empty_subsets_vectors,_ = step_two_check.all_non_empty_subsets_outside_ball()
    expected_non_empty_vectors = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0]
    ])
    assert np.array_equal(non_empty_subsets_vectors, expected_non_empty_vectors)


