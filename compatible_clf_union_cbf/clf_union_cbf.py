from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple
from typing_extensions import Self
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import (
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    check_array_of_polynomials,
)
from compatible_clf_union_cbf.union_cbf import (
    UnionCBF,
    UnionSubset,
    ActivationIndicator,
)

from compatible_clf_union_cbf.inclusion import (
    BallInclusionLagrangian,
    BallInclusionLagrangianDegree,
    BallInclusion,
    UnsafeExclusion,
    UnsafeRegionExclusionLagrangians,
    UnsafeRegionExclusionLagrangianDegrees,
    PointsInclusionConstriants,
)
from compatible_clf_union_cbf.utils import serialize_polynomial, deserialize_polynomial


"""
This module is defined for general verification of clf and union CBFs
"""
@dataclass
class SubsetVerifyLagrangian:
    """
    This class has the lagrangians for the compatibility verification
    inside a union subset. Here is what we wanted to do:
    given a non-empty subset of the union, verify that there exists
    an activated cbf in the subset that compatible with the clf.
    The verification SOS consists of the following lagrangians:
    1. lagragians for lambda_y of each activated cbf.
        This should be an array of array of polynomials, which means
        an ndarray with size of shape = 2. Shape[0] should be the number
        of activated cbfs. Shape[1] should be the number of control.
    2. lagragians for xi_y of each activated cbf.
        This should also be an array of polynomials with the shape =
        lambda_y.shape[0].
    3. lagrangians for activated cbfs in the subset.
        This should be an array of polynomials with the size equal to the
        number of activated cbfs.
    4. lagrangians for the clf.
        This should just be a polynomial
    5. lagrangians for all the deactivated cbfs.
        This should be an array of polynomials with the size equal to the
        number of deactivated cbfs.
    5. lagrangians for the ball.
        This should be a polynomial.
    6. lagrangians for state_eq_constraints (Optional)
        This should be an array of polynomials with the size equal to the
        number of state_eq_constraints.
    
    The following is a simple example of verification of compatible clf and
    union cbf inside a non-empty subset: 
    Consider subset{x|h1(x)≥0, h2(x)≥0, h3(x)<0, h4(x)<0, 1-V(x)≥0, r^2-xᵀx<0}
    we want to verify that for all x in the subset, there exists i∈{1,2} and
    u such that: Λᵢ(x)u ≤ ξᵢ(x). Using negation and Farkas' lemma, this is
    equivalent to verify that the following set is empty:
    {(x,y,c)| Λ₁(x)ᵀy₁²=0, Λ₂(x)ᵀy₂²=0, ξ₁(x)ᵀy₁²+1=0, ξ₂(x)ᵀy₂²+1=0,
              h1(x)≥0, h2(x)≥0, 1-V(x)≥0,
              c₁²h₃(x)+1=0, c₂²h₄(x)+1=0, c₃²(r²-xᵀx)+1=0}
    where y = (y₁, y₂) and c = (c₁, c₂, c₃)

    Transformed into SOS program, we have:
    -1 - s1(x,y2,c1,c2,c3)ᵀΛ₁(x)ᵀy₁² - s2(x,y1,c1,c2,c3)ᵀΛ₂(x)ᵀy₂²
       - s3(x,y2,c1,c2,c3)(ξ₁(x)ᵀy₁²+1) - s4(x,y1,c1,c2,c3)(ξ₂(x)ᵀy₂²+1)
       - s5(x,y1,y2,c1,c2,c3)h1(x) - s6(x,y1,y2,c1,c2,c3)h2(x)
       - s7(x,y1,y2,c1,c2,c3)(1-V(x))
       - s8(x,y1,y2,c2,c3)(c₁²h₃(x)+1) - s9(x,y1,y2,c1,c3)(c₂²h₄(x)+1)
       - s10(x,y1,y2,c1,c2)(c₃²(r²-xᵀx)+1) is SOS
    where s5,s6,s7 are SOS, others are free polynomials.
    """

    lambda_y: np.ndarray
    xi_y: np.ndarray
    activated_cbfs: np.ndarray
    rho_minus_V: sym.Polynomial
    deactivated_cbfs: Optional[np.ndarray]
    ball: sym.Polynomial
    state_eq_constraints: Optional[np.ndarray]

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        lambda_y_result = get_polynomial_result(result, self.lambda_y, coefficient_tol)
        xi_y_result = get_polynomial_result(result, self.xi_y, coefficient_tol)
        activated_cbfs_result = get_polynomial_result(
            result, self.activated_cbfs, coefficient_tol
        )
        rho_minus_V_result = get_polynomial_result(result, self.clf, coefficient_tol)
        deactivated_cbfs_result = (
            get_polynomial_result(result, self.deactivated_cbfs, coefficient_tol)
            if self.deactivated_cbfs is not None
            else None
        )
        ball_result = get_polynomial_result(result, self.ball, coefficient_tol)
        state_eq_constraints_result = (
            get_polynomial_result(result, self.state_eq_constraints, coefficient_tol)
            if self.state_eq_constraints is not None
            else None
        )
        return SubsetVerifyLagrangian(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            activated_cbfs=activated_cbfs_result,
            rho_minus_V=rho_minus_V_result,
            deactivated_cbfs=deactivated_cbfs_result,
            ball=ball_result,
            state_eq_constraints=state_eq_constraints_result,
        )

@dataclass
class SubsetVerifyLagrangianDegree:
    """
    This class specifies the lagrangian degrees. We have talked
    about the lagrngian components in last class above. This class
    should also follows the array size as the previous class. The
    only difference is that here we are defining array of Degree
    objects instead of array of polynomials.
    """

    lambda_y: List[List[Degree]]
    xi_y: List[Degree]
    activated_cbfs: List[Degree]
    rho_minus_V: Degree
    deactivated_cbfs: Optional[List[Degree]]
    ball: Degree
    state_eq_constraints: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        y_sets: np.ndarray,
        c_sets: np.ndarray,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_lambda_y: Optional[np.ndarray] = None,
        lagrangian_xi_y: Optional[np.ndarray] = None,
        lagrangian_activated_cbfs: Optional[np.ndarray] = None,
        lagrangian_rho_minus_V: Optional[sym.Polynomial] = None,
        lagrangian_deactivated_cbfs: Optional[np.ndarray] = None,
        lagrangian_ball: Optional[sym.Polynomial] = None,
        lagrangian_state_eq_constraints: Optional[np.ndarray] = None,
    ) -> SubsetVerifyLagrangian:
        """
        we may use different
        y and c variables in different lagrangians. For example,
        the lagrangians for the lambda_y and xi_y of cbf h_a1(x) will
        not use y1, the lagrangians for deactivated cbf h_d1 will not
        use c1. Hence, if we have activated cbfs h_a1, h_a2, ..., h_ap
        and deactivated cbfs h_d1, h_d2, ..., h_dq, the set of y and c
        for the final SOS program should be:
        y = [y1, y2, ..., yp]
        c = [c1, c2, ..., cq]
        but the lagrangian for each deactivated cbf will use different
        subset of c, and the lagrangians for each activated cbf's lambda_y
        and xi_y will use different subset of y.

        In the args above:
        y_sets is an array of sym.variables. which is:
            y_full = [ya1, ya2, ..., yap, y_all]
            where y_all = [y1, y2, ..., yp] and each
            yai = [y1, y2, ...,yi-1, yi+1, ..., yp]
        c_sets is an array of sym.variables. which is:
            c_full = [cd1, cd2, ..., cdq, c_ball, c_all]
            where c_all = [c1, c2, ..., cq, c_ball] and each
            cdi = [c1, c2, ...,ci-1, ci+1, ..., cq, c_ball]
            c_ball = [c1, c2, c3, ..., cq]
        """
        assert len(self.lambda_y) == len(self.activated_cbfs)
        assert len(self.lambda_y) == len(self.xi_y)
        assert y_sets.shape[0] == len(self.lambda_y) + 1
        if self.deactivated_cbfs is not None:
            assert c_sets.shape[0] == len(self.deactivated_cbfs) + 2
        else:
            assert c_sets.shape[0] == 2

        lambda_y_lagrangians = np.empty(
            shape=(len(self.lambda_y), len(self.lambda_y[0])), dtype=object
        )
        if lagrangian_lambda_y is not None:
            assert lagrangian_lambda_y.shape == self.lambda_y.shape
        for i in range(len(self.lambda_y)):
            lambda_y_lagrangians[i] = to_lagrangian_impl(
                prog,
                x,
                y=y_sets[i],
                c=c_sets[-1],
                sos_type=sos_type,
                is_sos=False,
                degree=self.lambda_y[i],
                lagrangian=(
                    lagrangian_lambda_y[i] if lagrangian_lambda_y is not None else None
                ),
            )

        if lagrangian_xi_y is not None:
            assert lagrangian_xi_y.shape[0] == len(self.xi_y)
        xi_y_lagrangians = np.empty(shape=(len(self.xi_y),), dtype=object)
        for i in range(len(self.xi_y)):
            xi_y_lagrangians[i] = to_lagrangian_impl(
                prog,
                x,
                y=y_sets[i],
                c=c_sets[-1],
                sos_type=sos_type,
                is_sos=False,
                degree=self.xi_y[i],
                lagrangian=(
                    lagrangian_xi_y[i] if lagrangian_xi_y is not None else None
                ),
            )

        if lagrangian_activated_cbfs is not None:
            assert lagrangian_activated_cbfs.shape[0] == len(self.activated_cbfs)
        activated_cbfs_lagrangians = np.empty(
            shape=(len(self.activated_cbfs),), dtype=object
        )
        for i in range(len(self.activated_cbfs)):
            activated_cbfs_lagrangians[i] = to_lagrangian_impl(
                prog,
                x,
                y=y_sets[-1],
                c=c_sets[-1],
                sos_type=sos_type,
                is_sos=True,
                degree=self.activated_cbfs[i],
                lagrangian=(
                    lagrangian_activated_cbfs[i]
                    if lagrangian_activated_cbfs is not None
                    else None
                ),
            )

        rho_minus_V_lagrangian = to_lagrangian_impl(
            prog,
            x,
            y=y_sets[-1],
            c=c_sets[-1],
            sos_type=sos_type,
            is_sos=True,
            degree=self.clf,
            lagrangian=lagrangian_rho_minus_V,
        )

        if self.deactivated_cbfs is not None:
            if lagrangian_deactivated_cbfs is not None:
                assert lagrangian_deactivated_cbfs.shape[0] == len(
                    self.deactivated_cbfs
                )
            deactivated_cbfs_lagrangians = np.empty(
                shape=len(
                    self.deactivated_cbfs,
                ),
                dtype=object,
            )
            for i in range(len(self.deactivated_cbfs)):
                deactivated_cbfs_lagrangians[i] = to_lagrangian_impl(
                    prog,
                    x,
                    y=y_sets[-1],
                    c=c_sets[i],
                    sos_type=sos_type,
                    is_sos=False,
                    degree=self.deactivated_cbfs[i],
                    lagrangian=(
                        lagrangian_deactivated_cbfs[i]
                        if lagrangian_deactivated_cbfs is not None
                        else None
                    ),
                )
        else:
            deactivated_cbfs_lagrangians = None

        ball_lagrangian = to_lagrangian_impl(
            prog,
            x,
            y=y_sets[-1],
            c=c_sets[-2],
            sos_type=sos_type,
            is_sos=False,
            degree=self.ball,
            lagrangian=lagrangian_ball,
        )

        state_eq_constraints_lagrangians = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y_sets[-1],
            c=c_sets[-1],
            sos_type=sos_type,
            is_sos=False,
            degree=self.state_eq_constraints,
            lagrangian=lagrangian_state_eq_constraints,
        )

        return SubsetVerifyLagrangian(
            lambda_y=lambda_y_lagrangians,
            xi_y=xi_y_lagrangians,
            activated_cbfs=activated_cbfs_lagrangians,
            clf=rho_minus_V_lagrangian,
            deactivated_cbfs=deactivated_cbfs_lagrangians,
            ball=ball_lagrangian,
            state_eq_constraints=state_eq_constraints_lagrangians,
        )


"""
The following module is defined for the simplified verification of
compatible clf and union cbfs. 
"""
@dataclass
class CompatibleInRangeLagrangain:
    """
    This class is defined for the lagrangian of the compatibility of clf V and
    a single cbf h_i(x) in the range:
    {x|h1(x)≤0, h2(x)≤0,..., h(i-1)(x)≤0, hi(x)≥0, 1-V(x)≥0}
    """
    lambda_y: np.ndarray
    xi_y: np.ndarray
    rho_minus_V: sym.Polynomial
    cbf_i: sym.Polynomial
    deactivated_cbfs: Optional[np.ndarray] # this is None if i=1
    state_eq_constraints: Optional[np.ndarray]

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        lambda_y_result = get_polynomial_result(result, self.lambda_y, coefficient_tol)
        xi_y_result = get_polynomial_result(result, self.xi_y, coefficient_tol)
        cbf_i_result = get_polynomial_result(
            result, self.cbf_i, coefficient_tol
        )
        rho_minus_V_result = get_polynomial_result(result, self.rho_minus_V, coefficient_tol)
        deactivated_cbfs_result = (
            get_polynomial_result(result, self.deactivated_cbfs, coefficient_tol)
            if self.deactivated_cbfs is not None
            else None
        )
        state_eq_constraints_result = (
            get_polynomial_result(result, self.state_eq_constraints, coefficient_tol)
            if self.state_eq_constraints is not None
            else None
        )
        return CompatibleInRangeLagrangain(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            cbf_i=cbf_i_result,
            rho_minus_V=rho_minus_V_result,
            deactivated_cbfs=deactivated_cbfs_result,
            state_eq_constraints=state_eq_constraints_result,
        )

@dataclass
class CompatibleInRnageLagrangianDegree:
    lambda_y: List[Degree]
    xi_y: Degree
    rho_minus_V: Degree
    cbf_i: Degree
    deactivated_cbfs: Optional[List[Degree]]
    state_eq_constraints: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        y: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_lambda_y: Optional[np.ndarray] = None,
        lagrangian_xi_y: Optional[sym.Polynomial] = None,
        lagrangian_cbf_i: Optional[sym.Polynomial] = None,
        lagrangian_rho_minus_V: Optional[sym.Polynomial] = None,
        lagrangian_deactivated_cbfs: Optional[np.ndarray] = None,
        lagrangian_state_eq_constraints: Optional[np.ndarray] = None,
    ) -> CompatibleInRangeLagrangain:
        lambda_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y
        )
        xi_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.xi_y,
            lagrangian=lagrangian_xi_y
        )
        cbf_i_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.cbf_i,
            lagrangian=lagrangian_cbf_i
        )
        rho_minus_V_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.rho_minus_V,
            lagrangian=lagrangian_rho_minus_V
        )
        deactivated_cbfs_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.deactivated_cbfs,
            lagrangian=lagrangian_deactivated_cbfs
        )
        state_eq_constraints_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.state_eq_constraints,
            lagrangian=lagrangian_state_eq_constraints
        )

        return CompatibleInRangeLagrangain(
            lambda_y=lambda_y_lagrangian,
            xi_y=xi_y_lagrangian,
            cbf_i=cbf_i_lagrangian,
            rho_minus_V=rho_minus_V_lagrangian,
            deactivated_cbfs=deactivated_cbfs_lagrangian,
            state_eq_constraints=state_eq_constraints_lagrangian,
        )


"""
This class is defined for the verification and synthesis of compatible clf and
union cbf.
"""
class ComaptibleClfUnionCbf:
    def __init__(self):
        pass
    def general_verification(self):
        pass
    def simplified_verification(self):
        pass
    def synthesis(self):
        pass
    def _calc_xi_lambda():
        pass
    def _created_gerneral_verification_lagrangian_degrees():
        pass
    def _created_simplified_verification_lagrangian_degrees():
        pass
    def bilinear_laternation():
        pass
    
