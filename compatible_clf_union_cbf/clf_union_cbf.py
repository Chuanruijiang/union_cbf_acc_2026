from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union
from typing_extensions import Self
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import (
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    check_array_of_polynomials
)
from compatible_clf_union_cbf.union_cbf import(
    UnionCBF,
    UnionSubset,
    ActivationIndicator
)

from compatible_clf_union_cbf.inclusion import(
    BallInclusionLagrangian,
    BallInclusionLagrangianDegree,
    BallInclusion
)


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
    """
    lambda_y: np.ndarray
    xi_y: np.ndarray
    activated_cbfs: np.ndarray
    clf: sym.Polynomial
    deactivated_cbfs: Optional[np.ndarray]
    ball: sym.Polynomial
    state_eq_constraints: Optional[np.ndarray]

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float]
    )-> Self:
        lambda_y_result = get_polynomial_result(
            result, self.lambda_y, coefficient_tol
        )
        xi_y_result = get_polynomial_result(
            result, self.xi_y, coefficient_tol
        )
        activated_cbfs_result = get_polynomial_result(
            result, self.activated_cbfs, coefficient_tol
        )
        clf_result = get_polynomial_result(
            result, self.clf, coefficient_tol
        )
        deactivated_cbfs_result = (
            get_polynomial_result(
                result, self.deactivated_cbfs, coefficient_tol
            )
            if self.deactivated_cbfs is not None
            else None
        )
        ball_result = get_polynomial_result(
            result, self.ball, coefficient_tol
        )
        state_eq_constraints_result = (
            get_polynomial_result(
                result, self.state_eq_constraints, coefficient_tol
            )
            if self.state_eq_constraints is not None
            else None
        )
        return SubsetVerifyLagrangian(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            activated_cbfs=activated_cbfs_result,
            clf=clf_result,
            deactivated_cbfs=deactivated_cbfs_result,
            ball=ball_result,
            state_eq_constraints=state_eq_constraints_result
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
    clf: Degree
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
        lagrangian_clf: Optional[sym.Polynomial] = None,
        lagrangian_deactivated_cbfs: Optional[np.ndarray] = None,
        lagrangian_ball: Optional[sym.Polynomial] = None,
        lagrangian_state_eq_constraints: Optional[np.ndarray] = None
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
        y_full is an array of sym.variables. which is:
            y_full = [ya1, ya2, ..., yap, y_all]
            where y_all = [y1, y2, ..., yp] and each
            yai = [y1, y2, ...,yi-1, yi+1, ..., yp]
        c_full is an array of sym.variables. which is:
            c_full = [cd1, cd2, ..., cdq, c_ball, c_all]
            where c_all = [c1, c2, ..., cq, c_ball] and each
            cdi = [c1, c2, ...,ci-1, ci+1, ..., cq, c_ball]
            c_ball = [c1, c2, c3, ..., cq]
        """
        assert len(self.lambda_y) == len(self.xi_y)
        assert y_sets.shape[0] == len(self.lambda_y) + 1
        if self.deactivated_cbfs is not None:
            assert c_sets.shape[0] == len(self.deactivated_cbfs) + 2
        else:
            assert c_sets.shape[0] == 2
        
        lambda_y_lagrangians  = np.empty(
            shape=(len(self.lambda_y), len(self.lambda_y[0])), 
            dtype=object
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
                    lagrangian_lambda_y[i] 
                    if lagrangian_lambda_y is not None
                    else None)
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
                    lagrangian_xi_y[i] 
                    if lagrangian_xi_y is not None
                    else None)
            )
        
        if lagrangian_activated_cbfs is not None:
            assert lagrangian_activated_cbfs.shape[0] == len(self.activated_cbfs)
        activated_cbfs_lagrangians = np.empty(
            shape=(len(self.activated_cbfs),),
            dtype=object
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
                    else None)
            )

        clf_lagrangian = to_lagrangian_impl(
            prog,
            x,
            y=y_sets[-1],
            c=c_sets[-1],
            sos_type=sos_type,
            is_sos=True,
            degree=self.clf,
            lagrangian=lagrangian_clf
        )

        if self.deactivated_cbfs is not None:
            if lagrangian_deactivated_cbfs is not None:
                assert lagrangian_deactivated_cbfs.shape[0] == len(self.deactivated_cbfs)
            deactivated_cbfs_lagrangians = np.empty(
                shape=len(self.deactivated_cbfs,),
                dtype=object
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
                        else None)
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
            lagrangian=lagrangian_ball
        )
        
        if self.state_eq_constraints is not None:
            if lagrangian_state_eq_constraints is not None:
                assert (lagrangian_state_eq_constraints.shape
                        ) == self.state_eq_constraints.shape
            state_eq_constraints_lagrangians = np.empty(
                shape=(len(self.state_eq_constraints),),
                dtype=object
            )
            for i in range(len(self.state_eq_constraints)):
                state_eq_constraints_lagrangians[i] = to_lagrangian_impl(
                    prog,
                    x,
                    y=y_sets[-1],
                    c=c_sets[-1],
                    sos_type=sos_type,
                    is_sos=False,
                    degree=self.state_eq_constraints[i],
                    lagrangian=(
                        lagrangian_state_eq_constraints[i] 
                        if lagrangian_state_eq_constraints is not None
                        else None)
                )
        else:
            state_eq_constraints_lagrangians = None
        
        return SubsetVerifyLagrangian(
            lambda_y=lambda_y_lagrangians,
            xi_y=xi_y_lagrangians,
            activated_cbfs=activated_cbfs_lagrangians,
            clf=clf_lagrangian,
            deactivated_cbfs=deactivated_cbfs_lagrangians,
            ball=ball_lagrangian,
            state_eq_constraints=state_eq_constraints_lagrangians
        )



@dataclass
class CompatibilityInRangeLagragians:
    """
    We wanted to verify the following:
    for all x in p(x)>=0, q(x) > 0, a given CBF h(x)
    is compatible with a given CLF V(x). Namely:
    ∀ x ∈ p(x) ≥ 0, q(x) > 0, ∃ u ∈ U, such that:
    Λ(x)u<= ξ(x).

    Put in SOS form:
    -1 - s1(x,y,c)ᵀp(x) - s2(x,y,c)ᵀ(c²*q(x)-1) 
        - s3(x,y,c)ᵀΛ(x)ᵀy² - s4(x,y,c)(ξ(x)ᵀy²+1)
        - s5(x,y,c)ᵀstate_eq(x).
    """
    lambda_y: np.ndarray
    xi_y: sym.Polynomial
    range_non_negative: Optional[np.ndarray]
    range_strictly_positive: Optional[np.ndarray]
    state_eq_const: Optional[np.ndarray]

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficitent_tol: Optional[float]
    )-> Self:
        lambda_y_result = get_polynomial_result(
            result, self.lambda_y, coefficitent_tol
        )
        xi_y_result = get_polynomial_result(
            result, self.xi_y, coefficitent_tol
        )
        range_non_negative_result = (
            get_polynomial_result(
            result, self.range_non_negative, coefficitent_tol)
            if self.range_non_negative is not None
            else None
        )
        range_strictly_positive_result = (
            get_polynomial_result(
            result, self.range_strictly_positive, coefficitent_tol)
            if self.range_strictly_positive is not None
            else None
        )
        state_eq_const_result = (
            get_polynomial_result(
            result, self.state_eq_const, coefficitent_tol)
            if self.state_eq_const is not None
            else None
        )
        return CompatibilityInRangeLagragians(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            range_non_negative=range_non_negative_result,
            range_strictly_positive=range_strictly_positive_result,
            state_eq_const=state_eq_const_result
            )
    

@dataclass
class CompatibilityInRangeLagragianDegrees:
    lambda_y: List[Degree]
    xi_y: Degree
    range_non_negative: Optional[List[Degree]]
    range_strictly_positive: Optional[List[Degree]]
    state_eq_const: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        y: sym.Variables,
        c_sets: Optional[np.ndarray],
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_lambda_y: Optional[np.ndarray] = None,
        lagrangian_xi_y: Optional[sym.Polynomial] = None,
        lagrangian_range_non_negative: Optional[List[np.ndarray]] = None,
        lagrangian_range_strictly_positive: Optional[List[np.ndarray]] = None,
        lagrangian_state_eq_const: Optional[List[np.ndarray]] = None,
    ) -> CompatibilityInRangeLagragians:
        """
        If we have multiple strictly positive polynomials, we should use different
        c_variables for lagrangians in front of different positive polynomials. 
        Assume we have q1,...qm as strictly positive polynomials, then the c_set
        should be:
        c_set = [c_1, c_2, ..., c_m, c_all]
        which is an array of sym.Variables. Each c_i is an array of sym.Variables
        """
        assert isinstance(self.lambda_y, list)
        assert isinstance(self.xi_y, Degree)
        if self.range_non_negative is not None:
            assert isinstance(self.range_non_negative, list)
        if self.range_strictly_positive is not None:
            assert isinstance(self.range_strictly_positive, list)
            assert c_sets.shape[0] == len(self.range_strictly_positive) + 1
        else:
            assert c_sets is None
        if self.state_eq_const is not None:
            assert isinstance(self.state_eq_const, list)
        lambda_y_lagragian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=(c_sets[-1] if c_sets is not None else None),
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y
        )
        xi_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=(c_sets[-1] if c_sets is not None else None),
            sos_type=sos_type,
            is_sos=False,
            degree=self.xi_y,
            lagrangian=lagrangian_xi_y
        )
        range_non_negative_lagrangians = (
            to_lagrangian_impl(
                prog=prog,
                x=x,
                y=y,
                c=(c_sets[-1] if c_sets is not None else None),
                sos_type=sos_type,
                is_sos=True,
                degree=self.range_non_negative,
                lagrangian=lagrangian_range_non_negative
            )
            if self.range_non_negative is not None
            else None
        )
        range_strictly_positive_lagrangians = (
            np.array([
                to_lagrangian_impl(
                    prog=prog,
                    x=x,
                    y=y,
                    c=c_sets[i],
                    sos_type=sos_type,
                    is_sos=False,
                    degree=self.range_strictly_positive[i],
                    lagrangian=(
                        lagrangian_range_strictly_positive[i]
                        if lagrangian_range_strictly_positive is not None
                        else None)
                )
                for i in range(len(self.range_strictly_positive))
            ])
            if self.range_strictly_positive is not None
            else None
        )
        state_eq_const_lagrangians = (
            to_lagrangian_impl(
                prog=prog,
                x=x,
                y=y,
                c=(c_sets[-1] if c_sets is not None else None),
                sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
                is_sos=False,
                degree=self.state_eq_const,
                lagrangian=lagrangian_state_eq_const
            )
            if self.state_eq_const is not None
            else None
        )
        return CompatibilityInRangeLagragians(
            lambda_y=lambda_y_lagragian,
            xi_y=xi_y_lagrangian,
            range_non_negative=range_non_negative_lagrangians,
            range_strictly_positive=range_strictly_positive_lagrangians,
            state_eq_const=state_eq_const_lagrangians
        )


class CompatibleClfCbfInRange:
    """
    This class is used to verifiy the compatibility of a given
    CLF V(x) and CBF h(x) in a given range.
    the range is defined by the following:
    {x | p(x) >= 0, q(x) > 0, state_eq(x) = 0}
    where p(x) is a group of polynomials that are non-negative
    q(x) is a group of polynomials that are strictly positive
    state_eq(x) is a group of polynomials that are equal to zero
    However this class does not verify that the set {x | h(x) >= 0}
    is a subset in the range above. The user should verify this
    before using this class.
    """
    def __init__(
        self,
        x: np.ndarray,
        sys_dyn_f: np.ndarray,
        sys_dyn_g: np.ndarray,
        V: sym.Polynomial,
        h: sym.Polynomial,
        range_non_negative: Optional[np.ndarray],
        range_strictly_positive: Optional[np.ndarray],
        state_eq_const: Optional[np.ndarray],
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
    ):
        self.sys_dyn_f = sys_dyn_f
        self.sys_dyn_g = sys_dyn_g
        self.V = V
        self.h = h
        if range_non_negative is None and range_strictly_positive is None:
            raise ValueError(
                "range_non_negative and range_strictly_positive are None.")
        self.range_non_negative = range_non_negative
        self.range_strictly_positive = range_strictly_positive
        self.state_eq_const = state_eq_const
        assert (Au is None) == (bu is None)
        if Au is not None:
            assert Au.shape[1] == sys_dyn_g.shape[1]
            assert Au.shape[0] == bu.shape[0]
        self.Au = Au
        self.bu = bu

        # creating necessary symbolic variables:
        self.x = x
        self.y = (
            sym.MakeVectorContinuousVariable(2, "y")
            if Au is None
            else sym.MakeVectorContinuousVariable(2+Au.shape[0], "y")
            )
        self.c = (
            sym.MakeVectorContinuousVariable(range_strictly_positive.shape[0], "c")
            if range_strictly_positive is not None
            else None
        )
        self.y_squared_poly = np.array([
            sym.Polynomial(sym.Monomial(self.y[i], 2)) 
            for i in range(self.y.shape[0])
        ])
        self.c_squared_poly = (
            np.array([
                sym.Polynomial(sym.Monomial(self.c[i], 2))
                for i in range(self.c.shape[0])
            ])
            if range_strictly_positive is not None
            else None
        )
        self.x_set = sym.Variables(self.x)
        self.y_set = sym.Variables(self.y)
        
        #computing c_sets:
        if range_strictly_positive is not None:
            c_sets = np.empty(shape=(range_strictly_positive.shape[0]+1,), dtype=object)
            c_sets[-1] = sym.Variables(self.c)
            for i in range(range_strictly_positive.shape[0]):
                c_i = np.delete(self.c, i, axis=0)
                if c_i.size == 0:
                    pass
                else:
                    c_sets[i] = sym.Variables(c_i)
            self.c_sets = c_sets
        else:   
            self.c_sets = None

        self.xy_set = sym.Variables(np.concatenate([self.x, self.y], axis=0))
        self.xyc_set = (
            sym.Variables(np.concatenate([self.x, self.y, self.c], axis=0))
            if range_strictly_positive is not None
            else None
        )

        # check the polynomials:
        if range_non_negative is not None:
            check_array_of_polynomials(range_non_negative, self.x_set)
        if range_strictly_positive is not None:
            check_array_of_polynomials(range_strictly_positive, self.x_set)
        if state_eq_const is not None:
            check_array_of_polynomials(self.state_eq_const, self.x_set)
    
    def _calc_xi_lambda(
        self,
        *,
        kappa_V: float,
        kappa_h: float,
        epsilon: Optional[float]
    )-> Tuple[np.ndarray, np.ndarray]:
        """
        Compute
        Λ(x) = [-∂b/∂x*g(x)]
               [ ∂V/∂x*g(x)]
               [        Au ]
        ξ(x) = [ ∂b/∂x*f(x)+κ_b*b(x)]
               [-∂V/∂x*f(x)-κ_V*V(x)]
               [                 bu ]
        for just a single CBF and CLF.
        Args:
          V: The CLF function.
          h: The CBF function.
          kappa_V: The convergence rate for V(x) during evlove.
          kappa_b: The kappa function for cbf h(x).
          epsilon: The positive scalar for verification.
                    If this is not None, then the computed ξ(x) = ξ(x)-ϵ 
        Returns:
          (xi, lambda_mat) ξ(x) and Λ(x) above.
        """
        lambda_rows = 2 # a cbf + a clf
        if self.Au is not None:
            lambda_rows += self.Au.shape[0]
        lambda_cols = self.sys_dyn_g.shape[1]
        lambda_mat = np.empty((lambda_rows, lambda_cols), dtype=object)
        xi = np.empty((lambda_rows,), dtype=object)

        # loading CBF constriants:
        Lgh = lie_derivative(self.h, self.sys_dyn_g, self.x, 1)
        Lfh = lie_derivative(self.h, self.sys_dyn_f, self.x, 1)
        lambda_mat[0] = -Lgh
        xi[0] = Lfh + kappa_h*self.h

        # loading CLF constraints:
        LgV = lie_derivative(self.V, self.sys_dyn_g, self.x, 1)
        LfV = lie_derivative(self.V, self.sys_dyn_f, self.x, 1)
        lambda_mat[1] = LgV
        xi[1] = -LfV - kappa_V*self.V

        # loading Au and bu:
        if self.Au is not None:
            lambda_mat[2:] = self.Au
            xi[2:] = self.bu
        
        # if we have epsilon:
        if epsilon is not None:
            xi = xi - epsilon
        
        return xi, lambda_mat

    def _add_compatibility_constraint(
        self,
        prog: solvers.MathematicalProgram,
        xi: np.ndarray,
        lambda_mat: np.ndarray,
        lagrangians: CompatibilityInRangeLagragians,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> sym.Polynomial:
        """
        This function only adds the constraint to verify the compatibility
        in the given range. It does NOT include the constraints for checking
        whether the set {x | h(x) >= 0} is a subset of the given range.
        """
        poly_one = sym.Polynomial(1)
        poly = -poly_one
        
        # add S_0(x,y,c)*p(x)
        if self.range_non_negative is not None:
            poly -= lagrangians.range_non_negative.dot(self.range_non_negative)
        
        # if rang_strictly_positive is not none, add S_1(x,y,c)*q(x)
        if self.range_strictly_positive is not None:
            assert self.range_strictly_positive.shape[0] == self.c_squared_poly.shape[0]
            poly -= lagrangians.range_strictly_positive.dot(
                self.c_squared_poly * self.range_strictly_positive - 1)
        
        # add S_2(x,y,c)*Λ(x)ᵀ*y²
        assert lambda_mat.shape[0] == self.y_squared_poly.shape[0]
        poly -= lagrangians.lambda_y.dot(self.y_squared_poly.dot(lambda_mat))
        
        # add S_3(x,y,c)*(ξ(x)ᵀ*y²+1)
        poly -= lagrangians.xi_y * (self.y_squared_poly.dot(xi) + 1)
        
        # if we have state_eq_const:
        if self.state_eq_const is not None:
            assert lagrangians.state_eq_const is not None
            poly -= lagrangians.state_eq_const.dot(self.state_eq_const)
        
        prog.AddSosConstraint(poly, sos_type)
        return poly
    
    def construct_compatibility_sos_program(
        self,
        kappa_V: float,
        kappa_h: float,
        epsilon: Optional[float],
        lagrangian_degrees: CompatibilityInRangeLagragianDegrees,
        lagrangian_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Tuple[
        solvers.MathematicalProgram,
        CompatibilityInRangeLagragians
    ]:
        assert (
            self.range_non_negative is None
            ) == (lagrangian_degrees.range_non_negative is None)
        assert (
            self.range_strictly_positive is None
            ) == (lagrangian_degrees.range_strictly_positive is None)
        prog = solvers.MathematicalProgram()
        if lagrangian_degrees.range_strictly_positive is None:
            prog.AddIndeterminates(self.xy_set)
        else:
            prog.AddIndeterminates(self.xyc_set)
        lagrangians = lagrangian_degrees.to_lagrangians(
            prog=prog,
            x=self.x_set,
            y=self.y_set,
            c_sets=self.c_sets,
            sos_type=lagrangian_sos_type
        )
        xi, lambda_mat = self._calc_xi_lambda(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=epsilon
        )
        self._add_compatibility_constraint(
            prog=prog,
            xi=xi,
            lambda_mat=lambda_mat,
            lagrangians=lagrangians,
            sos_type=compatible_sos_type
        )
        return (prog, lagrangians)

    def verify_compatibility(
        self,
        kappa_V: float,
        kappa_h: float,
        epsilon: Optional[float],
        lagrangian_degrees: CompatibilityInRangeLagragianDegrees,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
        lagrangian_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Optional[CompatibilityInRangeLagragians]:

        prog, lagrangians = self.construct_compatibility_sos_program(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=epsilon,
            lagrangian_degrees=lagrangian_degrees,
            lagrangian_sos_type=lagrangian_sos_type,
            compatible_sos_type=compatible_sos_type
        )
        result = solve_with_id(prog, solver_id, solver_options)
        lagrangians_result = (
            lagrangians.get_results(result, lagrangian_coefficient_tol)
            if result.is_success()
            else None
            )
        return lagrangians_result


class CompatibleClfCbfInSubset:
    def __init__(
        self,
        x: np.ndarray,
        sys_dyn_f: np.ndarray,
        sys_dyn_g: np.ndarray,
        subset: UnionSubset,
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
        state_eq_constraints: Optional[np.ndarray]
    ):
        self.sys_dyn_f = sys_dyn_f
        self.sys_dyn_g = sys_dyn_g
        self.subset = subset
        assert (Au is None) == (bu is None)
        with_control_limits = False
        control_limit_size = None
        if Au is not None:
            assert Au.shape[1] == sys_dyn_g.shape[1]
            assert Au.shape[0] == bu.shape[0]
            with_control_limits = True
            control_limit_size = Au.shape[0]
        self.Au = Au
        self.bu = bu
        self.state_eq_constraints = state_eq_constraints

        # creating necessary symbolic variables:
        self.x = x
        self.x_set = sym.Variables(self.x)
        (
            y_sets, c_sets, y_all, c_all
        ) = self.subset.yc_sets_for_verification_wclf_outball(
            with_control_input_limits=with_control_limits,
            control_constriant_size=control_limit_size
        )
        self.y_all = y_all
        self.c_all = c_all
        self.xyc = np.concatenate([self.x, self.y_all.flatten(), self.c_all], axis=0)
        self.xyc_set = sym.Variables(self.xyc)
        self.y_sets = y_sets
        self.c_sets = c_sets
    
    def _add_compatible_in_subset_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        xis: List[np.ndarray],
        lambdas: List[np.ndarray],
        lagrangians: SubsetVerifyLagrangian,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> sym.Polynomial:
        poly_one = sym.Polynomial(1)
        poly = -poly_one

        # add s0(x,y,c)*lambda_y - s1(x,y,c)*xi_y
        for i in range(len(lambdas)):
            y_i_sequare_poly = np.array(
                [sym.Polynomial(sym.Monomial(each_y, 2))
                 for each_y in self.y_all[i]]
                 )
            poly -= lagrangians.lambda_y[i].dot(y_i_sequare_poly.dot(lambdas[i]))
            poly -= lagrangians.xi_y[i] * (y_i_sequare_poly.dot(xis[i]) + 1)
        
        # add s2(x,y,c)*activated_cbfs
        for i in range(lagrangians.activated_cbfs.shape[0]):
            poly -= lagrangians.activated_cbfs[i] * self.subset.activated[1+i]
        # add s3(x,y,c)*clf
        poly -= lagrangians.clf * self.subset.activated[0]
        # add s4(x,y,c)*deactivated_cbfs
        if lagrangians.deactivated_cbfs is not None:
            for i in range(lagrangians.deactivated_cbfs.shape[0]):
                poly -= lagrangians.deactivated_cbfs[i] * self.subset.deactivated[i]
        # add s5(x,y,c)*ball
        poly -= lagrangians.ball * self.subset.deactivated[-1]
        # if we have state_eq_constraints:
        if self.state_eq_constraints is not None:
            for i in range(lagrangians.state_eq_constraints.shape[0]):
                poly -= lagrangians.state_eq_constraints[i] * self.state_eq_constraints[i]

        prog.AddSosConstraint(poly, sos_type)
        return poly
    
    def _calc_xi_lambda(
        self,
        kappa_V: float,
        kappa_h: List[float],
        epsilon: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute
        Λ(x) = [-∂b/∂x*g(x)]
               [ ∂V/∂x*g(x)]
               [        Au ]
        ξ(x) = [ ∂b/∂x*f(x)+κ_b*b(x) - ϵ]
               [-∂V/∂x*f(x)-κ_V*V(x) - ϵ]
               [                 bu  - ϵ]
        for all the activated CBFs and the CLF.
        Different from the CompatibleClfCbfInRange, this function will
        compute the ξ(x) and Λ(x) for all the activated CBFs and the CLF
        one by one and then pack them in an array. Hence, the return
        should be array of ξ(x)s and array of Λ(x)s for each activated
        cbf.
        Args:
          V: The CLF function.
          h: The CBF function.
          kappa_V: The convergence rate for V(x) during evlove.
          kappa_b: The kappa function for cbf h(x).
          epsilon: The positive scalar for verification.
                    If this is not None, then the computed ξ(x) = ξ(x)-ϵ 
        Returns:
          arrays of ξ(x)s and Λ(x)s above.
        """
        # the size of array of lambda and xi should be the number of
        # activated cbfs (without the clf)
        # Since rho-V is also counted as an activated poly in a subset
        # object, then we should use the subset.activated.shape[0] - 1
        # Also, during the iterations, since the first activated poly
        # is the clf, we should start from the second activated poly in
        # the subset.activated.
        array_of_lambda_mat = []
        array_of_xi_vec = []
        for i in range(1, self.subset.activated.shape[0]):
            lambda_rows = 2
            if self.Au is not None:
                lambda_rows += self.Au.shape[0]
            lambda_cols = self.sys_dyn_g.shape[1]
            lambda_mat = np.empty((lambda_rows, lambda_cols), dtype=object)
            xi = np.empty((lambda_rows,), dtype=object)

            # loading CBF constriants:
            Lgh = lie_derivative(self.subset.activated[i], self.sys_dyn_g, self.x, 1)
            Lfh = lie_derivative(self.subset.activated[i], self.sys_dyn_f, self.x, 1)
            lambda_mat[0] = -Lgh
            xi[0] = Lfh + kappa_h[i-1]*self.subset.activated[i]

            # loading CLF constraints:
            # noted that the first activated poly is not the original clf
            # but the rho-V, so we should use -self.subset.activated[0] to 
            # compute the CLF contents in lambda and xi.
            LgV = lie_derivative(-self.subset.activated[0], self.sys_dyn_g, self.x, 1)
            LfV = lie_derivative(-self.subset.activated[0], self.sys_dyn_f, self.x, 1)
            lambda_mat[1] = LgV
            xi[1] = -LfV - kappa_V*self.subset.activated[0]

            # loading Au and bu:
            if self.Au is not None:
                lambda_mat[2:] = self.Au
                xi[2:] = self.bu

            # minus epsilon:
            xi = xi - epsilon
            
            array_of_lambda_mat.append(lambda_mat)
            array_of_xi_vec.append(xi)
        
        return array_of_xi_vec, array_of_lambda_mat

    def construct_compatibility_in_subset_program(
        self,
        kappa_V: float,
        kappa_h: List[float],
        epsilon: float,
        lagrangian_degrees: SubsetVerifyLagrangianDegree,
        lagragians_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> Tuple[
        solvers.MathematicalProgram,
        SubsetVerifyLagrangian
    ]:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xyc_set)
        lagrangians = lagrangian_degrees.to_lagrangians(
            prog=prog,
            x=self.x_set,
            y_sets=self.y_sets,
            c_sets=self.c_sets,
            sos_type=lagragians_sos_type
        )
        xis, lambdas = self._calc_xi_lambda(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=epsilon
        )
        self._add_compatible_in_subset_lagrangians(
            prog=prog,
            xis=xis,
            lambdas=lambdas,
            lagrangians=lagrangians,
            sos_type=compatible_sos_type
        )
        return (prog, lagrangians)

    def verify_compatibility_in_subset(
        self,
        kappa_V: float,
        kappa_h: List[float],
        epsilon: float,
        lagrangian_degrees: SubsetVerifyLagrangianDegree,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coefficient_tol: Optional[float] = None,
        lagrangian_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        compatible_sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> Optional[SubsetVerifyLagrangian]:
        prog, lagrangians = self.construct_compatibility_in_subset_program(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=epsilon,
            lagrangian_degrees=lagrangian_degrees,
            lagragians_sos_type=lagrangian_sos_type,
            compatible_sos_type=compatible_sos_type
        )
        result = solve_with_id(prog, solver_id, solver_options)
        lagrangians_result = (
            lagrangians.get_results(result, lagrangian_coefficient_tol)
            if result.is_success()
            else None
            )
        return lagrangians_result
        

@dataclass
class StepOne:
    """
    The clf here is just V(x)
    """
    clf: sym.Polynomial
    cbfs: np.ndarray
    sys_dyn_f: np.ndarray
    sys_dyn_g: np.ndarray
    x: np.ndarray
    r_start: float
    r_lower_bound: float
    state_eq_constraints: Optional[np.ndarray]
    Au: Optional[np.ndarray]
    bu: Optional[np.ndarray]

    def _create_ball_inclusion_lagrangian_degrees(
        self,
        ball_x_degree: List[int],
        cbf_x_degree: List[int],
    ) -> List[BallInclusionLagrangianDegree]:
        assert len(ball_x_degree) == self.cbfs.shape[0]
        assert len(cbf_x_degree) == self.cbfs.shape[0]
        ball_inclusion_lagrangian_degrees = []
        for i in range(self.cbfs.shape[0]):
            ball_inclusion_lagrangian_degrees.append(
                BallInclusionLagrangianDegree(
                    r_minus_xTx=Degree(x=ball_x_degree[i], y=0, c=0),
                    h=Degree(x=cbf_x_degree[i], y=0, c=0)
                )
            )
        return ball_inclusion_lagrangian_degrees

    def _create_qp_feasible_in_ball_lagrangian_degrees(
        self,
        lambda_y_x_degrees: List[int],
        xi_y_x_degrees: List[int],
        ball_x_degree: List[int],
        state_eq_x_degrees: Optional[List[int]]
    ) -> List[CompatibilityInRangeLagragianDegrees]:
        assert len(lambda_y_x_degrees) == self.cbfs.shape[0]
        assert len(xi_y_x_degrees) == self.cbfs.shape[0]
        assert len(ball_x_degree) == self.cbfs.shape[0]
        if self.state_eq_constraints is not None:
            assert len(state_eq_x_degrees) == self.state_eq_constraints.shape[0]
        
        num_of_control = self.sys_dyn_g.shape[1]
        qp_feasible_in_ball_lagrangian_degrees = []
        for i in range(self.cbfs.shape[0]):
            lambda_y = [
                Degree(x=lambda_y_x_degrees[i], y=0, c=0)
                for _ in range(num_of_control)
            ]
            xi_y = Degree(x=xi_y_x_degrees[i], y=0, c=0)
            ball = [Degree(x=ball_x_degree[i], y=2, c=0)]
            state_eq = None
            if self.state_eq_constraints is not None:
                state_eq = [
                    Degree(x=state_eq_x_degrees[j], y=2, c=0)
                    for j in range(self.state_eq_constraints.shape[0])
                ]
            qp_feasible_in_ball_lagrangian_degrees.append(
                CompatibilityInRangeLagragianDegrees(
                    lambda_y=lambda_y,
                    xi_y=xi_y,
                    range_non_negative=ball,
                    range_strictly_positive=None,
                    state_eq_const=state_eq
                )
            )
        return qp_feasible_in_ball_lagrangian_degrees

    def _qp_feasible_in_ball(
        self,
        r: float,
        cbf: sym.Polynomial,
        kappa_V: float,
        kappa_h: float,
        lagrangian_degree: CompatibilityInRangeLagragianDegrees
    ) -> bool:
        ball = sym.Polynomial(r - self.x.dot(self.x))
        compatibility_obj = CompatibleClfCbfInRange(
            x=self.x,
            sys_dyn_f=self.sys_dyn_f,
            sys_dyn_g=self.sys_dyn_g,
            V=self.clf,
            h=cbf,
            range_non_negative=np.array([ball]),
            range_strictly_positive=None,
            state_eq_const=self.state_eq_constraints,
            Au=self.Au,
            bu=self.bu
        )
        lagrangians = compatibility_obj.verify_compatibility(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=0,
            lagrangian_degrees=lagrangian_degree
        )
        return lagrangians is not None

    def _ball_included_by_cbf(
        self,
        r: float,
        cbf: sym.Polynomial,
        lagrangian_degree: BallInclusionLagrangianDegree
    ) -> bool:
        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.x)
        prog.AddIndeterminates(x_set)
        lagrangian = lagrangian_degree.to_lagrangians(prog, x_set)
        prog.AddSosConstraint(sym.Polynomial(
            -1 - lagrangian.r_minus_xTx*(r**2 - self.x.dot(self.x)) + lagrangian.h*cbf
            )
        )
        result = solve_with_id(prog)
        return result.is_success()

    def step_one_verification(
        self,
        kappa_V: float,
        kappa_h: List[float],
        ball_inclusion_ball_x_degrees: List[int],
        ball_inclusion_cbf_x_degrees: List[int],
        qp_feasible_in_ball_lambda_y_x_degrees: List[int],
        qp_feasible_in_ball_xi_y_x_degrees: List[int],
        qp_feasible_in_ball_ball_x_degrees: List[int],
        qp_feasible_in_ball_state_eq_x_degrees: Optional[List[List[int]]]
    ) -> Optional[float]:
        assert len(kappa_h) == self.cbfs.shape[0]
        assert len(ball_inclusion_ball_x_degrees) == self.cbfs.shape[0]
        assert len(ball_inclusion_cbf_x_degrees) == self.cbfs.shape[0]
        assert len(qp_feasible_in_ball_lambda_y_x_degrees) == self.cbfs.shape[0]
        assert len(qp_feasible_in_ball_xi_y_x_degrees) == self.cbfs.shape[0]
        assert len(qp_feasible_in_ball_ball_x_degrees) == self.cbfs.shape[0]
        if self.state_eq_constraints is not None:
            assert len(qp_feasible_in_ball_state_eq_x_degrees) == self.cbfs.shape[0]

        (ball_inclusion_lagrangian_degrees
        ) = self._create_ball_inclusion_lagrangian_degrees(
            ball_x_degree=ball_inclusion_ball_x_degrees,
            cbf_x_degree=ball_inclusion_cbf_x_degrees
        )
        (qp_feasible_in_ball_lagrangian_degrees
        ) = self._create_qp_feasible_in_ball_lagrangian_degrees(
            lambda_y_x_degrees=qp_feasible_in_ball_lambda_y_x_degrees,
            xi_y_x_degrees=qp_feasible_in_ball_xi_y_x_degrees,
            ball_x_degree=qp_feasible_in_ball_ball_x_degrees,
            state_eq_x_degrees=qp_feasible_in_ball_state_eq_x_degrees
        )
        
        current_r = self.r_start
        while current_r >= self.r_lower_bound:
            for i in range(self.cbfs.shape[0]):
                inclusion = self._ball_included_by_cbf(
                    r=current_r,
                    cbf=self.cbfs[i],
                    lagrangian_degree=ball_inclusion_lagrangian_degrees[i]
                )
                if inclusion:
                    compatible = self._qp_feasible_in_ball(
                        r=current_r,
                        cbf=self.cbfs[i],
                        kappa_V=kappa_V,
                        kappa_h=kappa_h[i],
                        lagrangian_degree=qp_feasible_in_ball_lagrangian_degrees[i]
                    )
                    if compatible:
                        return current_r
            # if we can get here, this means the for loop has finished and the current r ball
            # is either not included by any cbf or not clf-cbf is not compatible in it.
            current_r = current_r / 2
        # if the while loop finishes, this means we cannot find a valid r, meaning that we
        # cannot find a valid ball that is included by a cbf and the clf-cbf is compatible in it.
        return None

    def simplififed_step_one_verification(
        self,
        kappa_V: float,
        kappa_h: float,
        ball_inclusion_ball_x_degree: int,
        ball_inclusion_cbf_x_degree: int,
        qp_feasible_in_ball_lambda_y_x_degree: int,
        qp_feasible_in_ball_xi_y_x_degree: int,
        qp_feasible_in_ball_ball_x_degree: int,
        qp_feasible_in_ball_state_eq_x_degree: Optional[List[int]]
    ) -> Optional[float]:
        """
        For the simplified verification, the first step is to verify the
        inclusion of a ball with radius r in the first cbf h_1(x). Hence,
        in this function, the kappa_h and lagragian degrees are all about
        the first single CBF.
        """
        assert isinstance(kappa_h, float)
        assert isinstance(ball_inclusion_ball_x_degree, int)
        assert isinstance(ball_inclusion_cbf_x_degree, int)
        assert isinstance(qp_feasible_in_ball_lambda_y_x_degree, int)
        assert isinstance(qp_feasible_in_ball_xi_y_x_degree, int)
        assert isinstance(qp_feasible_in_ball_ball_x_degree, int)
        if (self.state_eq_constraints is not None
            ) and (qp_feasible_in_ball_state_eq_x_degree is not None):
            assert len(qp_feasible_in_ball_state_eq_x_degree
                ) == self.state_eq_constraints.shape[0]
        
        current_r = self.r_start
        while current_r >= self.r_lower_bound:
            ball_inclusion_lagrangian_degrees = BallInclusionLagrangianDegree(
                r_minus_xTx=Degree(x=ball_inclusion_ball_x_degree, y=0, c=0),
                h=Degree(x=ball_inclusion_cbf_x_degree, y=0, c=0)
            )
            inclusion = self._ball_included_by_cbf(
                r=current_r,
                cbf=self.cbfs[0],
                lagrangian_degree=ball_inclusion_lagrangian_degrees
                )
            if inclusion:
                qp_feasible_in_ball_lagrangian_degrees = CompatibilityInRangeLagragianDegrees(
                    lambda_y=[
                        Degree(x=qp_feasible_in_ball_lambda_y_x_degree, y=0, c=0)
                        for _ in range(self.sys_dyn_g.shape[1])
                    ],
                    xi_y=Degree(x=qp_feasible_in_ball_xi_y_x_degree, y=0, c=0),
                    range_non_negative=[Degree(x=qp_feasible_in_ball_ball_x_degree, y=2, c=0)],
                    range_strictly_positive=None,
                    state_eq_const=None
                )
                compatible = self._qp_feasible_in_ball(
                    r=current_r,
                    cbf=self.cbfs[0],
                    kappa_V=kappa_V,
                    kappa_h=kappa_h,
                    lagrangian_degree=qp_feasible_in_ball_lagrangian_degrees
                )
                if compatible:
                    return current_r
            current_r = current_r / 2
        return None 


@ dataclass
class StepTwo:
    """
    clf is ρ - V(x), 
    ball is eps0^2 - xᵀx,
    """
    clf: sym.Polynomial
    cbfs: np.ndarray
    ball: sym.Polynomial
    sys_dyn_f: np.ndarray
    sys_dyn_g: np.ndarray
    x: np.ndarray
    r_start: float
    r_lower_bound: float
    state_eq_constraints: Optional[np.ndarray]
    Au: Optional[np.ndarray]
    bu: Optional[np.ndarray]

    def all_non_empty_subsets_outside_ball(
        self,
        emptiness_lagragian_x_degrees: Optional[np.ndarray] = None,
        emptiness_lagragian_common_x_degree: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function returns all the non-empty subsets of the union.
        The subsets all have clf as the activated polynomial and the ball
        as the deactivated polynomial.
        The return of this function is an array of UnionSubset objects.
        """
        if emptiness_lagragian_x_degrees is not None:
            assert emptiness_lagragian_x_degrees.shape[0] == (
                self.cbfs.shape[0] + 2
            )
        union_of_cbfs = UnionCBF(
            h = self.cbfs,
            x = self.x
        )
        non_empty_subset_01_vector = union_of_cbfs.non_empty_disjoint_subsets(
            outside_ball=True,
            ball_poly=self.ball,
            with_clf=True,
            clf=self.clf,
            lagragian_x_degrees=emptiness_lagragian_x_degrees,
            common_x_degree=emptiness_lagragian_common_x_degree
        )
        activation_indicator = ActivationIndicator(
            vecotor01=non_empty_subset_01_vector
        )
        non_empty_subsets = activation_indicator.to_union_subsets(
            polys=np.concatenate(
                [np.array([self.clf]), self.cbfs, np.array([self.ball])], 
                axis=0
            ),
            variables=self.x
        )
        return non_empty_subset_01_vector, non_empty_subsets

    def _create_subset_verification_lagragian_degrees(
        self,
        subset: UnionSubset,
        x_degree: int,
        y_degree: int,
        c_degree: int
    ) -> SubsetVerifyLagrangianDegree:
        """
        For different subsets, the activated and deactivated polynomials are different.
        Hence, the lagragian degrees for verfication of each subset may be different.
        This function is used to create the lagragian degrees for a given subset.

        A union subset object has two components:
        activated_polynomials:
            The first polynomial is the ρ - V(x), the rest are activated cbfs.
        deactivated_polynomials:
            The last polynomial is the ball (r - xᵀx), the rest are deactivated cbfs.

        Noted that the subset object should present a non-empty subset. Hence this function
        should be called after the "all_non_empty_subsets_outside_ball()".
        """
        # we need to make sure that the subset has activated CBF, otherwise this subset
        # is in unsafe region.
        assert subset.activated.shape[0] > 1
        assert subset.deactivated.shape[0] >= 1

        num_control = self.sys_dyn_g.shape[1]
        num_activated_cbfs = subset.activated.shape[0] - 1
        num_deactivated_cbfs = subset.deactivated.shape[0] - 1
        num_state_eq = 0
        if self.state_eq_constraints is not None:
            num_state_eq = self.state_eq_constraints.shape[0]

        lambda_y_y_degree = y_degree
        xi_y_y_degree = y_degree
        if num_activated_cbfs == 1:
            lambda_y_y_degree = 0
            xi_y_y_degree = 0
        ball_c_degree = c_degree
        if num_deactivated_cbfs == 0:
            ball_c_degree = 0
        
        lambda_y_degree = [
            [
                Degree(x=x_degree, y=lambda_y_y_degree, c=c_degree)
                for _ in range(num_control)
            ]
            for _ in range(num_activated_cbfs)
        ]

        xi_y_degree = [
            Degree(x=x_degree, y=xi_y_y_degree, c=c_degree)
            for _ in range(num_activated_cbfs)
        ]

        clf_degree = Degree(x=x_degree, y=y_degree, c=c_degree)

        activated_cbfs_degree = [
            Degree(x=x_degree, y=y_degree, c=c_degree)
            for _ in range(num_activated_cbfs)
        ]

        if num_deactivated_cbfs != 0:
            deactivated_cbf_degree = [
                Degree(x=x_degree, y=y_degree, c=c_degree)
                for _ in range(num_deactivated_cbfs)
            ]
        else:   
            deactivated_cbf_degree = None
        
        ball_degree = Degree(x=x_degree, y=y_degree, c=ball_c_degree)

        if num_state_eq != 0:
            state_eq_degree = [
                Degree(x=x_degree, y=y_degree, c=c_degree)
                for _ in range(num_state_eq)
            ]
        else:
            state_eq_degree = None
        
        return SubsetVerifyLagrangianDegree(
            lambda_y=lambda_y_degree,
            xi_y=xi_y_degree,
            activated_cbfs=activated_cbfs_degree,
            clf=clf_degree,
            deactivated_cbfs=deactivated_cbf_degree,
            ball=ball_degree,
            state_eq_constraints=state_eq_degree
        )

    def _create_compatible_in_range_lagrangian_degrees(
        self,
        activated_cbf_x_degree: int,
        lambda_y_x_degrees: List[int],
        xi_y_x_degree: int,
        deactivated_cbfs_degrees: Optional[List[int]],
        ball_x_degree: int,
        clf_x_degree: int,
        state_eq_x_degrees: Optional[List[int]]
    ) -> CompatibilityInRangeLagragianDegrees:
        assert len(lambda_y_x_degrees) == self.sys_dyn_g.shape[1]
        
        range_non_negative_lagrangian_degree = []
        range_strictly_positive_lagrangian_degree = []

        if deactivated_cbfs_degrees is not None:
            for i in range(len(deactivated_cbfs_degrees)):
                range_strictly_positive_lagrangian_degree.append(
                    Degree(x=deactivated_cbfs_degrees[i], y=2, c=2)
                    )
        else:
            range_strictly_positive_lagrangian_degree.append(
                Degree(x=ball_x_degree, y=2, c=0)
            )
        range_non_negative_lagrangian_degree.append(
            Degree(x=activated_cbf_x_degree, y=2, c=2)
        )
        range_non_negative_lagrangian_degree.append(
            Degree(x=clf_x_degree, y=2, c=2)
        )
        lambda_y_lagrangian_degree = [
            Degree(x=lambda_y_x_degrees[i], y=0, c=2)
            for i in range(self.sys_dyn_g.shape[1])
        ]
        xi_y_lagrangian_degree = Degree(x=xi_y_x_degree, y=0, c=2)

        if state_eq_x_degrees is not None:
            assert len(state_eq_x_degrees) == self.state_eq_constraints.shape[0]
            state_eq_lagrangian_degree = [
                Degree(x=state_eq_x_degrees[i], y=2, c=2)
                for i in range(len(state_eq_x_degrees))
            ]
        else:
            state_eq_lagrangian_degree = None
        
        return CompatibilityInRangeLagragianDegrees(
            lambda_y=lambda_y_lagrangian_degree,
            xi_y=xi_y_lagrangian_degree,
            range_non_negative=range_non_negative_lagrangian_degree,
            range_strictly_positive=range_strictly_positive_lagrangian_degree,
            state_eq_const=state_eq_lagrangian_degree
        )
        
    def verify_subset(
        self,
        subset: UnionSubset,
        kappa_V: float,
        kappa_h: List[float],
        epsilon: float,
        lagrangian_degrees: SubsetVerifyLagrangianDegree
    ) -> bool:
        compatibility_obj = CompatibleClfCbfInSubset(
            x=self.x,
            sys_dyn_f=self.sys_dyn_f,
            sys_dyn_g=self.sys_dyn_g,
            subset=subset,
            Au=None,
            bu=None,
            state_eq_constraints=None
        )
        lagrangians = compatibility_obj.verify_compatibility_in_subset(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=epsilon,
            lagrangian_degrees=lagrangian_degrees
        )
        return lagrangians is not None    

    def verify_compatible_in_range(
        self,
        lagrangian_degrees: CompatibilityInRangeLagragianDegrees,
        activated_cbf: sym.Polynomial,
        deactivated_cbfs: Optional[np.ndarray],
        rho: float,
        kappa_V: float,
        kappa_h: float,
        epsilon: float
    ) -> bool:
        """
        This function verifies the compatibility of h_i(x) in the following range:
        {x | h_i(x)>=0, h_{i-1}(x) < 0,..., h_1(x) < 0, ρ - V(x)>= 0, ϵ₀² - xᵀx < 0}
        According to the design of CompatibilityInRange class, it only accepts the
        polynomial with strict positive and non-negative ranges. Hence, the activated
        cbf and clf should be in the non-negative range, and the deactivated cbfs and
        ball should be in the strictly positive range.
        non-negative range: [h_i(x), ρ - V(x)]
        strictly positive range: [-h_{i-1}(x), ..., -h_1(x), -ϵ₀² + xᵀx]
        Noted that in this class, self.clf is ρ - V(x), and self.ball is ϵ₀² - xᵀx.
        """
        range_non_negative_polys = np.array([activated_cbf, self.clf])
        range_strictly_positive_polys = (
            -deactivated_cbfs
            if deactivated_cbfs is not None
            else -np.array([self.ball])
        )

        compatibility_obj = CompatibleClfCbfInRange(
            x=self.x,
            sys_dyn_f=self.sys_dyn_f,
            sys_dyn_g=self.sys_dyn_g,
            V=rho - self.clf,
            h=activated_cbf,
            range_non_negative=range_non_negative_polys,
            range_strictly_positive=range_strictly_positive_polys,
            state_eq_const=self.state_eq_constraints,
            Au=self.Au,
            bu=self.bu
        )
        lagrangians = compatibility_obj.verify_compatibility(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=epsilon,
            lagrangian_degrees=lagrangian_degrees
        )
        return lagrangians is not None

    def step_two_verification(
        self,
        kappa_V: float,
        kappa_h: List[float],
        lagragian_x_degree: int,
        lagragian_y_degree: int,
        lagragian_c_degree: int,
        emptiness_lagragian_x_degrees: Optional[np.ndarray] = None,
        emptiness_lagragian_common_x_degree: Optional[int] = None
    )-> Optional[float]:
        """
        This is the second step of the verification. 
        The output is going to be the feasible epsilon, if there is no
        feasible epsilon, the output is None.
        """
        _, non_empty_subsets = self.all_non_empty_subsets_outside_ball(
            emptiness_lagragian_x_degrees=emptiness_lagragian_x_degrees,
            emptiness_lagragian_common_x_degree=emptiness_lagragian_common_x_degree
        )
        print("non_empty_subsets computed, number of non-empty subsets: ", non_empty_subsets.shape[0])

        eps_current = self.r_start
        infeasible = False
        while(eps_current >= self.r_lower_bound):
            to_be_verified_subsets = non_empty_subsets
            for subset in non_empty_subsets:
                lagrangian_degrees = self._create_subset_verification_lagragian_degrees(
                    subset=subset,
                    x_degree=lagragian_x_degree,
                    y_degree=lagragian_y_degree,
                    c_degree=lagragian_c_degree
                )
                current_is_feasible = self.verify_subset(
                    subset=subset,
                    kappa_V=kappa_V,
                    kappa_h=kappa_h,
                    epsilon=eps_current,
                    lagrangian_degrees=lagrangian_degrees
                )
                if current_is_feasible:
                    infeasible = False
                    to_be_verified_subsets = np.delete(to_be_verified_subsets, 0)
                else:
                    infeasible = True
                    non_empty_subsets = to_be_verified_subsets
                    break
            if infeasible:
                eps_current = eps_current / 2
            else:
                return eps_current
        return None

    def simplified_step_two_verification(
        self,
        activated_cbf_x_degree: int,
        lambda_y_x_degrees: List[int],
        xi_y_x_degree: int,
        deactivated_cbfs_common_degree: Optional[int],
        ball_x_degree: int,
        clf_x_degree: int,
        state_eq_x_degrees: Optional[List[int]],
        rho: float,
        kappa_V: float,
        kappa_h: List[float],
    ) -> Optional[float]:
        
        assert len(lambda_y_x_degrees) == self.sys_dyn_g.shape[1]
        if state_eq_x_degrees is not None:
            assert len(state_eq_x_degrees) == self.state_eq_constraints.shape[0]

        eps_current = self.r_start
        feasible_in_range = True
        while eps_current >= self.r_lower_bound:
            for i in range(self.cbfs.shape[0]):
                activated_cbf = self.cbfs[i]
                deactivated_cbfs = (
                    self.cbfs[:i]
                    if i > 0
                    else None
                )
                deactivated_cbfs_degrees = (
                    [deactivated_cbfs_common_degree] * deactivated_cbfs.shape[0]
                    if i > 0
                    else None
                )

                lagrangian_degrees = self._create_compatible_in_range_lagrangian_degrees(
                    activated_cbf_x_degree=activated_cbf_x_degree,
                    lambda_y_x_degrees=lambda_y_x_degrees,
                    xi_y_x_degree=xi_y_x_degree,
                    deactivated_cbfs_degrees=deactivated_cbfs_degrees,
                    ball_x_degree=ball_x_degree,
                    clf_x_degree=clf_x_degree,
                    state_eq_x_degrees=state_eq_x_degrees
                )
                current_is_feasible = self.verify_compatible_in_range(
                    lagrangian_degrees=lagrangian_degrees,
                    activated_cbf=activated_cbf,
                    deactivated_cbfs=deactivated_cbfs,
                    rho=rho,
                    kappa_V=kappa_V,
                    kappa_h=kappa_h[i],
                    epsilon=eps_current
                )
                if not current_is_feasible:
                    feasible_in_range = False
                    break
            if feasible_in_range:
                return eps_current
            else:
                eps_current = eps_current / 2
        return None                        
                

class CompatibleClfUnionCbfs:
    def __init__(
        self,
        x: np.ndarray,
        sys_dyn_f: np.ndarray,
        sys_dyn_g: np.ndarray,
        clf: sym.Polynomial,
        cbfs: np.ndarray,
        state_eq_constraints: Optional[np.ndarray],
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray]
    ):
        self.x = x
        self.sys_dyn_f = sys_dyn_f
        self.sys_dyn_g = sys_dyn_g
        self.clf = clf
        self.cbfs = cbfs
        self.state_eq_constraints = state_eq_constraints
        self.Au = Au
        self.bu = bu
    
    def general_union_verification(
        self,
        epsilon0_start: float,
        epsilon0_lower_bound: float,
        epsilon_start: float,
        epsilon_lower_bound: float,
        kappa_V: float,
        rho: float,
        kappa_h: List[float],
        ball_inclusion_ball_x_degrees: List[int],
        ball_inclusion_cbf_x_degrees: List[int],
        qp_feasible_in_ball_lambda_y_x_degrees: List[int],
        qp_feasible_in_ball_xi_y_x_degrees: List[int],
        qp_feasible_in_ball_ball_x_degrees: List[int],
        qp_feasible_in_ball_state_eq_x_degrees: Optional[List[List[int]]],
        compatible_in_subset_x_degree: int,
        compatible_in_subset_y_degree: int,
        compatible_in_subset_c_degree: int,
    ) -> bool:
        assert len(kappa_h) == self.cbfs.shape[0]
        assert len(ball_inclusion_ball_x_degrees) == self.cbfs.shape[0]
        assert len(ball_inclusion_cbf_x_degrees) == self.cbfs.shape[0]
        assert len(qp_feasible_in_ball_lambda_y_x_degrees) == self.cbfs.shape[0]
        assert len(qp_feasible_in_ball_xi_y_x_degrees) == self.cbfs.shape[0]
        assert len(qp_feasible_in_ball_ball_x_degrees) == self.cbfs.shape[0]
        if self.state_eq_constraints is not None:
            assert len(qp_feasible_in_ball_state_eq_x_degrees) == self.cbfs.shape[0]
        
        step_one = StepOne(
            clf=self.clf,
            cbfs=self.cbfs,
            sys_dyn_f=self.sys_dyn_f,
            sys_dyn_g=self.sys_dyn_g,
            x=self.x,
            r_start=epsilon0_start,
            r_lower_bound=epsilon0_lower_bound,
            state_eq_constraints=self.state_eq_constraints,
            Au=self.Au,
            bu=self.bu
        )

        epsilon0 = step_one.step_one_verification(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            ball_inclusion_ball_x_degrees=ball_inclusion_ball_x_degrees,
            ball_inclusion_cbf_x_degrees=ball_inclusion_cbf_x_degrees,
            qp_feasible_in_ball_lambda_y_x_degrees=qp_feasible_in_ball_lambda_y_x_degrees,
            qp_feasible_in_ball_xi_y_x_degrees=qp_feasible_in_ball_xi_y_x_degrees,
            qp_feasible_in_ball_ball_x_degrees=qp_feasible_in_ball_ball_x_degrees,
            qp_feasible_in_ball_state_eq_x_degrees=qp_feasible_in_ball_state_eq_x_degrees
        )
        if epsilon0 is None:
            print("Step one failed. Could not find a valide eps_0")
            return False
        else:
            print("Step one passed! epsilon0 is: ", epsilon0)
        
        step_two = StepTwo(
            clf=(rho - self.clf),
            cbfs=self.cbfs,
            ball=sym.Polynomial(epsilon0**2 - self.x.dot(self.x)),
            sys_dyn_f=self.sys_dyn_f,
            sys_dyn_g=self.sys_dyn_g,
            x=self.x,
            r_start=epsilon_start,
            r_lower_bound=epsilon_lower_bound,
            state_eq_constraints=self.state_eq_constraints,
            Au=self.Au,
            bu=self.bu
        )

        epsilon = step_two.step_two_verification(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            lagragian_x_degree=compatible_in_subset_x_degree,
            lagragian_y_degree=compatible_in_subset_y_degree,
            lagragian_c_degree=compatible_in_subset_c_degree
        )
        if epsilon is None:
            print("Step two failed. Could not find a valide eps")
            return False
        else:
            print("Step two passed! epsilon is: ", epsilon)
        
        return True
    
    def simplified_union_verification(
        self,
        epsilon0_start: float,
        epsilon0_lower_bound: float,
        epsilon_start: float,
        epsilon_lower_bound: float,
        kappa_V: float,
        rho: float,
        kappa_h: List[float],
        ball_inclusion_ball_x_degree: int,
        ball_inclusion_cbf_x_degree: int,
        qp_feasible_in_ball_lambda_y_x_degree: int,
        qp_feasible_in_ball_xi_y_x_degree: int,
        qp_feasible_in_ball_ball_x_degree: int,
        qp_feasible_in_ball_state_eq_x_degree: Optional[List[int]],
        activated_cbf_x_degree: int,
        lambda_y_x_degrees: List[int],
        xi_y_x_degree: int,
        deactivated_cbfs_common_degree: Optional[int],
        step_two_ball_x_degree: int,
        clf_x_degree: int,
        state_eq_x_degrees: Optional[List[int]],
    ) -> bool:
        step_one = StepOne(
            clf=self.clf,
            cbfs=self.cbfs,
            sys_dyn_f=self.sys_dyn_f,
            sys_dyn_g=self.sys_dyn_g,
            x=self.x,
            r_start=epsilon0_start,
            r_lower_bound=epsilon0_lower_bound,
            state_eq_constraints=self.state_eq_constraints,
            Au=self.Au,
            bu=self.bu
        )

        epsilon0 = step_one.simplififed_step_one_verification(
            kappa_V=kappa_V,
            kappa_h=kappa_h[0],
            ball_inclusion_ball_x_degree=ball_inclusion_ball_x_degree,
            ball_inclusion_cbf_x_degree=ball_inclusion_cbf_x_degree,
            qp_feasible_in_ball_lambda_y_x_degree=qp_feasible_in_ball_lambda_y_x_degree,
            qp_feasible_in_ball_xi_y_x_degree=qp_feasible_in_ball_xi_y_x_degree,
            qp_feasible_in_ball_ball_x_degree=qp_feasible_in_ball_ball_x_degree,
            qp_feasible_in_ball_state_eq_x_degree=qp_feasible_in_ball_state_eq_x_degree
        )
        if epsilon0 is None:
            print("Step one failed. Could not find a valide eps_0")
            return False
        else:
            print("Step one passed! epsilon0 is: ", epsilon0)
        
        step_two = StepTwo(
            clf=(rho - self.clf),
            cbfs=self.cbfs,
            ball=sym.Polynomial(epsilon0**2 - self.x.dot(self.x)),
            sys_dyn_f=self.sys_dyn_f,
            sys_dyn_g=self.sys_dyn_g,
            x=self.x,
            r_start=epsilon_start,
            r_lower_bound=epsilon_lower_bound,
            state_eq_constraints=self.state_eq_constraints,
            Au=self.Au,
            bu=self.bu
        )

        epsilon = step_two.simplified_step_two_verification(
            activated_cbf_x_degree=activated_cbf_x_degree,
            lambda_y_x_degrees=lambda_y_x_degrees,
            xi_y_x_degree=xi_y_x_degree,
            deactivated_cbfs_common_degree=deactivated_cbfs_common_degree,
            ball_x_degree=step_two_ball_x_degree,
            clf_x_degree=clf_x_degree,
            state_eq_x_degrees=state_eq_x_degrees,
            rho=rho,
            kappa_V=kappa_V,
            kappa_h=kappa_h
        )
        if epsilon is None:
            print("Step two failed. Could not find a valide eps")
            return False
        else:
            print("Step two passed! epsilon is: ", epsilon)
            return True

    


