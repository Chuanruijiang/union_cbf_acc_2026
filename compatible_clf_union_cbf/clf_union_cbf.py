# create a clf_union_cbf_class: 
# it should be inherited from clf_cbf class
# we also need to include UnionCBF class in this class
# we don't need UnionSubset class in this class but we
# will use it as a external class
# we need to include the following methods:
# 1. __init__ method:
# 2. VerificationInSubset
# 3. GeneralizedVerification
# 4. SimplifiedVerification
# 5. Synthesis

# For the first step of the verification:

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
    lie_derivative
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
        c: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_lambda_y: Optional[np.ndarray] = None,
        lagrangian_xi_y: Optional[np.ndarray] = None,
        lagrangian_range_non_negative: Optional[List[np.ndarray]] = None,
        lagrangian_range_strictly_positive: Optional[List[np.ndarray]] = None,
        lagrangian_state_eq_const: Optional[List[np.ndarray]] = None,
    ) -> CompatibilityInRangeLagragians:
        lambda_y_lagragian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=c,
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y
        )
        xi_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=c,
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
                c=c,
                sos_type=sos_type,
                is_sos=False,
                degree=self.range_non_negative,
                lagrangian=lagrangian_range_non_negative
            )
            if self.range_non_negative is not None
            else None
        )
        range_non_negative_lagrangians = (
            to_lagrangian_impl(
                prog=prog,
                x=x,
                y=y,
                c=c,
                sos_type=sos_type,
                is_sos=False,
                degree=self.range_strictly_positive,
                lagrangian=lagrangian_range_strictly_positive
            )
            if self.range_strictly_positive is not None
            else None
        )
        state_eq_const_lagrangians = (
            to_lagrangian_impl(
                prog=prog,
                x=x,
                y=y,
                c=c,
                sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
                is_sos=True,
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
            range_strictly_positive=range_non_negative_lagrangians,
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
        range_non_negative: Optional[List[sym.Polynomial]],
        range_strictly_positive: Optional[List[sym.Polynomial]],
        state_eq_const: Optional[List[sym.Polynomial]],
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
    ):
        self.x = x
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
        
    )
