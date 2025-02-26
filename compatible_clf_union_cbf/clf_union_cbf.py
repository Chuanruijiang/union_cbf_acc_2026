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
    solve_with_id
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





