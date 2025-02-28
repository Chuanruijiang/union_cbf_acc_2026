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


@dataclass
class BallInclusionLagrangian:
    r_minus_xTx: Degree
    h: Degree

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficitent_tol: Optional[float]
    )-> Self:
        r_minus_xTx_result = get_polynomial_result(
            result, self.r_minus_xTx, coefficitent_tol
        )
        h_result = get_polynomial_result(
            result, self.h, coefficitent_tol
        )
        return BallInclusionLagrangian(
            r_minus_xTx=r_minus_xTx_result,
            h=h_result
            )


@dataclass
class BallInclusionLagrangianDegree:
    r_minus_xTx: Degree
    h: Degree

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_r_minus_xTx: Optional[np.ndarray] = None,
        lagrangian_h: Optional[np.ndarray] = None,
    ) -> BallInclusionLagrangian:
        assert self.r_minus_xTx.y == 0
        assert self.h.y == 0
        r_minus_xTx = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.r_minus_xTx,
            lagrangian=lagrangian_r_minus_xTx
        )
        h = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.h,
            lagrangian=lagrangian_h
        )
        return BallInclusionLagrangian(
            r_minus_xTx=r_minus_xTx,
            h=h
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
        lagrangian_xi_y: Optional[sym.Polynomial] = None,
        lagrangian_range_non_negative: Optional[List[np.ndarray]] = None,
        lagrangian_range_strictly_positive: Optional[List[np.ndarray]] = None,
        lagrangian_state_eq_const: Optional[List[np.ndarray]] = None,
    ) -> CompatibilityInRangeLagragians:
        assert isinstance(self.lambda_y, list)
        assert isinstance(self.xi_y, Degree)
        if self.range_non_negative is not None:
            assert isinstance(self.range_non_negative, list)
        if self.range_strictly_positive is not None:
            assert isinstance(self.range_strictly_positive, list)
        if self.state_eq_const is not None:
            assert isinstance(self.state_eq_const, list)
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
                is_sos=True,
                degree=self.range_non_negative,
                lagrangian=lagrangian_range_non_negative
            )
            if self.range_non_negative is not None
            else None
        )
        range_strictly_positive_lagrangians = (
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
        self.c_set = (
            sym.Variables(self.c)
            if range_strictly_positive is not None
            else None
        )
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
        sos_type: solvers.MathematicalProgram.NonnegativePolynomial
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
            c=self.c_set,
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
    

@dataclass
class StepOne:
    clf: sym.Polynomial
    cbfs: np.ndarray
    sys_dyn_f: np.ndarray
    sys_dyn_g: np.ndarray
    x: np.ndarray
    r_start: float
    r_lower_bound: float

    def _compatible_in_ball(
        self,
        r: float,
        sys_dyn_f: np.ndarray,
        sys_dyn_g: np.ndarray,
        clf: sym.Polynomial,
        cbf: sym.Polynomial,
        kappa_V: float,
        kappa_h: float,
        lagragian_degree: CompatibilityInRangeLagragianDegrees
    ) -> bool:
        ball = sym.Polynomial(r - self.x.dot(self.x))
        compatibility_obj = CompatibleClfCbfInRange(
            x=self.x,
            sys_dyn_f=sys_dyn_f,
            sys_dyn_g=sys_dyn_g,
            V=clf,
            h=cbf,
            range_non_negative=np.array([ball]),
            range_strictly_positive=None,
            state_eq_const=None,
            Au=None,
            bu=None
        )
        lagrangians = compatibility_obj.verify_compatibility(
            kappa_V=kappa_V,
            kappa_h=kappa_h,
            epsilon=0,
            lagrangian_degrees=lagragian_degree
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
        ball_inclusion_lagrangian_degrees: List[BallInclusionLagrangianDegree],
        compatibility_lagrangian_degrees: List[CompatibilityInRangeLagragianDegrees]
    ) -> Optional[float]:
        assert len(kappa_h) == self.cbfs.shape[0]
        assert len(ball_inclusion_lagrangian_degrees) == self.cbfs.shape[0]
        assert len(compatibility_lagrangian_degrees) == self.cbfs.shape[0]
        
        current_r = self.r_start
        while current_r >= self.r_lower_bound:
            for i in range(self.cbfs.shape[0]):
                inclusion = self._ball_included_by_cbf(
                    r=current_r,
                    cbf=self.cbfs[i],
                    lagrangian_degree=ball_inclusion_lagrangian_degrees[i]
                )
                if inclusion:
                    compatible = self._compatible_in_ball(
                        r=current_r,
                        sys_dyn_f=self.sys_dyn_f,
                        sys_dyn_g=self.sys_dyn_g,
                        clf=self.clf,
                        cbf=self.cbfs[i],
                        kappa_V=kappa_V,
                        kappa_h=kappa_h[i],
                        lagragian_degree=compatibility_lagrangian_degrees[i]
                    )
                    if compatible:
                        return current_r
            # if we can get here, this means the for loop has finished and the current r ball
            # is either not included by any cbf or not clf-cbf is not compatible in it.
            current_r = current_r / 2
        # if the while loop finishes, this means we cannot find a valid r, meaning that we
        # cannot find a valid ball that is included by a cbf and the clf-cbf is compatible in it.
        return None


@ dataclass
class StepTwo:
    clf: sym.Polynomial
    cbfs: np.ndarray
    ball: sym.Polynomial
    sys_dyn_f: np.ndarray
    sys_dyn_g: np.ndarray
    x: np.ndarray
    r_start: float
    r_lower_bound: float

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
    



