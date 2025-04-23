from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_union_cbf.utils import (
    truth_table,
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    elementary_symetric_polynomials,
    Degree,
    to_lagrangian_impl,
    is_sos,
    BackoffScale
)
from compatible_clf_union_cbf.inclusion import(
    BallInclusionLagrangian,
    BallInclusionLagrangianDegree,
    BallInclusion,
    UnsafeRegionExclusionLagrangians,
    UnsafeRegionExclusionLagrangianDegrees,
    UnsafeExclusion,
    PointsInclusionConstriants
)


@dataclass
class EmptinessLagrangians:
    """
    We want to check whether a given semi-algrbraic set is empty or not.
    To be more specific, given a set of polynomials h_i(x), i=1,...,n.
    Define a 0-1 vector with the same length n, and each element defines
    whether the corresponding polynomial is active or not.

    For example, given a set of polynomials h_1(x), h_2(x), h_3(x).
    If we have a 0-1 vector [1, 0, 1], then we will check whether the set
    {x| h_1(x) >= 0, h_2(x) < 0, h_3(x) >= 0} is empty or not.

    In this example, we want to verify that thes does not exist any x that
    satisfies h_1(x) >= 0, h_2(x) < 0, h_3(x) >= 0. This is equivalent to
    say there does not exist any [x, y] such that:
    h_1(x)>=0, y^2*h_2(x) = -1, and h_3(x)>=0.

    Hence the SOS program for verifying the emptiness of the set is:
    Find p(x, y) and q(x) such that:
    -1 - p(x, y)^T[h_1(x), h_3(x)] - q(x)(y^2 * h_2(x) + 1 ) is SOS.

    This example can be extened to the general case of a semi-algebraic
    set. Hence, to verify the emptiness of a set, we need to find the
    lagragians for activated polinomials (p(x,c)), and the lagragians for
    the deactivated polynomials (q(x)).
    """

    # both of them are array of sym.polynomial
    activated: np.ndarray
    deactivated: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        lagrangian_activaed_result = get_polynomial_result(
            result, self.activated, coefficient_tol
        )
        lagrangian_deactivated_result = (
            None
            if self.deactivated is None
            else get_polynomial_result(result, self.deactivated, coefficient_tol)
        )
        return EmptinessLagrangians(
            activated=lagrangian_activaed_result,
            deactivated=lagrangian_deactivated_result,
        )


@dataclass
class EmptinessLagrangianDegrees:
    """
    This class specifies the degrees for the lagragians of activated and
    deactivated polynomials that metioned above. Since all these lagragians
    are array of sym.polynomials, then we need a list of XYdegrees for each.
    """

    activated: List[Degree]
    deactivated: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        y: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_activated: Optional[np.ndarray] = None,
        lagrangian_deactivated: Optional[np.ndarray] = None,
    ) -> EmptinessLagrangians:
        activated = to_lagrangian_impl(
            prog,
            x,
            y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.activated,
            lagrangian=lagrangian_activated,
        )
        deactivated = (
            None
            if self.deactivated is None
            else to_lagrangian_impl(
                prog,
                x,
                y,
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.deactivated,
                lagrangian=lagrangian_deactivated,
            )
        )
        return EmptinessLagrangians(activated=activated, deactivated=deactivated)


@dataclass
class UnionSubset:
    """
    This class defines a disjoint subset in the union. For this
    subset, all the activated and deactivcated polynomials are
    represented by array of sym.polynomials.
    """

    activated: np.ndarray
    deactivated: Optional[np.ndarray]
    variables: np.ndarray

    def is_empty(
        self,
        emptiness_lagrangian_degrees: EmptinessLagrangianDegrees,
    ) -> bool:
        """
        This function checks whether the subset is empty or not by
        solving the SOS program that talked about above.
        """
        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.variables)
        if self.deactivated is None:
            assert emptiness_lagrangian_degrees.deactivated is None
            assert (
                len(emptiness_lagrangian_degrees.activated) == self.activated.shape[0]
            )
            activated_polys = self.activated
            deactivated_polys = None
            y_squared_poly = None
            y_set = None
            prog.AddIndeterminates(x_set)
        else:
            assert emptiness_lagrangian_degrees.deactivated is not None
            assert (
                len(emptiness_lagrangian_degrees.activated) == self.activated.shape[0]
            )
            assert (
                len(emptiness_lagrangian_degrees.deactivated)
                == self.deactivated.shape[0]
            )
            activated_polys = self.activated
            deactivated_polys = self.deactivated
            y = sym.MakeVectorContinuousVariable(self.deactivated.shape[0], "y")
            y_squared_poly = np.array(
                [sym.Polynomial(sym.Monomial(y[i], 2)) for i in range(y.shape[0])]
            )
            y_set = sym.Variables(y)
            xy_set = sym.Variables(np.concatenate([self.variables, y]))
            prog.AddIndeterminates(xy_set)

        emptiness_lagrangians = emptiness_lagrangian_degrees.to_lagrangians(
            prog=prog,
            x=x_set,
            y=y_set,
        )

        self._add_emptiness_constraints(
            prog=prog,
            y_square=y_squared_poly,
            activated=activated_polys,
            deactivated=deactivated_polys,
            emptiness_lagrangians=emptiness_lagrangians,
        )
        emptiness_result = solve_with_id(prog)
        return emptiness_result.is_success()

    def yc_sets_for_verification_wclf_outball(
        self,
        with_control_input_limits: bool = False,
        control_constriant_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        If we call this function, then at least we should have the
        ball polynomial as the deactivated polynomial. Hence, the
        deactivated member in this class should be not None. Also,
        we consider CLF, so the activated member should be more than 1.
        This function finds the number of activated cbfs and the
        number of deactivated cbfs. Then, it will construct all the
        y_sets and c_sets for the compatibility verifcation lagrangians,
        packed them into two ndarrays and return them.
        The outputs are:
        1. y_sets,
        2. c_sets,
        3. y_all,
        4. c_all
        """
        assert self.deactivated is not None
        assert self.deactivated.shape[0] >= 1
        assert self.activated.shape[0] > 1
        n_activated_cbfs = self.activated.shape[0] - 1
        n_deactivated_polys = self.deactivated.shape[0]
        
        # compute y_sets
        y_i_size = 2
        if with_control_input_limits:
            assert control_constriant_size is not None
            y_i_size += control_constriant_size
        y_sets = np.empty((n_activated_cbfs + 1), dtype=object)
        y_all = np.empty((n_activated_cbfs, y_i_size), dtype=object)
        for i in range(n_activated_cbfs):
            y_i = sym.MakeVectorContinuousVariable(y_i_size, "y"+str(i))
            y_all[i] = y_i
        y_all_set = sym.Variables(y_all.flatten())
        y_sets[-1] = y_all_set
        for i in range(n_activated_cbfs):
            ya_i = np.delete(y_all, i, axis=0)
            if ya_i.size == 0:
                pass
            else:
                y_sets[i] = sym.Variables(ya_i.flatten())
        
        # compute c_sets
        c_sets = np.empty((n_deactivated_polys + 1,), dtype=object)
        c_all = sym.MakeVectorContinuousVariable(n_deactivated_polys, "c")
        c_sets[-1] = sym.Variables(c_all)
        for i in range(n_deactivated_polys):
            c_i = np.delete(c_all, i, axis=0)
            if c_i.size == 0:
                pass
            else:
                c_sets[i] = sym.Variables(c_i)
        
        return (y_sets, c_sets, y_all, c_all)

    def _add_emptiness_constraints(
        self,
        prog: solvers.MathematicalProgram,
        y_square: Optional[np.ndarray],
        activated: np.ndarray,
        deactivated: Optional[np.ndarray],
        emptiness_lagrangians: EmptinessLagrangians,
    ) -> None:
        assert activated.shape[0] == emptiness_lagrangians.activated.shape[0]
        poly = -sym.Polynomial(sym.Monomial())
        poly -= emptiness_lagrangians.activated.dot(activated)

        if deactivated is None:
            assert emptiness_lagrangians.deactivated is None
            assert y_square is None
        else:
            assert emptiness_lagrangians.deactivated is not None
            assert y_square is not None
            assert deactivated.shape[0] == emptiness_lagrangians.deactivated.shape[0]
            poly -= emptiness_lagrangians.deactivated.dot(y_square * deactivated + 1)

        prog.AddSosConstraint(poly)


@dataclass
class ActivationIndicator:
    """
    This class stores the list of 0-1 vectors as the activation
    indicator of the disjoint subsets in the union.
    """

    vecotor01: np.ndarray

    def to_union_subsets(
        self,
        polys: np.ndarray,
        variables: np.ndarray,
    ) -> np.ndarray:
        assert len(self.vecotor01.shape) == 2
        assert self.vecotor01.shape[1] == polys.shape[0]
        subsets = np.array([])
        for i in range(self.vecotor01.shape[0]):
            if np.all(self.vecotor01[i] == 1):
                subset = UnionSubset(
                    activated=polys,
                    deactivated=None,
                    variables=variables,
                )
            else:
                activated_idx = np.where(self.vecotor01[i] == 1)
                deactivated_idx = np.where(self.vecotor01[i] == 0)
                activated_polys = polys[activated_idx]
                deactivated_polys = polys[deactivated_idx]
                subset = UnionSubset(
                    activated=activated_polys,
                    deactivated=deactivated_polys,
                    variables=variables,
                )
            subsets = np.append(subsets, subset)
        return subsets


class UnionCBF:
    def __init__(self, h: np.ndarray, x: np.ndarray):
        self.h = h
        self.x = x

    def non_empty_disjoint_subsets(
        self,
        *,
        outside_ball: bool = False,
        ball_poly: Optional[sym.Polynomial] = None,
        with_clf: bool = False,
        clf: Optional[sym.Polynomial] = None,
        lagragian_x_degrees: Optional[np.ndarray] = None,
        common_x_degree: Optional[int] = None,
    ) -> np.ndarray:
        """
        This function outputs all the 0-1 vector of all non-empty disjoint
        subsets of the union. If the outside_ball is True, then the output
        will be all the 0-1 verctors that represent the non-empty disjoint
        subsets of the union that are outside the ball.

        The genral idea is to see the ball as a deactivated polynomial. If
        ourside ball is False, the output only contains the 0-1 vectors that
        presents the actication of CBFs. But if outside_ball is True, the
        output contains the 0-1 vectors repesenting the activation and deactivation
        of CBFs and also the decativation of the ball polynomial.

        For lagrangian degrees:
        During emptiness checking of a subset, each CBF
        will have a lagragain in fronr of it no matter that CBF is activated or
        deactivated. Also, assume an activated CBF become deactivated, the only 
        difference is that the CBF will need to time a y^2 term, the x_degree of
        that CBF does not change. Hence, the x_degree of the Lagrangian in front
        of that CBF can be independent of the activation status of that CBF.
        Also, if the user does not set the degree of x variables for the lagragians
        of CBFs, then the default value is 2.

        Finally, if outside_ball is True, we should also provide the lagrangian 
        x degrees for the ball polynomial. The ball polynomial is always deactivated.
        """
        # if outside ball, the deactivated ball poly should be r-x^Tx
        # the clf poly should be rho-V(x)
        all_possible_activation_cases = truth_table(self.h.shape[0])
        all_polys = self.h

        if outside_ball:
            assert ball_poly is not None
            all_possible_activation_cases = np.concatenate(
                [all_possible_activation_cases, 
                 np.zeros((all_possible_activation_cases.shape[0], 1))],
                axis=1
            )
            all_polys = np.concatenate([all_polys, np.array([ball_poly])])
        
        if with_clf:
            assert clf is not None
            all_possible_activation_cases = np.concatenate(
                [np.ones((all_possible_activation_cases.shape[0], 1)),
                 all_possible_activation_cases],
                axis=1
            )
            all_polys = np.concatenate([np.array([clf]), all_polys])
        
        non_empty_01_vectors = np.array([])
        for i in range(1, all_possible_activation_cases.shape[0]):
            assert all_possible_activation_cases[i].shape[0] == (all_polys.shape[0])
            
            activated_idx = np.where(all_possible_activation_cases[i] == 1)
            deactivated_idx = np.where(all_possible_activation_cases[i] == 0)
            activated_polys = all_polys[activated_idx]
            deactivated_polys = (
                all_polys[deactivated_idx]
                if deactivated_idx[0].shape[0] > 0
                else None
            )
            if lagragian_x_degrees is not None:
                assert common_x_degree is None
                assert len(lagragian_x_degrees) == all_polys.shape[0]
                various_act_x_degrees = lagragian_x_degrees[activated_idx]
                various_deact_x_degrees = lagragian_x_degrees[deactivated_idx]
            else:
                various_act_x_degrees = None
                various_deact_x_degrees = None

            current_subset = UnionSubset(
                activated=activated_polys,
                deactivated=deactivated_polys,
                variables=self.x,
            )
            emptiness_lagrangian_degrees = self._create_emptiness_lagragian_degrees(
                activated_polys=activated_polys,
                deactivated_polys=deactivated_polys,
                common_x_degree=common_x_degree,
                various_act_x_degrees=various_act_x_degrees,
                various_deact_x_degrees=various_deact_x_degrees,
            )

            current_subset_is_empty = current_subset.is_empty(
                emptiness_lagrangian_degrees=emptiness_lagrangian_degrees
            )
            if not current_subset_is_empty:
                non_empty_01_vectors = np.append(
                    non_empty_01_vectors, all_possible_activation_cases[i]
                )

        return non_empty_01_vectors.reshape(-1, all_polys.shape[0])

    def _create_emptiness_lagragian_degrees(
        self,
        activated_polys: np.ndarray,
        deactivated_polys: Optional[np.ndarray],
        common_x_degree: Optional[int],
        various_act_x_degrees: Optional[List[int]],
        various_deact_x_degrees: Optional[List[int]],
    ) -> EmptinessLagrangianDegrees:
        if deactivated_polys is None:
            # In this case, we only have activated polynomials, we need to
            # get the lagrangian degrees for activated polynomials only.
            if common_x_degree is not None:
                assert various_act_x_degrees is None
                activated_degrees = [
                    Degree(x=common_x_degree, y=0, c=0)
                    for _ in range(activated_polys.shape[0])
                ]
            elif various_act_x_degrees is not None:
                assert common_x_degree is None
                assert len(various_act_x_degrees) == activated_polys.shape[0]
                activated_degrees = [
                    Degree(x=x, y=0, c=0)
                    for x in various_act_x_degrees
                    ]
            else:
                activated_degrees = [
                    Degree(x=2, y=0, c=0) for _ in range(activated_polys.shape[0])
                ]
            
            return EmptinessLagrangianDegrees(
                activated=activated_degrees, deactivated=None
            )
        else:
            # In this case, we have deactivated polynomials, we need to 
            # get the lagrangian degrees for both activated and deactivated polynomials
            if common_x_degree is not None:
                assert various_act_x_degrees is None
                assert various_deact_x_degrees is None
                activated_degrees = [
                    Degree(x=common_x_degree, y=2, c=0)
                    for _ in range(activated_polys.shape[0])
                ]
                deactivated_degrees = [
                    Degree(x=common_x_degree, y=0, c=0)
                    for _ in range(deactivated_polys.shape[0])
                ]
            elif various_act_x_degrees is not None:
                assert various_deact_x_degrees is not None
                assert len(various_act_x_degrees) == activated_polys.shape[0]
                assert len(various_deact_x_degrees) == deactivated_polys.shape[0]
                activated_degrees = [
                    Degree(x=x, y=2, c=0)
                    for x in various_act_x_degrees
                    ]
                deactivated_degrees = [
                    Degree(x=x, y=0, c=0)
                    for x in various_deact_x_degrees
                    ]
            else:
                activated_degrees = [
                    Degree(x=2, y=2, c=0)
                    for _ in range(activated_polys.shape[0])
                ]
                deactivated_degrees = [
                    Degree(x=2, y=0, c=0)
                    for _ in range(deactivated_polys.shape[0])
                ]
            
            
            return EmptinessLagrangianDegrees(
                activated=activated_degrees, deactivated=deactivated_degrees
            )


class CbfConstraint:
    """
    Add the linear constraint ∂h(x)/∂x * f(x) + ∂h(x)/∂x * g(x)*u >= -κₕ * h(x) on u.
    """

    def __init__(
        self,
        h: sym.Polynomial,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        kappa: tuple[float, List[float]],
        relative_degree: Optional[int],
    ):
        if relative_degree is None:
            relative_degree = 1
            kappa = [kappa]
        else:
            assert isinstance(kappa, List)

        beta_vector = elementary_symetric_polynomials(kappa)
        lie_derivative_vector = np.array(
            [
                lie_derivative(poly=h, vector_feild=f, variables=x, pow=j)
                for j in range(relative_degree, -1, -1)
            ]
        )
        self.rhs = -np.dot(lie_derivative_vector, beta_vector)
        self.lhs_coeff = lie_derivative(
            poly=lie_derivative_vector[1], vector_feild=g, variables=x, pow=1
        )
        self.x = x

    def add_to_prog(
        self, prog: solvers.MathematicalProgram, x_val: np.ndarray, u: np.ndarray
    ) -> solvers.Binding[solvers.LinearConstraint]:
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        constraint = prog.AddLinearConstraint(lhs_coeff, rhs, np.inf, u)
        return constraint



@dataclass
class CompatibilityLagragians:
    lambda_y: np.ndarray
    xi_y: sym.Polynomial
    rho_minus_V: sym.Polynomial
    h: sym.Polynomial
    deactivated_h: Optional[np.ndarray]
    state_eq_constraints: Optional[np.ndarray]

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        lambda_y_result = get_polynomial_result(result, self.lambda_y, coefficient_tol)
        xi_y_result = get_polynomial_result(result, self.xi_y, coefficient_tol)
        rho_minus_V_result = get_polynomial_result(result, self.rho_minus_V, coefficient_tol)
        h_result = get_polynomial_result(result, self.h, coefficient_tol)
        deactivated_h_result = (
            None
            if self.deactivated_h is None
            else get_polynomial_result(result, self.deactivated_h, coefficient_tol)
        )
        state_eq_constraints_result = (
            None
            if self.state_eq_constraints is None
            else get_polynomial_result(result, self.state_eq_constraints, coefficient_tol)
        )
        return CompatibilityLagragians(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            rho_minus_V=rho_minus_V_result,
            h=h_result,
            deactivated_h=deactivated_h_result,
            state_eq_constraints=state_eq_constraints_result,
        )

@dataclass
class CompatibilityLagragianDegrees:
    lambda_y: List[Degree]
    xi_y: Degree
    rho_minus_V: Degree
    h: Degree
    deactivated_h: Optional[List[Degree]]
    state_eq_constraints: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        y: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_lambda_y: Optional[np.ndarray] = None,
        lagrangian_xi_y: Optional[np.ndarray] = None,
        lagrangian_rho_minus_V: Optional[np.ndarray] = None,
        lagrangian_h: Optional[np.ndarray] = None,
        lagrangian_deactivated_h: Optional[np.ndarray] = None,
    ) -> CompatibilityLagragians:
        assert isinstance(self.lambda_y, List)
        assert isinstance(self.xi_y, Degree)
        if self.deactivated_h is not None:
            assert isinstance(self.deactivated_h, List)
        if self.state_eq_constraints is not None:
            assert isinstance(self.state_eq_constraints, list)
        
        lambda_y_lagrangians = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y,
        )

        xi_y_lagrangian = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.xi_y,
            lagrangian=lagrangian_xi_y,
        )

        rho_minus_V_lagrangian = to_lagrangian_impl(
            prog,
            x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.rho_minus_V,
            lagrangian=lagrangian_rho_minus_V,
        )

        h_lagrangian = to_lagrangian_impl(
            prog,
            x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.h,
            lagrangian=lagrangian_h,
        )

        deactivated_h_lagrangian = (
            to_lagrangian_impl(
                prog=prog,
                x=x,
                y=y,
                c=None,
                sos_type=sos_type,
                is_sos=True,
                degree=self.deactivated_h,
                lagrangian=(
                    lagrangian_deactivated_h
                    if lagrangian_deactivated_h is not None
                    else None)
            )
            if self.deactivated_h is not None
            else None
        )

        state_eq_constraints = (
            None
            if self.state_eq_constraints is None
            else to_lagrangian_impl(
                prog,
                x,
                y=y,
                c=None,
                sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
                is_sos=False,
                degree=self.state_eq_constraints,
                lagrangian=None,
            )
        )

        return CompatibilityLagragians(
            lambda_y=lambda_y_lagrangians,
            xi_y=xi_y_lagrangian,
            rho_minus_V=rho_minus_V_lagrangian,
            h=h_lagrangian,
            deactivated_h=deactivated_h_lagrangian,
            state_eq_constraints=state_eq_constraints,
        )
    

class UnionCbfSynthesisGivenClf:
    def __init__(
        self,
        x: np.ndarray,
        sys_dyn_f: np.ndarray,
        sys_dyn_g: np.ndarray,
        clf: sym.Polynomial,
        rho: float,
        num_cbf: int,
        unsafe_polys: np.ndarray,
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
        state_eq_constraints: Optional[np.ndarray],
        
        kappaV: float,
        kappah: List[float],
        epsilon_0: float,
        epsilon: float,
        
        cbf_x_degrees: List[int],
        cbf_ball_inclusion_ball_x_degree: int,
        cbf_ball_inclusion_cbf_x_degree: int,
        compatible_lambda_y_x_degrees: List[int],
        compatible_xi_y_x_degree: int,
        compatible_rho_minus_V_x_degree: int,
        compatible_h_x_degree: int,
        compatible_deact_cbf_x_degree: List[int],
        state_eq_x_degrees: Optional[List[int]],
        safety_h_x_degree: int,
        safety_unsafe_polys_x_degree: List[int],
    ):
        self.x = x
        self.sys_dyn_f = sys_dyn_f
        self.sys_dyn_g = sys_dyn_g
        self.clf = clf
        self.rho = rho
        self.num_cbf = num_cbf
        self.unsafe_polys = unsafe_polys
        self.Au = Au
        self.bu = bu
        if self.Au is not None and self.bu is not None:
            assert self.Au.shape[1] == self.sys_dyn_g.shape[1]
            assert self.Au.shape[0] == self.bu.shape[0]
        self.state_eq_constraints = state_eq_constraints
        self.kappaV = kappaV
        self.kappah = kappah
        assert len(self.kappah) == self.num_cbf
        self.epsilon_0 = epsilon_0
        self.epsilon = epsilon
        assert self.epsilon_0 > 0
        assert self.epsilon > 0
        self.cbf_x_degrees = cbf_x_degrees
        assert len(self.cbf_x_degrees) == self.num_cbf
        self.cbf_ball_inclusion_ball_x_degree = cbf_ball_inclusion_ball_x_degree
        self.cbf_ball_inclusion_cbf_x_degree = cbf_ball_inclusion_cbf_x_degree
        self.compatible_lambda_y_x_degrees = compatible_lambda_y_x_degrees
        self.compatible_xi_y_x_degree = compatible_xi_y_x_degree
        self.compatible_rho_minus_V_x_degree = compatible_rho_minus_V_x_degree
        self.compatible_h_x_degree = compatible_h_x_degree
        self.compatible_deact_cbf_x_degree = compatible_deact_cbf_x_degree
        assert len(self.compatible_deact_cbf_x_degree) == self.num_cbf-1
        self.state_eq_x_degrees = state_eq_x_degrees
        if self.state_eq_constraints is not None:
            assert len(self.state_eq_x_degrees) == self.state_eq_constraints.shape[0]
        self.safety_h_x_degree = safety_h_x_degree
        self.safety_unsafe_polys_x_degree = safety_unsafe_polys_x_degree
        assert len(self.safety_unsafe_polys_x_degree) == self.unsafe_polys.shape[0]

        # creating symbolic variables:
        self.nu = self.sys_dyn_g.shape[1]
        self.x = x
        self.y = (
            sym.MakeVectorContinuousVariable(2, "y")
            if Au is None
            else sym.MakeVectorContinuousVariable(2+Au.shape[0], "y")
            )
        self.y_squared_poly = np.array([
            sym.Polynomial(sym.Monomial(self.y[i], 2)) 
            for i in range(self.y.shape[0])
        ])
        self.x_set = sym.Variables(self.x)
        self.y_set = sym.Variables(self.y)
        self.xy_set = sym.Variables(np.concatenate([self.x, self.y], axis=0))
        
    def _calc_xi_lambda_for_cbf_synthesis(
        self,
        cbf: sym.Polynomial,
        kappa_h: float,
    )-> Tuple[np.ndarray, np.ndarray]:
        """
        This function is used after the synthesis of the CLF. 
        In this function, CLF is a given polynomial. 
        The epsilon is computed after the synthesis of the CLF.
        """
        lambda_rows = 2 # a cbf + a clf
        if self.Au is not None:
            lambda_rows += self.Au.shape[0]
        lambda_cols = self.sys_dyn_g.shape[1]
        lambda_mat = np.empty((lambda_rows, lambda_cols), dtype=object)
        xi = np.empty((lambda_rows,), dtype=object)

        # loading CBF constriants:
        Lgh = lie_derivative(cbf, self.sys_dyn_g, self.x, 1)
        Lfh = lie_derivative(cbf, self.sys_dyn_f, self.x, 1)
        lambda_mat[0] = -Lgh
        xi[0] = Lfh + kappa_h*cbf - self.epsilon

        # loading CLF constraints:
        LgV = lie_derivative(self.clf, self.sys_dyn_g, self.x, 1)
        LfV = lie_derivative(self.clf, self.sys_dyn_f, self.x, 1)
        lambda_mat[1] = LgV
        xi[1] = -LfV - self.kappaV*self.clf

        # loading Au and bu:
        if self.Au is not None:
            lambda_mat[2:] = self.Au
            xi[2:] = self.bu
        
        return xi, lambda_mat
   
    def _add_compatibility_constraints(
        self,
        prog: solvers.MathematicalProgram,
        cbf: sym.Polynomial,
        lambda_mat: np.ndarray,
        xi: np.ndarray,
        deactivated_h: Optional[sym.Polynomial],
        lagrangians: CompatibilityLagragians,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> sym.Polynomial:
        poly_one = sym.Polynomial(1)
        poly = -poly_one
        
        # s0(x)Λ(x)ᵀy²:
        assert lambda_mat.shape[0] == self.y_squared_poly.shape[0]
        poly -= lagrangians.lambda_y.dot(self.y_squared_poly.dot(lambda_mat))

        # s1(x)ξ(x)ᵀy:
        poly -= lagrangians.xi_y * (self.y_squared_poly.dot(xi) + 1)

        # s2(x)(ρ(x) - V(x)):
        poly -= lagrangians.rho_minus_V * (self.rho - self.clf)

        # s3(x)h(x):
        poly -= lagrangians.h * cbf

        # if we have deactivated cbfs:
        if deactivated_h is not None:
            assert lagrangians.deactivated_h is not None
            poly += lagrangians.deactivated_h.dot(deactivated_h)

        # if we have state_eq_const:
        if self.state_eq_constraints is not None:
            assert lagrangians.state_eq_constraints is not None
            poly -= lagrangians.state_eq_constraints.dot(self.state_eq_constraints)
    
        prog.AddSosConstraint(poly, sos_type)
        return poly

    def _add_cbf_ball_inclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        cbf: sym.Polynomial,
        ball_inclusion_lagrangian: BallInclusionLagrangian
    ) -> sym.Polynomial:
        ball_inclusion = BallInclusion(
            radius=self.epsilon_0,
            h=cbf,
            x=self.x
        )
        poly = ball_inclusion.add_ball_inclusion_constraint(
            prog=prog,
            ball_inclusion_lagrangian=ball_inclusion_lagrangian
            )
        return poly

    def _add_cbf_safety_constraint(
        self,
        prog: solvers.MathematicalProgram,
        cbf: sym.Polynomial,
        unsafe_lagrangians: UnsafeRegionExclusionLagrangians
    ) -> sym.Polynomial:
        unsafe_exclusion = UnsafeExclusion(
            unsafe_polys=self.unsafe_polys,
            h=cbf,
            x=self.x
        )
        poly = unsafe_exclusion.add_unsafe_exclusion_constraint(
            prog=prog,
            unsafe_exclusion_lagrangians=unsafe_lagrangians
        )
        return poly

    def _add_cbf_point_inclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        cbf: sym.Polynomial,
        points_to_include: np.ndarray,
        weights_to_include: np.ndarray,
        anchor_points: Optional[np.ndarray],
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]]
    ):
         point_inclusion = PointsInclusionConstriants(
             x=self.x,
             p=cbf,
             points_to_include=points_to_include,
             point_inclusion_weights=weights_to_include,
             anchor_points=anchor_points,
             p_anchor_bounds=anchor_bounds
         )
         point_inclusion.add_to_prog(prog=prog)
         point_inclusion.add_anchor_bound_to_prog(prog=prog)

    def _create_lagrangian_degrees_cbf_synthesis(
        self,
        cbf_index: int,
    ) -> Tuple[
        CompatibilityLagragianDegrees,
        UnsafeRegionExclusionLagrangianDegrees
    ]:
        """
        This is a function that creates the lagrangian degrees for synthesis of
        all cbfs. Henece, some of the input and output arguments are optional.
        When synthesiszing the first cbf:
            -compatibility_lagrangians does not need deactivated cbfs.
            -ball_inclusion_lagrangians are needed.
        When synthesiszing the other cbfs:
            -compatibility_lagrangians needs deactivated cbfs.
            -ball_inclusion lagrangians are not needed.
        """
        assert len(self.compatible_lambda_y_x_degrees) == self.nu
        if cbf_index == 0:
            compatible_deact_h_lagrangian_degree = None
        else:
            compatible_deact_h_lagrangian_degree = [
                Degree(x=self.compatible_deact_cbf_x_degree[i], y=2, c=0)
                for i in range(cbf_index)
            ]

        (
            compatibility_lagrangian_degrees
        ) = CompatibilityLagragianDegrees(
            lambda_y=[
                Degree(x=self.compatible_lambda_y_x_degrees[i], y=0, c=0)
                for i in range(self.nu)
            ],
            xi_y=Degree(x=self.compatible_xi_y_x_degree, y=0, c=0),
            rho_minus_V=Degree(
                x=self.compatible_rho_minus_V_x_degree, y=2, c=0
                ),
            h=Degree(x=self.compatible_h_x_degree, y=2, c=0),
            deactivated_h=compatible_deact_h_lagrangian_degree,
            state_eq_constraints=(
                None if self.state_eq_constraints is None
                else [
                    Degree(x=self.state_eq_x_degrees[i], y=2, c=0)
                    for i in range(len(self.state_eq_x_degrees))
                ]
            )
        )
        safety_lagrangian_degree = UnsafeRegionExclusionLagrangianDegrees(
            unsafe_polys=[
                Degree(x=self.safety_unsafe_polys_x_degree[i], y=0, c=0)
                for i in range(len(self.safety_unsafe_polys_x_degree))
            ],
            h=Degree(x=self.safety_h_x_degree, y=0, c=0)
        )
        return (
            compatibility_lagrangian_degrees,
            safety_lagrangian_degree
        )

    def _create_ball_inclusion_lagrangian_degree(
        self
        ) -> BallInclusionLagrangianDegree:
        ball_inclusion_lagrangian_degree = BallInclusionLagrangianDegree(
            r_minus_xTx=Degree(x=self.cbf_ball_inclusion_ball_x_degree, y=0, c=0),
            h=Degree(x=self.cbf_ball_inclusion_cbf_x_degree, y=0, c=0)
        )
        return ball_inclusion_lagrangian_degree

    def _evalueate_cbf_at_points(
        self,
        cbf: sym.Polynomial,
        evaluate_at_points: np.ndarray
    ) -> np.ndarray:
        cbf_at_points = cbf.EvaluateIndeterminates(
            indeterminates=self.x,
            indeterminates_values=evaluate_at_points.T
        )
        return cbf_at_points
    
    def search_lagrangian_given_cbf(
        self,
        cbf: sym.Polynomial,
        deact_cbfs: Optional[np.ndarray],
        kappah: float,
        compatible_lagragian_degree: CompatibilityLagragianDegrees,
        ball_inclusion_lagrangian_degree: Optional[BallInclusionLagrangianDegree],
        safety_lagrangian_degree: UnsafeRegionExclusionLagrangianDegrees,
        lagrangian_coeff_tol: Optional[float],
        solver_id: Optional[solvers.SolverId],
        solver_options: Optional[solvers.SolverOptions]
    ) -> Tuple[
            Optional[CompatibilityLagragians], 
            Optional[BallInclusionLagrangian], 
            Optional[UnsafeRegionExclusionLagrangians]
        ]:
        # initialize the program
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)

        # add cbf compatibility constraint
        compatible_lagrangians = compatible_lagragian_degree.to_lagrangians(
            prog=prog,
            x=self.x_set,
            y=self.y_set,
        )
        
        xi, lambda_mat = self._calc_xi_lambda_for_cbf_synthesis(
            cbf=cbf,
            kappa_h=kappah
        )
        self._add_compatibility_constraints(
            prog=prog,
            cbf=cbf,
            lambda_mat=lambda_mat,
            xi=xi,
            deactivated_h=deact_cbfs,
            lagrangians=compatible_lagrangians
        )

        # add ball inclusion constraint if this is the first cbf
        if ball_inclusion_lagrangian_degree is not None:
            assert deact_cbfs is None
            (
                ball_inclusion_lagrangians
            ) = ball_inclusion_lagrangian_degree.to_lagrangians(
                prog=prog,
                x=self.x_set
            )
            self._add_cbf_ball_inclusion_constraint(
                prog=prog,
                cbf=cbf,
                ball_inclusion_lagrangian=ball_inclusion_lagrangians
            )

        # add unsafe region exclusion constraint
        unsafe_exclusion_lagrangians = safety_lagrangian_degree.to_lagrangians(
            prog=prog,
            x=self.x_set
        )
        self._add_cbf_safety_constraint(
            prog=prog,
            cbf=cbf,
            unsafe_lagrangians=unsafe_exclusion_lagrangians
        )

        # solve the program
        result = solve_with_id(
            prog=prog,
            solver_id=solver_id,
            solver_options=solver_options
            )

        if result.is_success():
            compatible_lagrangian_reults = compatible_lagrangians.get_results(
                result=result,
                coefficient_tol=lagrangian_coeff_tol
            )
            if ball_inclusion_lagrangian_degree is not None:
                (
                    ball_inclusion_lagrangian_results
                ) = ball_inclusion_lagrangians.get_results(
                    result=result,
                    coefficient_tol=lagrangian_coeff_tol
                )
            else:
                ball_inclusion_lagrangian_results = None
            safety_lagrangian_results = unsafe_exclusion_lagrangians.get_results(
                result=result,
                coefficient_tol=lagrangian_coeff_tol
            )
            return (
                compatible_lagrangian_reults,
                ball_inclusion_lagrangian_results,
                safety_lagrangian_results
            )
        else:
            return (None, None, None)

    def search_cbf_given_lagrangian(
        self,
        deact_cbfs: Optional[np.ndarray],
        kappah: float,
        cbf_degree: int,
        compatible_lagrangian_degree: CompatibilityLagragianDegrees,
        ball_inclusion_lagrangian_degree: Optional[BallInclusionLagrangianDegree],
        safety_lagrangian_degree: UnsafeRegionExclusionLagrangianDegrees,
        compatible_lagrangians: CompatibilityLagragians,
        ball_inclusion_lagrangians: Optional[BallInclusionLagrangian],
        safety_lagrangians: UnsafeRegionExclusionLagrangians,
        points_to_include: np.ndarray,
        weights_to_include: np.ndarray,
        anchor_points: Optional[np.ndarray],
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        cbf_coeff_tol: Optional[float],
        solver_id: Optional[solvers.SolverId],
        solver_options: Optional[solvers.SolverOptions],
        back_off_scale: Optional[BackoffScale]
    ) -> Optional[sym.Polynomial]:
        # initialize the program
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)

        cbf_unsolved = prog.NewFreePolynomial(self.x_set, cbf_degree)

        # add cbf compatibility constraint
        xi, lambda_mat = self._calc_xi_lambda_for_cbf_synthesis(
            cbf=cbf_unsolved,
            kappa_h=kappah
        )

        # create the compatible lagragians for cbf synthesis and
        # add the compatibility constraint
        if deact_cbfs is not None:
           assert compatible_lagrangian_degree.deactivated_h is not None
        else:
            assert compatible_lagrangian_degree.deactivated_h is None
        (
            compatible_lagrangians_synthesis
        ) = compatible_lagrangian_degree.to_lagrangians(
            prog=prog,
            x=self.x_set,
            y=self.y_set,
            lagrangian_lambda_y=compatible_lagrangians.lambda_y,
            lagrangian_xi_y=compatible_lagrangians.xi_y,
            lagrangian_h=compatible_lagrangians.h
        )

        self._add_compatibility_constraints(
            prog=prog,
            cbf=cbf_unsolved,
            lambda_mat=lambda_mat,
            xi=xi,
            deactivated_h=deact_cbfs,
            lagrangians=compatible_lagrangians_synthesis
        )

        # add ball inclusion constraint if this is the first cbf
        if ball_inclusion_lagrangian_degree is not None:
            assert deact_cbfs is None
            (
                ball_inclusion_lagrangian_synthesis
            ) = ball_inclusion_lagrangian_degree.to_lagrangians(
                prog=prog,
                x=self.x_set,
                lagrangian_h=ball_inclusion_lagrangians.h,
            )
            self._add_cbf_ball_inclusion_constraint(
                prog=prog,
                cbf=cbf_unsolved,
                ball_inclusion_lagrangian=ball_inclusion_lagrangian_synthesis
            )
        
        # add unsafe region exclusion constraint
        (
            unsafe_exclusion_lagrangian_synthesis
        ) = safety_lagrangian_degree.to_lagrangians(
            prog=prog,
            x=self.x_set,
            h_lagrangian=safety_lagrangians.h
        )
        self._add_cbf_safety_constraint(
            prog=prog,
            cbf=cbf_unsolved,
            unsafe_lagrangians=unsafe_exclusion_lagrangian_synthesis
        )

        # add point inclusion constraint
        self._add_cbf_point_inclusion_constraint(
            prog=prog,
            cbf=cbf_unsolved,
            points_to_include=points_to_include,
            weights_to_include=weights_to_include,
            anchor_points=anchor_points,
            anchor_bounds=anchor_bounds
        )

        # solve the program
        result = solve_with_id(
            prog=prog,
            solver_id=solver_id,
            solver_options=solver_options,
            backoff_rel_scale=(
                back_off_scale.rel 
                if back_off_scale is not None 
                else None),
            backoff_abs_scale=(
                back_off_scale.abs 
                if back_off_scale is not None 
                else None)
            )

        if result.is_success():
            cbf_result = get_polynomial_result(
                result=result,
                p=cbf_unsolved,
                coefficient_tol=cbf_coeff_tol
            )
            return cbf_result
        else:
            return None
    
    def synthesis_first_cbf(
        self,
        cbf_init: sym.Polynomial,
        points_to_include: np.ndarray,
        weights_to_include: np.ndarray,
        anchor_points: Optional[np.ndarray],
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        max_iter: int,
        *,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coeff_tol: Optional[float] = None,
        cbf_coeff_tol: Optional[float] = None,
        back_off_scale: Optional[List[BackoffScale]]=None
    ) -> Optional[sym.Polynomial]:
        
        # create lagrangian degrees:
        # (1) create ball inclusion lagrangian degree
        (
            ball_inclusion_lagrangian_degree
        ) = self._create_ball_inclusion_lagrangian_degree()
        # (2) create compatibility and unsafe exclusion lagrangian degrees
        (
            compatible_lagrangian_degree,
            unsafe_exclusion_lagrangian_degree
        ) = self._create_lagrangian_degrees_cbf_synthesis(
            cbf_index=0
        )
        
        # start bilinear alternation:
        iter_count = 1
        cbf = cbf_init
        if back_off_scale is not None:
            assert len(back_off_scale) == max_iter
        else:
            back_off_scale = [None]*max_iter
        while(iter_count <= max_iter):
            (
                compatible_lagrangians,
                ball_inclusion_lagrangians,
                unsafe_exclusion_lagrangians
            ) = self.search_lagrangian_given_cbf(
                cbf=cbf,
                deact_cbfs=None,
                kappah=self.kappah[0],
                compatible_lagragian_degree=compatible_lagrangian_degree,
                ball_inclusion_lagrangian_degree=ball_inclusion_lagrangian_degree,
                safety_lagrangian_degree=unsafe_exclusion_lagrangian_degree,
                lagrangian_coeff_tol=lagrangian_coeff_tol,
                solver_id=solver_id,
                solver_options=solver_options
            )

            assert compatible_lagrangians is not None
            assert ball_inclusion_lagrangians is not None
            assert unsafe_exclusion_lagrangians is not None

            cbf_updated = self.search_cbf_given_lagrangian(
                deact_cbfs=None,
                kappah=self.kappah[0],
                cbf_degree=self.cbf_x_degrees[0],
                compatible_lagrangian_degree=compatible_lagrangian_degree,
                ball_inclusion_lagrangian_degree=ball_inclusion_lagrangian_degree,
                safety_lagrangian_degree=unsafe_exclusion_lagrangian_degree,
                compatible_lagrangians=compatible_lagrangians,
                ball_inclusion_lagrangians=ball_inclusion_lagrangians,
                safety_lagrangians=unsafe_exclusion_lagrangians,
                points_to_include=points_to_include,
                weights_to_include=weights_to_include,
                anchor_points=anchor_points,
                anchor_bounds=anchor_bounds,
                cbf_coeff_tol=cbf_coeff_tol,
                back_off_scale=back_off_scale[iter_count-1],
                solver_id=solver_id,
                solver_options=solver_options
            )

            assert cbf_updated is not None

            cbf_evaluation = self._evalueate_cbf_at_points(
                cbf=cbf_updated,
                evaluate_at_points=points_to_include
            )
            print_values = cbf_evaluation
            print(f"iteration: {iter_count}")
            print(f"[h0](points_to_include): {print_values}")

            cbf = cbf_updated
            iter_count += 1

        return cbf

    def synthesis_other_cbf(
        self,
        cbf_init: sym.Polynomial,
        cbf_index: int,
        deact_cbfs: np.ndarray,
        points_to_include: np.ndarray,
        weights_to_include: np.ndarray,
        anchor_points: Optional[np.ndarray],
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        max_iter: int,
        *,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coeff_tol: Optional[float] = None,
        cbf_coeff_tol: Optional[float] = None,
        back_off_scale: Optional[List[BackoffScale]]=None
    ) -> Optional[sym.Polynomial]:
        assert cbf_index > 0
        assert deact_cbfs.shape[0] == cbf_index
        # create lagrangian degrees:
        # (1) this is not for the first cbf, hence, we do not need ball inclusion
        ball_inclusion_lagrangian_degree = None
        # (2) create compatibility and unsafe exclusion lagrangian degrees
        (
            compatible_lagrangian_degree,
            unsafe_exclusion_lagrangian_degree
        ) = self._create_lagrangian_degrees_cbf_synthesis(cbf_index=cbf_index)
        
        # start bilinear alternation:
        iter_count = 1
        cbf = cbf_init
        if back_off_scale is not None:
            assert len(back_off_scale) == max_iter
        else:
            back_off_scale = [None]*max_iter
        while(iter_count <= max_iter):
            (
                compatible_lagrangians,
                ball_inclusion_lagrangians,
                unsafe_exclusion_lagrangians
            ) = self.search_lagrangian_given_cbf(
                cbf=cbf,
                deact_cbfs=deact_cbfs,
                kappah=self.kappah[cbf_index],
                compatible_lagragian_degree=compatible_lagrangian_degree,
                ball_inclusion_lagrangian_degree=ball_inclusion_lagrangian_degree,
                safety_lagrangian_degree=unsafe_exclusion_lagrangian_degree,
                lagrangian_coeff_tol=lagrangian_coeff_tol,
                solver_id=solver_id,
                solver_options=solver_options
            )

            assert compatible_lagrangians is not None
            assert ball_inclusion_lagrangians is None
            assert unsafe_exclusion_lagrangians is not None

            assert is_sos(compatible_lagrangians.h)
            # print(compatible_lagrangians.h)

            cbf_updated = self.search_cbf_given_lagrangian(
                deact_cbfs=deact_cbfs,
                kappah=self.kappah[cbf_index],
                cbf_degree=self.cbf_x_degrees[cbf_index],
                compatible_lagrangian_degree=compatible_lagrangian_degree,
                ball_inclusion_lagrangian_degree=ball_inclusion_lagrangian_degree,
                safety_lagrangian_degree=unsafe_exclusion_lagrangian_degree,
                compatible_lagrangians=compatible_lagrangians,
                ball_inclusion_lagrangians=ball_inclusion_lagrangians,
                safety_lagrangians=unsafe_exclusion_lagrangians,
                points_to_include=points_to_include,
                weights_to_include=weights_to_include,
                anchor_points=anchor_points,
                anchor_bounds=anchor_bounds,
                cbf_coeff_tol=cbf_coeff_tol,
                solver_id=solver_id,
                solver_options=solver_options,
                back_off_scale=back_off_scale[iter_count-1]
            )

            assert cbf_updated is not None
            cbf_evaluation = self._evalueate_cbf_at_points(
                cbf=cbf_updated,
                evaluate_at_points=points_to_include
            )
            print_values = cbf_evaluation
            print(f"iteration: {iter_count}")
            print(f"[h{cbf_index}](points_to_include): {print_values}")

            cbf = cbf_updated
            iter_count += 1
        
        return cbf

    def remove_included_points(
        self,
        included_points: np.ndarray, 
        points_inclusion_weights: np.ndarray, 
        solved_cbf: sym.Polynomial,
    ) -> Tuple[
        np.ndarray, np.ndarray
        ]:
        new_points_to_include = np.array([])
        new_weights_to_include = np.array([])
        for i in range(included_points.shape[0]):
            cbf_at_point = solved_cbf.EvaluateIndeterminates(
                indeterminates=self.x,
                indeterminates_values=included_points[i]
            )
            if cbf_at_point < 0:
                new_points_to_include = np.append(
                    new_points_to_include, included_points[i]
                )
                new_weights_to_include = np.append(
                    new_weights_to_include, points_inclusion_weights[i]
                )
            new_points_to_include = new_points_to_include.reshape(-1, self.x.shape[0])
        return new_points_to_include, new_weights_to_include

       