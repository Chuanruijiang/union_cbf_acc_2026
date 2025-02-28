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
    to_lagrangian_impl
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
        control_Au: Optional[np.ndarray] = None,
        control_bu: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        If we call this function, then at least we should have the
        ball polynomial as the deactivated polynomial. Hence, the
        deactivated member in this class should be not None. Also,
        we consider CLF, so the activated member should be more than 1.
        This function finds the number of activated cbfs and the
        number of deactivated cbfs. Then, it will construct all the
        y_sets and c_sets for the compatibility verifcation lagrangians,
        packed them into two ndarrays and return them.
        """
        assert self.deactivated is not None
        assert self.deactivated.shape[0] >= 1
        assert self.activated.shape[0] > 1
        n_activated_cbfs = self.activated.shape[0] - 1,
        n_deactivated_cbfs = self.deactivated.shape[0] - 1
        y_i_size = 2
        if with_control_input_limits:
            assert control_Au is not None
            assert control_bu is not None
            y_i_size += control_bu.shape[0]
        ## write down the y_sets and c_sets that you wanted to generate. 



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
        all_possible_activation_cases = truth_table(self.h.shape[0])
        all_polys = self.h

        if outside_ball:
            # the ball poly should be r-x^Tx
            assert ball_poly is not None
            all_possible_activation_cases = np.concatenate(
                [all_possible_activation_cases, 
                 np.zeros((all_possible_activation_cases.shape[0], 1))],
                axis=1
            )
            all_polys = np.concatenate([all_polys, np.array([ball_poly])])
        
        if with_clf:
            # the clf poly should be rho-V(x)
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
            assert isinstance(kappa, float)
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

