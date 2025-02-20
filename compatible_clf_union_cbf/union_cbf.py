from dataclasses import dataclass
from typing import List, Optional
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from compatible_clf_union_cbf.utils import (
    truth_table,
    get_polynomial_result,
    solve_with_id,
)

from compatible_clf_union_cbf.clf_cbf import (
    XYDegree,
    _to_lagrangian_impl,
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

    activated: List[XYDegree]
    deactivated: Optional[List[XYDegree]]

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
        activated = _to_lagrangian_impl(
            prog,
            x,
            y,
            sos_type,
            is_sos=True,
            degree=self.activated,
            lagrangian=lagrangian_activated,
        )
        deactivated = (
            None
            if self.deactivated is None
            else _to_lagrangian_impl(
                prog,
                x,
                y,
                sos_type,
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


class UnionCBF:
    def __init__(self, h: np.ndarray, x: np.ndarray):
        self.h = h
        self.x = x

    def non_empty_disjoint_subsets(
        self,
        *,
        common_act_x_degree: Optional[int] = None,
        various_act_x_degrees: Optional[List[int]] = None,
        common_deact_x_degree: Optional[int] = None,
        various_deact_x_degrees: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        This function outputs all the 0-1 vector of all non-empty disjoint
        subsets of the union
        """
        all_possible_activation_cases = truth_table(self.h.shape[0])
        activated_01_vectors = np.array([])
        for i in range(1, all_possible_activation_cases.shape[0]):
            assert all_possible_activation_cases[i].shape[0] == self.h.shape[0]
            # In the case below, there are no deactivated polynomials
            if np.all(all_possible_activation_cases[i] == 1):
                activated_polys = self.h
                deactivated_polys = None
            # In the case below, there are deactivated polynomials
            else:
                activated_idx = np.where(all_possible_activation_cases[i] == 1)
                deactivated_idx = np.where(all_possible_activation_cases[i] == 0)
                activated_polys = self.h[activated_idx]
                deactivated_polys = self.h[deactivated_idx]

            current_subset = UnionSubset(
                activated=activated_polys,
                deactivated=deactivated_polys,
                variables=self.x,
            )
            emptiness_lagrangian_degrees = self._create_emptiness_lagragian_degrees(
                activated_polys=activated_polys,
                deactivated_polys=deactivated_polys,
                common_act_x_degree=common_act_x_degree,
                various_act_x_degrees=various_act_x_degrees,
                common_deact_x_degree=common_deact_x_degree,
                various_deact_x_degrees=various_deact_x_degrees,
            )

            current_subset_is_empty = current_subset.is_empty(
                emptiness_lagrangian_degrees=emptiness_lagrangian_degrees
            )
            if not current_subset_is_empty:
                activated_01_vectors = np.append(
                    activated_01_vectors, all_possible_activation_cases[i]
                )

        return activated_01_vectors.reshape(-1, self.h.shape[0])

    def _create_emptiness_lagragian_degrees(
        self,
        activated_polys: np.ndarray,
        deactivated_polys: Optional[np.ndarray],
        common_act_x_degree: Optional[int],
        various_act_x_degrees: Optional[List[int]],
        common_deact_x_degree: Optional[int],
        various_deact_x_degrees: Optional[List[int]],
    ) -> EmptinessLagrangianDegrees:
        if deactivated_polys is None:
            if common_act_x_degree is not None:
                activated_degrees = [
                    XYDegree(x=common_act_x_degree, y=0)
                    for _ in range(activated_polys.shape[0])
                ]
            elif various_act_x_degrees is not None:
                activated_degrees = [XYDegree(x=x, y=0) for x in various_act_x_degrees]
            else:
                activated_degrees = [
                    XYDegree(x=2, y=0) for _ in range(activated_polys.shape[0])
                ]
            return EmptinessLagrangianDegrees(
                activated=activated_degrees, deactivated=None
            )
        else:
            if common_act_x_degree is not None:
                activated_degrees = [
                    XYDegree(x=common_act_x_degree, y=2)
                    for _ in range(activated_polys.shape[0])
                ]
            elif various_act_x_degrees is not None:
                activated_degrees = [XYDegree(x=x, y=2) for x in various_act_x_degrees]
            else:
                activated_degrees = [
                    XYDegree(x=2, y=2) for _ in range(activated_polys.shape[0])
                ]

            if common_deact_x_degree is not None:
                deactivated_degrees = [
                    XYDegree(x=common_deact_x_degree, y=0)
                    for _ in range(deactivated_polys.shape[0])
                ]
            elif various_deact_x_degrees is not None:
                deactivated_degrees = [
                    XYDegree(x=x, y=0) for x in various_deact_x_degrees
                ]
            else:
                deactivated_degrees = [
                    XYDegree(x=2, y=0) for _ in range(deactivated_polys.shape[0])
                ]
            return EmptinessLagrangianDegrees(
                activated=activated_degrees, deactivated=deactivated_degrees
            )


@dataclass
class ActivationIndicator:
    """
    This class stores the list of 0-1 vectors as the activation
    indicator of the disjoint subsets in the union.
    """

    vecotor01: np.ndarray

    def to_union_subsets(
        self,
        h: np.ndarray,
        variables: np.ndarray,
    ) -> np.ndarray:
        assert len(self.vecotor01.shape) == 2
        assert self.vecotor01.shape[1] == h.shape[0]
        subsets = np.array([])
        for i in range(self.vecotor01.shape[0]):
            if np.all(self.vecotor01[i] == 1):
                subset = UnionSubset(
                    activated=h,
                    deactivated=None,
                    variables=variables,
                )
                subsets = np.append(subsets, subset)
            else:
                activated_idx = np.where(self.vecotor01[i] == 1)
                deactivated_idx = np.where(self.vecotor01[i] == 0)
                activated_polys = h[activated_idx]
                deactivated_polys = h[deactivated_idx]
                subset = UnionSubset(
                    activated=activated_polys,
                    deactivated=deactivated_polys,
                    variables=variables,
                )
                subsets = np.append(subsets, subset)
        return subsets
