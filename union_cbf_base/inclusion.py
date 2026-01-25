# """
# This script develops the tools to check whether the region of the
# union of CBFs X_c = {x | h₁(x) ≥ 0 or h₂(x) ≥ 0 or ... or hₘ(x) ≥ 0}
# excludes the unsafe region X_u.

# We provide two options:
# 1. assume the unsafe region is presented by the intersection of
#     0-super-level sets of a collection of polynomial functions.
#     such that X_u = {x | l₁(x) ≤ 0, ..., lₙ(x) ≤ 0}.
#     This assumption is valid when the unsafe region is: 
#     (1) Polyhedral: Since any polyhedron can be presented by the
#         intersection of a collection of half-spaces presented by
#         hyperplanes. In this case,  each lᵢ(x) is a linear
#         function.
#     (2) Semialgebraic set: Since any semialgebraic set can be 
#         presented by the intersection of a collection of 
#         0-super-level sets of polynomial functions.
# 2. assume the unsafe region is presented by a collection of
#     unsafe points. As long as the number of unsafe points are
#     sampled sufficiently. This assumption is valid for the following
#     cases:
#     (1) The actual unsafe region are presented by X_u = {x|f(x) ≤ 0}
#         and f(x) has some complicated forms such as exponential,
#         compositional, signed-distance or even NN.
#     (2) The unsafe region is unknown but our sensor can provide some
#         observed unsafe points from the unsafe region (e.g. Lidar).
#     In such cases, we don't need to apply more l(x) polynomails to
#     approximate the unsafe region boundary. We can directly use the 
#     CBFs h(x) polynomials to excluded the unsafe points by evaluating
#     each of the unsafe points and checking h(x) < 0.

# For assumption 1, we create SOS constraints based on P-satz to verify
# that the CBF is negative over the unsafe region.
# Assume we have a cbf h(x), we hope that: 
# for all x in the unsafe region, h(x) < 0.
# This is equivalent to verifying that the set:
# {x | -p₁(x) ≥ 0, ..., -pₙ(x) ≥ 0, h(x) ≥ 0}
# is empty. 
# Using P-satz, we have the following SOS constraint:
# -1 + (∑ᵢ sᵢ(x)pᵢ(x)) - s_{n+1}(x) * h(x) is SOS
# where s₁(x), ..., s_{n+1}(x) are SOS polynomials.

# For assumption 2, we simply evaluate all the cbf h(x) in the union over
# all the unsafe points to check whether h(x) < 0 holds for all the unsafe
# points.
# """

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
from typing_extensions import Self

import pydrake.solvers as solvers
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    solve_with_id,
)

@dataclass
class UnsafeRegionExclusionLagrangians:
    # The array of polynomials presenting lagrangains for p(x)s above
    unsafe_polys: np.ndarray
    # The lagrangian for the cbf function
    h: sym.Polynomial

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        unsafe_polys_results = get_polynomial_result(
            result, self.unsafe_polys, coefficient_tol
        )
        h_result = get_polynomial_result(result, self.h, coefficient_tol)
        return UnsafeRegionExclusionLagrangians(
            unsafe_polys=unsafe_polys_results, h=h_result
        )


@dataclass
class UnsafeRegionExclusionLagrangianDegrees:
    unsafe_polys: List[Degree]
    h: Degree

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        unsafe_poly_lagrangians: Optional[np.ndarray] = None,
        h_lagrangian: Optional[sym.Polynomial] = None,
    ) -> UnsafeRegionExclusionLagrangians:
        lagrangians_unsafe_polys = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.unsafe_polys,
            lagrangian=unsafe_poly_lagrangians,
        )
        lagrangians_h = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.h,
            lagrangian=h_lagrangian,
        )
        return UnsafeRegionExclusionLagrangians(
            unsafe_polys=lagrangians_unsafe_polys, h=lagrangians_h
        )


class UnsafeExclusion:
    def __init__(
        self,
        h: sym.Polynomial,
        x: np.ndarray,
        unsafe_polys: Optional[np.ndarray],
        unsafe_points: Optional[np.ndarray],
    ):
        self.unsafe_polys = unsafe_polys
        self.unsafe_points = unsafe_points
        self.h = h
        self.x = x

    def verify_unsafe_exclusion(
        self,
        unsafe_poly_x_degrees: List[int],
        h_x_degree: int,
        *,
        output_lagrangians: bool = False,
        coefficient_tol: Optional[float] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Union[bool, UnsafeRegionExclusionLagrangians]:
        assert self.unsafe_polys is not None
        assert len(unsafe_poly_x_degrees) == self.unsafe_polys.shape[0]

        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.x)
        prog.AddIndeterminates(x_set)

        unsafe_exclusion_degree = UnsafeRegionExclusionLagrangianDegrees(
            unsafe_polys=[
                Degree(x=unsafe_poly_x_degrees[i], y=0, c=0)
                for i in range(self.unsafe_polys.shape[0])
            ],
            h=Degree(x=h_x_degree, y=0, c=0),
        )
        unsafe_exclusion_lagrangians = unsafe_exclusion_degree.to_lagrangians(
            prog, x_set, sos_type=sos_type
        )
        self._add_unsafe_exclusion_constraint(
            prog, unsafe_exclusion_lagrangians, sos_type=sos_type
        )

        result = solve_with_id(prog)

        if output_lagrangians:
            assert result.is_success()
            return unsafe_exclusion_lagrangians.get_results(
                result=result,
                coefficient_tol=coefficient_tol
            )
        else:
            return result.is_success()

    def verify_unsafe_point_exclusion(self):
        """
        checking whether h(x) < 0 for all the unsafe points
        """
        assert self.unsafe_points is not None
        assert len(self.unsafe_points.shape) == 2
        h_values = self.h.EvaluateIndeterminates(
            indeterminates=self.x,
            indeterminates_values=self.unsafe_points.T
        )
        binary_flags = (h_values < 0)
        return np.all(binary_flags==1)

    def _add_unsafe_exclusion_constraint(
            self,
            prog: solvers.MathematicalProgram,
            unsafe_exclusion_lagrangians: UnsafeRegionExclusionLagrangians,
            *,
            sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        ) -> sym.Polynomial:
            """
            This function adds the following SOS constraint for safe exclusion:
            -1 + (∑ᵢ sᵢ(x)pᵢ(x)) - s_{n+1}(x) * h(x) is SOS
            where s₁(x), ..., s_{n+1}(x) are SOS polynomials.
            """
            h = self.h
            poly = -sym.Polynomial(1)
            poly += unsafe_exclusion_lagrangians.unsafe_polys.dot(self.unsafe_polys)
            poly -= unsafe_exclusion_lagrangians.h * h
            prog.AddSosConstraint(poly, sos_type)
            return poly