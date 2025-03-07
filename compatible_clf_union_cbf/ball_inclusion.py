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
)

"""
This file contains the verification of the ball inclusion in the interior
of a 0-super-level set of a polynomial function. We wanted to verify that:
for all x in the ball B_r(0) = {xᵀx <= r}, h(x) > 0.
This is equivalent to verifying that the set:
{ x | xᵀx <= r, h(x) <= 0 }
is empty. We can use the S-procedure to verify this. We can write the above

-1 - s_1(x) * (r - xᵀx) + s_2(x)h(x) is SOS 
where s_1(x) and s_2(x) are SOS polynomials.
"""


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


class BallInclusion:
    def __init__(
        self,
        radius: float,
        h: sym.Polynomial,
        x: sym.Variables
    ):
        self.radius = radius
        self.h = h
        self.x = x

    def add_ball_inclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        ball_inclusion_lagrangian: BallInclusionLagrangian,
        sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> sym.Polynomial:
        x = self.x
        r = self.radius
        h = self.h
        s1 = ball_inclusion_lagrangian.r_minus_xTx
        s2 = ball_inclusion_lagrangian.h
        poly = -1 - s1 * (r - x.dot(x)) + s2 * h
        prog.AddSosConstraint(poly, sos_type)
        return poly
    

    def verify_ball_inclusion(
        self,
        ball_x_degree: int,
        h_x_degree: int,
        sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> bool:
        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.x)
        prog.AddIndeterminates(x_set)

        ball_inclusion_degree = BallInclusionLagrangianDegree(
            r_minus_xTx=Degree(x = ball_x_degree, y= 0, c=0),
            h=Degree(x = h_x_degree, y= 0, c=0)
        )
        ball_inclusion_lagrangian = ball_inclusion_degree.to_lagrangians(
            prog, x_set, sos_type=sos_type
        )

        self.add_ball_inclusion_constraint(
            prog, ball_inclusion_lagrangian, sos_type=sos_type
        )
        result = solve_with_id(prog)
        return result.is_success()

