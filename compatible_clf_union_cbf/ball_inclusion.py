"""
This part verifies the following:
Given a set of polynomials h₁, h₂ , ..., hₙ 
verify that there exists a radius r>0, a polynomial
from above whose 0-super level set that
includes the ball B(0, r).

The inner block of the code is to verify that:
for a given polinomial h and a radius r,
the 0-super level set of h includes the ball B(0, r).
Formulated into an SOS program, that is:

-1-s_1(x)*(r - x^Tx) + s_2(x)*h(x) is SOS.

where s_1 and s_2 are SOS polynomials.
"""
from dataclasses import dataclass
from typing import List, Optional
from typing_extensions import Self
import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import (
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    solve_with_id
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
class BallInclusion:
    r_start: float
    r_lower_bound: float
    polys: np.ndarray
    x: np.ndarray

    def create_inclusion_degrees(
        self,
        # this array should have size (n_h, 2) that
        # represents the degree of x for each ball inclusion
        # lagragian.
        degrees: np.ndarray
    )-> List[BallInclusionLagrangianDegree]:
        assert degrees.shape[1] == 2
        assert degrees.shape[0] == self.polys.shape[0]
        return [
            BallInclusionLagrangianDegree(
                r_minus_xTx=Degree(x=degrees[i][0], y=0, c=0),
                h=Degree(x=degrees[i][1], y=0, c=0)
            )
            for i in range(self.polys.shape[0])
        ]
    
    def ball_in_superlevel_set(
        self,
        r: float,
        poly: sym.Polynomial,
        lagrangian_degree: BallInclusionLagrangianDegree
    ) -> bool:
        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.x)
        prog.AddIndeterminates(x_set)
        lagrangian = lagrangian_degree.to_lagrangians(prog, x_set)
        prog.AddSosConstraint(sym.Polynomial(
            -1 - lagrangian.r_minus_xTx*(r - self.x.dot(self.x)) + lagrangian.h*poly
            )
        )
        result = solve_with_id(prog)
        return result.is_success()
    
    def ball_inclusion_verification(
        self,
        degrees: np.ndarray
    ) -> bool:
        current_r = self.r_start
        ball_inclusion_lagragian_degrees = self.create_inclusion_degrees(degrees)

        while(current_r >= self.r_lower_bound):
            unsatisfied_num = 0
            for i in range(self.polys.shape[0]):
                current_inclusion = self.ball_in_superlevel_set(
                    current_r, self.polys[i], ball_inclusion_lagragian_degrees[i]
                )
                if current_inclusion:
                    return True
                else:
                    unsatisfied_num += 1
            if unsatisfied_num == self.polys.shape[0]:
                current_r = current_r/2

        return False
