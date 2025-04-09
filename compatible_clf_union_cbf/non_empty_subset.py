from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union
from typing_extensions import Self
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import (
    truth_table,
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    solve_with_id,
)

"""
In this section, we want to get all the non-empty subsets of union CBFs 
ouside the ball.

Given CBFs h1,...,hn and r. Let N = {1,...,n}. We want get the following:
1. All possible set with form: {x| ∃ i∈N hi(x)≥0, r-xᵀx < 0}
2. Exclude the empty set among all the sets above
3. Return the rest of the sets.

All the three steps above are finally integrated into the function 
"get_non_empty_region(h, r)"
"""



"""
The following module will be used to check whether a subset with form
{x| ∃ i∈N hi(x)≥0, r-xᵀx < 0}
is empty

Let's use the following example to show how to verify the emptiness:
if a subset is: {x|h1(x)≥0, h2(x)<0, r-xᵀx < 0}
checking the emptiness of this subset is equivalent to checking
whether the following set is empty:
{(x,c)| h1(x)≥ 0, c1^2h2(x)=-1, c2^2(r-xᵀx)=-1}
hence, we can use the following SOS program to check the emptiness:

-1 - s1(x,c)*h1(x) - s2(x,c)*(c1^2h2(x)+1) - s3(x,c)*(c2^2(r-xᵀx)+1) is SOS

where s1 is SOS, s2 and s3 are free polynomials
Noted that c variable is not optional for the lagrangians since we always 
have r - xᵀx < 0 in the subset.

"""
@dataclass
class SubsetEmptinessLagrangian:
    stable_region: sym.Polynomial
    activated_cbf: np.ndarray
    deactivated_cbf: Optional[np.ndarray]
    ball_outside: sym.Polynomial

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        lagrangian_stable_region_result = get_polynomial_result(
            result, self.stable_region, coefficient_tol
        )
        lagrangian_activaed_cbf_result = get_polynomial_result(
            result, self.activated_cbf, coefficient_tol
        )
        lagrangian_deactivated_cbf_result = (
            None
            if self.deactivated_cbf is None
            else get_polynomial_result(
                result, self.deactivated_cbf, coefficient_tol
                )
        )
        lagrangian_ball_outside_result = get_polynomial_result(
            result, self.ball_outside, coefficient_tol
        )
        return SubsetEmptinessLagrangian(
            stable_region=lagrangian_stable_region_result,
            activated_cbf=lagrangian_activaed_cbf_result,
            deactivated_cbf=lagrangian_deactivated_cbf_result,
            ball_outside = lagrangian_ball_outside_result
        )

@dataclass
class SubsetEmptinessLagrangianDegree:
    stable_region: Degree
    activated_cbf: List[Degree]
    deactivated_cbf: Optional[List[Degree]]
    ball_outside: Degree

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variable,
        c: sym.Variable,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_stable_region: Optional[sym.Polynomial] = None,
        lagrangian_activated_cbf: Optional[np.ndarray] = None,
        lagrangian_deactivated_cbf: Optional[np.ndarray] = None,
        lagrangian_ball_outside: Optional[sym.Polynomial] = None,
    ):
        
        lagrangian_stable_region = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=c,
            sos_type=sos_type,
            degree=self.stable_region,
            is_sos=True,
            lagrangian=lagrangian_stable_region,
        )
        lagrangian_activated_cbf = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=c,
            sos_type=sos_type,
            degree=self.activated_cbf,
            is_sos=True,
            lagrangian=lagrangian_activated_cbf,
        )
        lagrangian_deactivated_cbf = (
            None
            if self.deactivated_cbf is None
            else to_lagrangian_impl(
                prog=prog,
                x=x,
                y=None,
                c=None,
                sos_type=sos_type,
                degree=self.deactivated_cbf,
                is_sos=False,
                lagrangian=(
                    lagrangian_deactivated_cbf 
                    if lagrangian_deactivated_cbf is not None 
                    else None
                    ),
            )
        )
        lagrangian_ball_outside = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=None,
            sos_type=sos_type,
            degree=self.ball_outside,
            is_sos=False,
            lagrangian=lagrangian_ball_outside,
        )

        return SubsetEmptinessLagrangian(
            stable_region=lagrangian_stable_region,
            activated_cbf=lagrangian_activated_cbf,
            deactivated_cbf=lagrangian_deactivated_cbf,
            ball_outside=lagrangian_ball_outside
        )

@dataclass
class Subset:
    x: np.ndarray
    # This array of Polynomial containts all the CBFs
    # (including activated and deactivated CBFs)
    cbfs: np.ndarray
    rho_minus_clf: sym.Polynomial
    # this is the radius of the ball
    # note that it is NOT SQUARED
    ball_radius: float
    # a 1D array showing the activated CBF indecies.
    # for example [1, 0, 0] means h1≥0, h2<0, h3<0
    activation_index: np.ndarray 

    def _create_emtiness_lagrangian_degrees(
        self,
        lagrangian_x_degree: int,
        lagrangian_c_degree: int,
    ) -> SubsetEmptinessLagrangianDegree:
        num_activated_cbf = np.sum(self.activation_index)
        num_deactivated_cbf = self.cbfs.shape[0] - num_activated_cbf
        lagrangian_degrees = SubsetEmptinessLagrangianDegree(
            stable_region=Degree(x=lagrangian_x_degree, y=0, c=lagrangian_c_degree),
            activated_cbf=[
                Degree(x=lagrangian_x_degree, y = 0, c=lagrangian_c_degree)
                ] * num_activated_cbf,
            deactivated_cbf=(
                [Degree(x=lagrangian_x_degree, y = 0, c=0)]* num_deactivated_cbf 
                if num_deactivated_cbf > 0
                else None
            ),
            ball_outside=Degree(x=lagrangian_x_degree, y=0, c=0),
        )
        return lagrangian_degrees

    def is_empty(
        self,
        lagrangian_x_degree: int,
        lagrangian_c_degree: int,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> bool:
        
        lagrangian_degree = self._create_emtiness_lagrangian_degrees(
            lagrangian_x_degree=lagrangian_x_degree,
            lagrangian_c_degree=lagrangian_c_degree,
        )
        c_size = 1

        # initial check:
        assert self.cbfs.shape[0] == self.activation_index.shape[0]
        assert len(lagrangian_degree.activated_cbf) == np.sum(self.activation_index)
        if lagrangian_degree.deactivated_cbf is not None:
            assert self.cbfs.shape[0] - np.sum(self.activation_index) == len(
                lagrangian_degree.deactivated_cbf
            )
            c_size = len(lagrangian_degree.deactivated_cbf) + 1
        else:
            assert self.cbfs.shape[0] == np.sum(self.activation_index)
        
        # emptiness check
        prog = solvers.MathematicalProgram()
        c = sym.MakeVectorContinuousVariable(c_size,"c")
        xc = np.concatenate([self.x, c], axis=0)
        x_set = sym.Variables(self.x)
        c_set = sym.Variables(c)
        c_squared_poly = np.array(
                [sym.Polynomial(sym.Monomial(c[i], 2)) for i in range(c.shape[0])]
            )
        xc_set = sym.Variables(xc)
        prog.AddIndeterminates(xc_set)

        lagrangians = lagrangian_degree.to_lagrangians(
            prog=prog,
            x=x_set,
            c=c_set,
            sos_type=sos_type,
        )
        
        lagrangian_stable_region = lagrangians.stable_region*self.rho_minus_clf
        lagrangian_activated_cbf = lagrangians.activated_cbf.dot(
            self.cbfs[self.activation_index == 1]
            )
        lagrangian_deactivated_cbf = sym.Polynomial(0)
        if lagrangian_degree.deactivated_cbf is not None:
            deactivated_cbfs = self.cbfs[self.activation_index == 0]
            lagrangian_deactivated_cbf = lagrangians.deactivated_cbf.dot(
                c_squared_poly[:-1] * deactivated_cbfs
            ) + 1
        lagrangian_ball = lagrangians.ball_outside * (
            c_squared_poly[-1] * (self.ball_radius**2 - self.x.dot(self.x)) + 1
        )
        poly = (- sym.Polynomial(1)
                - lagrangian_stable_region
                - lagrangian_activated_cbf
                - lagrangian_deactivated_cbf
                - lagrangian_ball
                )
        prog.AddSosConstraint(poly)
        
        result = solve_with_id(prog)
        if result.is_success():
            return True
        else:
            return False
        

def get_non_empty_region(
    rho_minus_V: sym.Polynomial,
    h: np.ndarray,
    r: float, # NOT SQUARED
    x: np.ndarray,
    *,
    lagrangian_x_degree: int = 2,
    lagrangian_c_degree: int = 2,
    sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
) -> List[Subset]:
    """
    Get all the non-empty subsets of union CBFs outside the ball.
    
    Args:
        h: CBFs
        r: radius of the ball
        lagrangian_x_degree: degree of x in the lagrangian
        lagrangian_c_degree: degree of c in the lagrangian
        sos_type: type of SOS

    Returns:
        List of non-empty subsets
    """
    num_cbfs = h.shape[0]
    activation_cases = truth_table(num_cbfs)
    non_empty_subsets = []
    for activation_case in activation_cases:
        # skip the set with all CBFs deactivated
        if np.sum(activation_case) == 0:
            continue
        # create the subset
        subset = Subset(
            x=x,
            cbfs=h,
            rho_minus_clf=rho_minus_V,
            ball_radius=r,
            activation_index=activation_case,
        )
        # check the emptiness of the subset
        if not subset.is_empty(
            lagrangian_x_degree=lagrangian_x_degree,
            lagrangian_c_degree=lagrangian_c_degree,
            sos_type=sos_type,
        ):
            non_empty_subsets.append(subset)
        
    return non_empty_subsets
