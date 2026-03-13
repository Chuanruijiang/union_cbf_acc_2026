# In this script, we develop the class and methods to verify emptiness
# of a subset. For convenience, we de-couple this script from the information
# of relative degrees, system dynamics, and alpha parameters.. etc.. Hence,
# what this script verifies is the emptiness of the following kind of set:
# 
#  S = {x| an array of polynomials ≥ 0; an array of polynomials < 0}.
# 
# Let's use an example to show how to verify the emptiness:
# if a subset is: {x|[h1(x)≥ 0, h_2(x)≥ 0]; [h3(x)<0, h4(x)<0]}
# checking the emptiness of this subset is equivalent to checking
# whether the following set is empty:
# {(x,c)| [h1(x)≥0, h2(x)≥0]; [c1²h3(x)=-1, c2²h4(x)=-1]}
# hence, we can use the following SOS program to check the emptiness:

# -1 - s(x, c)ᵀ[h1(x); h2(x)]
#    - q(x, c)ᵀ[c1²h3(x)+1; c2²h4(x)+1] is SOS

# where s(x, c) are SOS and q(x, c) are free polynomials.

# Since we may also have a set with only activated polynomial groups:
#              S_N = {x|h1(x)≥ 0,...,hn(x)≥ 0}
# Hence, the variables c are optional.

# In the following codes, "activated poly" referes to a polynomial p(x)≥0, 
# while "deactivated" refers to a polynomial p(x)<0.

from dataclasses import dataclass
import numpy as np
from typing import List, Optional
from typing_extensions import Self
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from union_cbf_base.utils import (
    truth_table,
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    solve_with_id,
)

@dataclass
class SubsetEmptinessLagrangian:
    activated_polys: np.ndarray
    deactivated_polys: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        lagrangian_activated_polys_result = get_polynomial_result(
            result, self.activated_polys, coefficient_tol
        )
        lagrangian_deactivated_polys_result = (
            None
            if self.deactivated_polys is None
            else get_polynomial_result(
                result, self.deactivated_polys, coefficient_tol
                )
        )
        return SubsetEmptinessLagrangian(
            activated_polys=lagrangian_activated_polys_result,
            deactivated_polys=lagrangian_deactivated_polys_result
        )

@dataclass
class SubsetEmptinessLagrangianDegree:
    activated_polys: List[Degree]
    deactivated_polys: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variable,
        c: Optional[sym.Variable],
        *,
        lagrangian_activated_polys: Optional[np.ndarray] = None,
        lagrangian_deactivated_polys: Optional[np.ndarray] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        # if there is no activated cbfs in the current subset,
        # then the c variable should also be None.
        if self.deactivated_polys is None:
            assert c is None
        
        lagrangian_activated_polys = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=c,
            sos_type=sos_type,
            degree=self.activated_polys,
            is_sos=True,
            lagrangian=lagrangian_activated_polys,
        )
        lagrangian_deactivated_polys = (
            None
            if self.deactivated_polys is None
            else to_lagrangian_impl(
                prog=prog,
                x=x,
                y=None,
                c=c,
                sos_type=sos_type,
                degree=self.deactivated_polys,
                is_sos=False,
                lagrangian=(
                    lagrangian_deactivated_polys 
                    if lagrangian_deactivated_polys is not None 
                    else None
                    ),
            )
        )

        return SubsetEmptinessLagrangian(
            activated_polys=lagrangian_activated_polys,
            deactivated_polys=lagrangian_deactivated_polys
        )

@dataclass
class Subset:
    # x is the variable array for all the polynomials in the subset
    x: np.ndarray
    # all_polys is the array of all the polynomials in the subset
    all_polys: np.ndarray
    # activation_index is an array of 0-1 that indicates which 
    # polynomials in the all_polys array are activated (1) or 
    # deactivated (0)
    activation_index: np.ndarray

    def _create_emtiness_lagrangian_degrees(
        self,
        lagrangian_x_degree: int,
        lagrangian_act_c_degree: int,
        lagrangian_deact_c_degree: int
    ) -> SubsetEmptinessLagrangianDegree:
        num_activated_polys = np.sum(self.activation_index)
        
        # check whether the number of activated CBFs is valid:
        assert num_activated_polys > 0
        assert num_activated_polys <= self.all_polys.shape[0]
        
        num_deactivated_polys = self.all_polys.shape[0] - num_activated_polys
        lagrangian_degrees = SubsetEmptinessLagrangianDegree(
            activated_polys=[Degree(
                x=lagrangian_x_degree,
                y = 0,
                c=(lagrangian_act_c_degree
                   if num_deactivated_polys > 0
                   else 0)
                )] * num_activated_polys,
            deactivated_polys=([Degree(
                x=lagrangian_x_degree,
                y = 0,
                c=lagrangian_deact_c_degree
                )] * num_deactivated_polys
                if num_deactivated_polys > 0
                else None
            )
        )
        return lagrangian_degrees

    def is_empty(
        self,
        lagrangian_x_degree: int,
        lagrangian_act_c_degree: int,
        lagrangian_deact_c_degree: int,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> bool:
        
        lagrangian_degree = self._create_emtiness_lagrangian_degrees(
            lagrangian_x_degree=lagrangian_x_degree,
            lagrangian_act_c_degree=lagrangian_act_c_degree,
            lagrangian_deact_c_degree=lagrangian_deact_c_degree
        )
        # total number of c variables in the lagrangians,
        # if there are some deactivated cbfs in the subset's 
        # expression, then the c_size should be the number of 
        # deactivated cbfs. Otherwise it is just 0.
        c_size = 0 

        # initial check:
        assert len(
            lagrangian_degree.activated_polys
            ) == np.sum(self.activation_index)
        if lagrangian_degree.deactivated_polys is not None:
            assert len(
                lagrangian_degree.deactivated_polys
            ) == self.all_polys.shape[0] - np.sum(self.activation_index)
            c_size = len(lagrangian_degree.deactivated_polys)
        else:
            assert self.all_polys.shape[0] == np.sum(self.activation_index)

        # emptiness check
        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.x)
        c_set = None
        if c_size > 0:
            c = sym.MakeVectorContinuousVariable(c_size,"c")
            c_set = sym.Variables(c)
            c_squared_poly = np.array(
                [sym.Polynomial(sym.Monomial(c[i], 2)) for i in range(c.shape[0])]
            )
            xc = np.concatenate([self.x, c], axis=0)
            xc_set = sym.Variables(xc)
            prog.AddIndeterminates(xc_set)
        else:
            prog.AddIndeterminates(x_set)

        lagrangians = lagrangian_degree.to_lagrangians(
            prog=prog,
            x=x_set,
            c=c_set,
            sos_type=sos_type,
        )
        
        lagrangian_times_activated_polys = lagrangians.activated_polys.dot(
            self.all_polys[self.activation_index == 1]
            )
        lagrangian_times_deactivated_polys = sym.Polynomial(0)
        if lagrangian_degree.deactivated_polys is not None:
            deactivated_polys = self.all_polys[self.activation_index == 0]
            lagrangian_times_deactivated_polys = lagrangians.deactivated_polys.dot(
                c_squared_poly * deactivated_polys + 1
                )
        poly = (- sym.Polynomial(1)
                - lagrangian_times_activated_polys
                - lagrangian_times_deactivated_polys
                )
        prog.AddSosConstraint(poly)
        
        result = solve_with_id(prog)
        if result.is_success():
            return True
        else:
            return False

        