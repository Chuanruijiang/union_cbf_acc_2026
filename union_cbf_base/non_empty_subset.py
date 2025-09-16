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

# """
# In this section, we want to get all the non-empty subsets of union CBFs.

# Given CBFs h1,...,hn. Let P = {1,...,n}. We want get the following:
# 1. All possible subsets of P except the empty set, denoted as Sᴾ
# 2. For all N ∈ Sᴾ, let the set 
#       S_N = {x| hₚ(x)≥0, ∀ i∈ N; hₚ'(x) < 0, ∀ p'∈ P\N}
#    We check if S_N is empty.
# 3. Return all the S_N that are not empty.

# All the three steps above are finally integrated into the function 
# "get_non_empty_region(h)", where the input argument "h" should be an
# array of polynomials. h=[h1, h2, ..., hn]. 

# Note: In this code base, we will not compute all possible subset of P,
# instead, we first generate a truth table for n-number of elements. 
# Except the all-zero row in the truth table, all the other rows are 
# boolean mask that tells us which cbf is ≥ 0 and which is not for a
# subset S_N, with each N ∈ Sᴾ.  

# The following module will be used to check whether a subset with form
# S_N = {x| hₚ(x)≥0, ∀ i∈ N; hₚ'(x) < 0, ∀ p'∈ P\N}
# is empty

# Let's use the following example to show how to verify the emptiness:
# if a subset is: {x|h1(x)≥0, h2(x)<0}
# checking the emptiness of this subset is equivalent to checking
# whether the following set is empty:
# {(x,c)| h1(x)≥ 0, c1^2h2(x)=-1}
# hence, we can use the following SOS program to check the emptiness:

# -1 - s1(x,c)*h1(x) - s2(x,c)*(c1^2h2(x)+1) is SOS

# where s1 is SOS, s2 is a free polynomial.

# Since we may also have a set S_N = {x|h1(x)≥ 0,...,hn(x)≥ 0}, then
# the variables c are optional.

# In the following codes, "activated" referes to h(x)≥0, while 
# "deactivated" refers to h'(x)<0. 
# """

@dataclass
class SubsetEmptinessLagrangian:
    activated_cbf: np.ndarray
    deactivated_cbf: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
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
        return SubsetEmptinessLagrangian(
            activated_cbf=lagrangian_activaed_cbf_result,
            deactivated_cbf=lagrangian_deactivated_cbf_result
        )

@dataclass
class SubsetEmptinessLagrangianDegree:
    activated_cbf: List[Degree]
    deactivated_cbf: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variable,
        c: Optional[sym.Variable],
        *,
        lagrangian_activated_cbf: Optional[np.ndarray] = None,
        lagrangian_deactivated_cbf: Optional[np.ndarray] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        # if there is no activated cbfs in the current subset,
        # then the c variable should also be None.
        if self.deactivated_cbf is None:
            assert c is None
        
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
                c=c,
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

        return SubsetEmptinessLagrangian(
            activated_cbf=lagrangian_activated_cbf,
            deactivated_cbf=lagrangian_deactivated_cbf
        )

@dataclass
class Subset:
    x: np.ndarray
    # This array of Polynomial containts all the CBFs
    # (including activated and deactivated CBFs)
    cbfs: np.ndarray
    # a 1D array showing the activated CBF indecies. 
    # According to the setup above, this indices vector stands
    # for a possible N ∈ P. 
    activation_index: np.ndarray 

    def _create_emtiness_lagrangian_degrees(
        self,
        lagrangian_x_degree: int,
        lagrangian_act_c_degree: int,
        lagrangian_deact_c_degree: int
    ) -> SubsetEmptinessLagrangianDegree:
        num_activated_cbf = np.sum(self.activation_index)
        
        # check whether the number of activated CBFs is valid:
        assert num_activated_cbf > 0
        assert num_activated_cbf <= self.cbfs.shape[0]
        
        num_deactivated_cbf = self.cbfs.shape[0] - num_activated_cbf
        lagrangian_degrees = SubsetEmptinessLagrangianDegree(
            activated_cbf=[Degree(
                x=lagrangian_x_degree,
                y = 0,
                c=lagrangian_act_c_degree
                )] * num_activated_cbf,
            deactivated_cbf=([Degree(
                x=lagrangian_x_degree,
                y = 0,
                c=lagrangian_deact_c_degree
                )] * num_deactivated_cbf
                if num_deactivated_cbf > 0
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
            lagrangian_degree.activated_cbf
            ) == np.sum(self.activation_index)
        if lagrangian_degree.deactivated_cbf is not None:
            assert len(
                lagrangian_degree.deactivated_cbf
            ) == self.cbfs.shape[0] - np.sum(self.activation_index)
            c_size = len(lagrangian_degree.deactivated_cbf)
        else:
            assert self.cbfs.shape[0] == np.sum(self.activation_index)

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
        
        lagrangian_times_activated_cbf = lagrangians.activated_cbf.dot(
            self.cbfs[self.activation_index == 1]
            )
        lagrangian_times_deactivated_cbf = sym.Polynomial(0)
        if lagrangian_degree.deactivated_cbf is not None:
            deactivated_cbfs = self.cbfs[self.activation_index == 0]
            lagrangian_times_deactivated_cbf = lagrangians.deactivated_cbf.dot(
                c_squared_poly * deactivated_cbfs + 1
                )
        poly = (- sym.Polynomial(1)
                - lagrangian_times_activated_cbf
                - lagrangian_times_deactivated_cbf
                )
        prog.AddSosConstraint(poly)
        
        result = solve_with_id(prog)
        if result.is_success():
            return True
        else:
            return False
        

def get_non_empty_region(
    h: np.ndarray,
    x: np.ndarray,
    *,
    lagrangian_x_degree: int = 2,
    lagrangian_act_c_degree: int = 2,
    lagrangian_deact_c_degree: int = 0,
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
            activation_index=activation_case,
        )
        if np.sum(activation_case) == h.shape[0]:
            lagrangian_act_c_degree = 0
        # check the emptiness of the subset
        if not subset.is_empty(
            lagrangian_x_degree=lagrangian_x_degree,
            lagrangian_act_c_degree=lagrangian_act_c_degree,
            lagrangian_deact_c_degree=lagrangian_deact_c_degree,
            sos_type=sos_type,
        ):
            non_empty_subsets.append(subset)
        
    return non_empty_subsets
