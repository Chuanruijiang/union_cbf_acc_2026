"""
In this script, we develop the class and methods to verify emptiness
of a subset. For convenience, we de-couple this script from the information
of relative degrees, system dynamics, and alpha parameters.. etc.. Hence,
what this script verifies is the emptiness of the following kind of set:

 S = {x| List of array of polynomials ≥ 0; an array of polynomials < 0;
          an optional array of polynomials = 0 }.

Let's use an example to show how to verify the emptiness:
if a subset is: {x|[h11(x)≥0, h12(x)≥0], 
                   [h21(x)≥0, h22(x)≥0]; [h31(x)<0, h32(x)<0]; [e1(x)=0, e2(x)=0]}
checking the emptiness of this subset is equivalent to checking
whether the following set is empty:
{(x,c)| [h11(x)≥0, h12(x)≥0], 
        [h21(x)≥0, h22(x)≥0]; [c1²h31(x)=-1, c2²h32(x)=-1]; [e1(x)=0, e2(x)=0]}
hence, we can use the following SOS program to check the emptiness:

-1 - s1(x,c)ᵀh1(x) 
   - s2(x,c)ᵀh2(x) 
   - q(x, c)ᵀ[c1²h31(x)+1, c2²h32(x)+1]
   - p(x, c)ᵀ[e1(x), e2(x)] is SOS

where s1 are SOS, s2 are SOS, and q and p are free polynomials.

Since we may also have a set with only activated polynomial groups:
             S_N = {x|h1(x)≥ 0,...,hn(x)≥ 0}
Hence, the deactivated polynomials and the variables c are optional.

In the following codes, "activated" referes to a polynomial p(x)≥0, 
while "deactivated" refers to a polynomial p(x)<0. In the journal project,
we use the Subset class defined here to represent the subset (8) or (9)
in Section 5.1 of the journal paper.
"""

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
    """
    The emptiness verification program is defined in porg (10) lemma 4
    of the Section 5.1 of the journal paper. This class defines the 
    lagrangians needed in Prog (10).
    """
    # s_i(x, c) and s_p(x, c) for all i in I and p in N
    activated_poly_groups: List[np.ndarray]
    # q_p'(x, c) for all p' in P\N
    deactivated_polys: Optional[np.ndarray]
    # if we also have state equation constraints,
    # then the prog (10) will also have state equation 
    # lagrangians.
    equation_constraints: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        activated_poly_groups = [
            get_polynomial_result(
                result=result,
                p=lagrangian,
                coefficient_tol=coefficient_tol,
            )
            for lagrangian in self.activated_poly_groups
        ]
        deactivated_polys = (
            None
            if self.deactivated_polys is None
            else get_polynomial_result(
                result=result,
                p=self.deactivated_polys,
                coefficient_tol=coefficient_tol,
            )
        )
        equation_constraints = (
            None
            if self.equation_constraints is None
            else get_polynomial_result(
                result=result,
                p=self.equation_constraints,
                coefficient_tol=coefficient_tol,
            )
        )
        return SubsetEmptinessLagrangian(
            activated_poly_groups=activated_poly_groups,
            deactivated_polys=deactivated_polys,
            equation_constraints=equation_constraints,
        )

@dataclass
class SubsetEmptinessLagrangianDegree:
    activated_poly_groups: List[List[Degree]]
    deactivated_polys: Optional[List[Degree]]
    equation_constraints: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variable,
        c: Optional[sym.Variable],
        *,
        lagrangian_activated_poly_groups: Optional[List[np.ndarray]] = None,
        lagrangian_deactivated_polys: Optional[np.ndarray] = None,
        lagrangian_equation_constraints: Optional[np.ndarray] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        # if there is no activated cbfs in the current subset,
        # then the c variable should also be None.
        if self.deactivated_polys is None:
            assert c is None
        
        lagrangian_activated_poly_groups = [
                to_lagrangian_impl(
                    prog=prog,
                    x=x,
                    y=None,
                    c=c,
                    sos_type=sos_type,
                    degree=self.activated_poly_groups[i],
                    is_sos=True,
                    lagrangian=(
                        lagrangian_activated_poly_groups[i]
                        if lagrangian_activated_poly_groups is not None 
                        else None
                        ),
                ) for i in range(len(self.activated_poly_groups))
            ]
                          
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

        lagrangian_equation_constraints = (
            None
            if self.equation_constraints is None
            else to_lagrangian_impl(
                prog=prog,
                x=x,
                y=None,
                c=c,
                sos_type=sos_type,
                degree=self.equation_constraints,
                is_sos=False,
                lagrangian=(
                    lagrangian_equation_constraints 
                    if lagrangian_equation_constraints is not None 
                    else None
                    ),
            )
        )

        return SubsetEmptinessLagrangian(
            activated_poly_groups=lagrangian_activated_poly_groups,
            deactivated_polys=lagrangian_deactivated_polys,
            equation_constraints=lagrangian_equation_constraints
        )

@dataclass
class Subset:
    # "x" is the array of variables of polynomail objects in the subset.
    x: np.ndarray
    # "activated_poly_groups" is the list of arrays of the activated polys
    # in the subset.
    # During the verification of Verif-I, this member's length should be
    # the total number of static HOCBFs plus tha amount of avaliable
    # switching HOCBFs.
    activated_poly_groups: List[np.ndarray]
    # "deactivated_polys" is an array of deactivated polys in the subet.
    deactivated_polys: Optional[np.ndarray]
    # "equation_constraints" is an array of equation constraints polys in 
    # the subset.
    equation_constraints: Optional[np.ndarray]
    # 
    # The following class memebers are not used in the emptiness checking
    # of a subset. But they provides meta infromation for Verif-I when
    # verifying the partitioned subsets. 
    # (see journal paper Section 5.1 eq. (8) and (9))
    #
    # "swtiching_cbfs_active_num" is the total amount of avaliable switching 
    # CBF/HOCBFs in the current subset. It referes to the quantity |N| in the
    # Section 5.1 of the paper.
    num_avaliable_switching_cbfs: Optional[int] = None
    # "switching_cbfs_active_mask" is a 0-1 vector with size |N| that shows
    # which specific swtiching CBFs are avaliable in the current subset.
    mask_avaliable_switching_cbfs: Optional[np.ndarray] = None
    # "subset_component_indicator" referes to the vector "\bar{i}" vector in
    # the Section 5.1 eq. (9). 
    subset_component_indicator: Optional[np.ndarray] = None
        
    def _create_emtiness_lagrangian_degrees(
        self,
        lagrangian_x_degree: int,
        lagrangian_act_c_degree: int,
        lagrangian_deact_c_degree: int,
    ) -> SubsetEmptinessLagrangianDegree:
        num_activated_groups = len(self.activated_poly_groups)
        # form the list of list of degrees for activated poly groups:
        activated_poly_groups_degrees = [
            [
                Degree(
                    x=lagrangian_x_degree,
                    y=0,
                    c=(lagrangian_act_c_degree
                       if self.deactivated_polys is not None
                       else 0),
                )
                for _ in range(self.activated_poly_groups[i].shape[0])
            ]
            for i in range(num_activated_groups)
        ]
        # form the list of degrees for deactivated polys:
        deactivated_polys_degrees = (
            None
            if self.deactivated_polys is None
            else [
                Degree(
                    x=lagrangian_x_degree,
                    y=0,
                    c=lagrangian_deact_c_degree,
                )
                for _ in range(self.deactivated_polys.shape[0])
            ]
        )
        equation_constraints_degrees = (
            None
            if self.equation_constraints is None
            else [
                Degree(
                    x=lagrangian_x_degree,
                    y=0,
                    c=(lagrangian_act_c_degree
                       if self.deactivated_polys is not None
                       else 0),
                )
                for _ in range(self.equation_constraints.shape[0])
            ]
        )
        lagrangian_degrees = SubsetEmptinessLagrangianDegree(
            activated_poly_groups=activated_poly_groups_degrees,
            deactivated_polys=deactivated_polys_degrees,
            equation_constraints=equation_constraints_degrees
        )
        return lagrangian_degrees
        
    def is_empty(
        self,
        lagrangian_x_degree: int,
        lagrangian_act_c_degree: int,
        lagrangian_deact_c_degree: int,
        *,
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
        num_activated_groups = len(
            lagrangian_degree.activated_poly_groups
            )
        num_deactivated_polys = (
            self.deactivated_polys.shape[0]
            if lagrangian_degree.deactivated_polys is not None
            else 0
        )
        num_equation_constraints = (
            self.equation_constraints.shape[0]
            if lagrangian_degree.equation_constraints is not None
            else 0
        )

        # initial check:
        assert len(
            lagrangian_degree.activated_poly_groups
            ) == num_activated_groups
        if lagrangian_degree.deactivated_polys is not None:
            assert len(
                lagrangian_degree.deactivated_polys
            ) == num_deactivated_polys
            c_size = num_deactivated_polys
        if lagrangian_degree.equation_constraints is not None:
            assert len(
                lagrangian_degree.equation_constraints
            ) == num_equation_constraints
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

        # we build the emptiness verification progam here,
        # the verification program is in prog (10) Lemma 4 in
        # journal extension.
        sum_lagrangian_times_activated_poly_group = sym.Polynomial(0)
        for i in range(num_activated_groups):
            lagrangian_times_activated_poly_group \
                = lagrangians.activated_poly_groups[i].dot(
                self.activated_poly_groups[i]
                )
            sum_lagrangian_times_activated_poly_group \
                += lagrangian_times_activated_poly_group
        
        lagrangian_times_deactivated_polys = sym.Polynomial(0)
        if lagrangian_degree.deactivated_polys is not None:
            lagrangian_times_deactivated_polys = lagrangians.deactivated_polys.dot(
                c_squared_poly * self.deactivated_polys + 1
                )
        lagrangian_times_equation_constraints = sym.Polynomial(0)
        if lagrangian_degree.equation_constraints is not None:
            lagrangian_times_equation_constraints = lagrangians.equation_constraints.dot(
                self.equation_constraints
                )
        poly = (- sym.Polynomial(1)
                - sum_lagrangian_times_activated_poly_group
                - lagrangian_times_deactivated_polys
                - lagrangian_times_equation_constraints
                )
        prog.AddSosConstraint(poly)
        
        result = solve_with_id(prog)
        if result.is_success():
            return True
        else:
            return False