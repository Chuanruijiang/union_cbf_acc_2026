from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

import time

from union_cbf_base.utils import (
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

from union_cbf_base.non_empty_subset import (
    Subset
)

"""
In the non_empty_subset.py file, we aleady computed all the non-empty subsets
of the union of CBFs. In this section, we will check the feasiblility of all
these non-empty subsets. Namely, given a set of subsets S' = {S_N}, we wanted to
verify whether for all S_N ∈ S', and ∀ x ∈ S_N, there exists a control u, and an
activated CBF such that the activated CBF condition is satisfied. More formally,
we verifies that: ∀ S_N ∈ S', ∀ x ∈ S_N, ∃ u, ∃ i ∈ N, with:
    Lfhᵢ(x) + Lghᵢ(x) u + αᵢ*hᵢ(x) >= η;
    Au ≤ c - ϵ
where Au ≤ c identifies the control input limits and η, ϵ>0 are given constants.
To be more simple, we will replace the two constraints above with a single one:
    Λᵢ(x) u ≤ ξᵢ(x)
where:
    Λᵢ(x) = [-Lghᵢ(x);
                A     ],

    ξᵢ(x) = [Lfhᵢ(x) + αᵢ*hᵢ(x)-η ;
                c-ϵ   ]

Now, we use an example to give the detailed verification framework.
Assume the subset has h1, h2 and h3 as the activated CBFs, and we
have h4 and h5 as the deactivated CBFs such that:
    S_N={x | h1(x) >= 0, h2(x) >= 0, h3(x) >= 0, h4(x) < 0, h5(x) < 0}
We want to verify that ∀ x ∈ S_N, ∃ u, ∃ i ∈ {1,2,3}, with:
    Λᵢ(x) u ≤ ξᵢ(x)

According to the verification of non-empty subsets, we already saw that
having strict inequalities like h4(x)<0 and h5(x)<0 in a subset S_N may
brings about more number of axilluary variables (the c-variables) in the
SOS programming. Hence, in order to simplify the SOS feasibility verifi-
cation program, we work on a sufficient condtion of the above statement.
We define a new subset S_N' as:
    S_N'={x | h1(x) >= 0, h2(x) >= 0, h3(x) >= 0, -h4(x) ≥ 0, -h5(x) ≥ 0}
Clearly, S_N ⊆ S_N'. Hence, if we can verify that ∀ x ∈ S_N', 
∃ u, ∃ i ∈ {1,2,3}, with:
    Λᵢ(x) u ≤ ξᵢ(x)
then the original statement is also true.       

The negation of the above statement is:
    ∃ x ∈ S_N' such that ∀ i ∈ {1,2,3}, ∄ u, with:
        Λᵢ(x) u ≤ ξᵢ(x)
By Farkas' lemma, the above statement is equivalent to:
    ∃ x ∈ S_N' such that ∀ i ∈ {1,2,3}, ∃ yᵢ ∈ Rⁿ, with:
        Λᵢ(x)ᵀyᵢ² = 0 and ξᵢ(x)ᵀyᵢ² + 1 = 0
The above statement can be verified by verifying the non-emptiness for
the following set:
{(x, y₁, y₂, y₃)| 
    h₁(x) >= 0, h₂(x) >= 0, h₃(x) >= 0, -h₄(x) ≥ 0, -h₅(x) ≥ 0,
    Λ₁(x)ᵀy₁² = 0, ξ₁(x)ᵀy₁² + 1 = 0,
    Λ₂(x)ᵀy₂² = 0, ξ₂(x)ᵀy₂² + 1 = 0,
    Λ₃(x)ᵀy₃² = 0, ξ₃(x)ᵀy₃² + 1 = 0,
}
Now we can negate the above statement and get the original statement.
We wanted to verify the set above is EMPTY by using SOS programming.
let y = [y1; y2; y3], the above set is empty if:
 -1 - s₀(x,y)ᵀ *[h1, h2, h3, -h4, -h5]ᵀ
    - s₁(x,y)ᵀΛ₁(x)ᵀy₁² - s₂(x,y)ᵀΛ₂(x)ᵀy₂² - s₃(x,y)ᵀΛ₃(x)ᵀy₃²
    - q₁(x,y)(ξ₁(x)ᵀy₁² + 1) 
    - q₂(x,y)(ξ₂(x)ᵀy₂² + 1) 
    - q₃(x,y)(ξ₃(x)ᵀy₃² + 1)
where s0 is a vector of SOS, all the others are Polynomials.

Now, we summarize the number of SOS and polynomials that are needed to
verify the subset S_N in the general case:
Assume we have N_h number of CBFs in total. 
    n number of CBFs are activated in S_N
    m number of CBFs are deactivated in S_N (m+n=N_h)
    u is the control input with dimension n_u
Then:
    S_0: is an SOS vector with N_h elements.
    there are n number of SOS vectors s_i, each with dimension n_u
    there are n number of polynomial q_i, each is a scalar polynomial.
    The vector of y has dimension n*(n_u + 1)
    The vector of x has dimension n_x
In total, the number of variables is:
    n_x + n*(n_u + 1)
"""

@dataclass
class SubsetFeasibilityLagrangian:
    # an array of polynomials that presents the multipliers in s_0:
    cbfs: np.ndarray
    # a list of arrays of polynomials that presents the multipliers in s_i:
    # since we have n number of activated CBFs, then the length of this list
    # is equal to n. Each element in this list is an array of polynomials
    # with dimension equal to the control input dimension.
    lambda_y: List[np.ndarray]
    # a list of arrays of polynomials that presents the multipliers in q_i:
    # the length of this list is equal to n. Each element in this list is a
    # scalar polynomial that presents the multiplier q_i. 
    xi_y: List[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        cbfs_result = get_polynomial_result(
            result=result,
            p=self.cbfs,
            coefficient_tol=coefficient_tol
        )
        lambda_y_result = [
            get_polynomial_result(
                result=result,
                p=lamb_y,
                coefficient_tol=coefficient_tol
            ) for lamb_y in self.lambda_y
        ]
        xi_y_result = [
            get_polynomial_result(
                result=result,
                p=xi_y,
                coefficient_tol=coefficient_tol
            ) for xi_y in self.xi_y
        ]
        return Self(
            cbfs=cbfs_result,
            lambda_y=lambda_y_result,
            xi_y=xi_y_result
        )


@dataclass
class SubsetFeasibilityLagrangianDegrees:
    # the length of the following list is equal to the
    # number of CBFs in the subset.
    s0: List[Degree]
    # the length of the outer list is equal to the number of
    # activated CBFs in the subset. The length of the inner list
    # is equal to the control input dimension.
    s_lambda_y: List[List[Degree]]
    # the length of the following list is equal to the number of
    # activated CBFs in the subset.
    q_xi_y: List[Degree]
    """
    Note that all the number-of-dimension requirements should
    be checked before the instantiation of this class.
    """

    def to_lagrangian(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        y_sets: List[sym.Variables],
        *,
        lagrangian_cbfs: Optional[np.ndarray] = None,
        lagrangian_lambda_y: Optional[List[np.ndarray]] = None,
        lagrangian_xi_y: Optional[List[np.ndarray]] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        """
        Back to the given example at the top of this file,
        since the lamda_y and xi_y terms for h_1(x) already
        have the y_1 variables in degree 2 and there is no
        cross terms between variables in y_1. Then we don't
        need to include y_1 in the lagrangian multipliers for
        lamdba_y and xi_y for h_1(x). However, for 
        only-x-dependent terms like h_1(x),..., h_5(x), the
        lagrangian multipliers should include all the y variables.

        Hence, the argument y_sets is a list of sym.Variables,
        it has the y-sets for each of the multipliers for lambda_y
        and xi_y, and also has the y-set that includes all the y
        variables. The length of y_sets should be n+1 where n is
        the number of activated CBFs in the subset. For the example
        above the y_sets should be:

        [y_set1, y_set2, y_set3, y_set_all]

        where y_seti is the set of x and y_i variables, and
        y_set_all is the set of x and all y variables.

        However, if the subset only has one activated CBF, then
        the y_sets should be [y_set1, y_set_all], where y_set1
        is none since the s_lambda_y and q_xi_y lagrangians in 
        this case will not need to inlcude any y variables.

        Note that all the number-of-dimension requirements should
        be checked before calling this function.
        """
        cbfs = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_sets[-1],  # y_set_all,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.s0,
            lagrangian=lagrangian_cbfs
        )
        lambda_y = [
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[i], # y_seti
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.s_lambda_y[i],
                lagrangian=(None if lagrangian_lambda_y is None
                            else lagrangian_lambda_y[i])
            ) for i in range(len(self.s_lambda_y))
        ]
        xi_y = [
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[i], # y_seti
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.q_xi_y[i],
                lagrangian=(None if lagrangian_xi_y is None
                            else lagrangian_xi_y[i])
            ) for i in range(len(self.q_xi_y))
        ]
        return SubsetFeasibilityLagrangian(
            cbfs=cbfs,
            lambda_y=lambda_y,
            xi_y=xi_y
        )


class UnionCbf: 
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        cbfs: np.ndarray,
        alpha: float,
        control_limits: Optional[Tuple[np.ndarray, np.ndarray]]):
        # basic checks
        assert isinstance(x, np.ndarray)
        assert isinstance(cbfs, np.ndarray)
        assert isinstance(cbfs[0], sym.Polynomial)
        assert f.shape[0] == x.shape[0]
        assert g.shape[0] == x.shape[0]
        
        if control_limits is not None:
            assert control_limits[0].shape[0] == control_limits[1].shape[0]
            assert g.shape[1] == control_limits[0].shape[1]
            self.control_limits = control_limits
            self.xi_lambda_rows = control_limits[0].shape[0] + 1
        else:
            self.control_limits = None
            self.xi_lambda_rows = 1
        
        self.x = x
        self.f = f
        self.g = g
        self.cbfs = cbfs
        self.alpha = alpha
        self.n_x = x.shape[0]
        self.n_u = g.shape[1]
        self.n_h = cbfs.shape[0]
    
    def all_possible_subsets(
        self,
    ) -> np.ndarray:
        """
        Return an array of shape (2**n_h, n_h) that includes
        all the possible subsets of the union of CBFs.
        Each row is a boolean mask of subset, telling which CBFs are activated.
        1 means the CBF is activated in the subset, 0 means
        the CBF is deactivated in the subset.
        """
        return truth_table(self.n_h)
    
    def get_non_empty_subsets(
        self,
        all_subsets_mask: np.ndarray,
        *,
        lagrangian_x_degree: int = 2,
        lagrangian_act_c_degree: int = 2,
        lagrangian_deact_c_degree: int = 0,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> List[Subset]:
        """
        Given a set of subsets (each subset is represented by a boolean mask),
        return the list of non-empty subsets.
        """
        non_empty_subsets = []
        for each_mask in all_subsets_mask:
            # skip the set with all CBFs deactivated
            if np.sum(each_mask) == 0:
                continue
            # create the subset
            subset = Subset(
                x=self.x,
                cbfs=self.cbfs,
                activation_index=each_mask,
            )
            if np.sum(each_mask) == self.cbfs.shape[0]:
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

    def construct_feasibility_check_prog(
        self,
        subset: Subset,
        cbf_lagrangian_x_degree: int,
        cbf_lagrangian_y_degree: int,
        lambda_y_lagrangian_x_degree: int,
        lambda_y_lagrangian_y_degree: int,
        xi_y_lagrangian_x_degree: int,
        xi_y_lagrangian_y_degree: int,
        eta: float,
        epsilon: float,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> solvers.MathematicalProgram:
        # 1. construct the lagrangian degrees
        lagrangian_degrees = self._construct_lagrangian_degrees(
            subset=subset,
            cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
            cbf_lagrangian_y_degree=cbf_lagrangian_y_degree,
            lambda_y_lagrangian_x_degree=lambda_y_lagrangian_x_degree,
            lambda_y_lagrangian_y_degree=lambda_y_lagrangian_y_degree,
            xi_y_lagrangian_x_degree=xi_y_lagrangian_x_degree,
            xi_y_lagrangian_y_degree=xi_y_lagrangian_y_degree
        )
        # 2. construct the x_set, and y_sets
        (
            xy_set, x_set, y_sets, y_squared_polys
        ) = self._construct_x_y_sets(
            subset=subset
        )
        # 3. compute the xis and lambdas
        lambda_list, xi_list = self._lambda_xi(
            subset=subset,
            eta=eta,
            epsilon=epsilon
        )

        # create the program and sepcify the indeterminates
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(xy_set)

        # checking dimentions:
        num_activated = int(np.sum(subset.activation_index))
        assert len(y_sets) == num_activated + 1

        # 4. create the lagrangian multipliers
        subset_lagrangian = lagrangian_degrees.to_lagrangian(
            prog=prog,
            x_set=x_set,
            y_sets=y_sets,
            sos_type=sos_type
        )
        # 5. add_sos_constraints
        self._add_sos_constraints(
            prog=prog,
            subset=subset,
            subset_lagrangian=subset_lagrangian,
            lambda_list=lambda_list,
            xi_list=xi_list,
            y_squared_polys=y_squared_polys
        )
        return prog
        
    def check_feasibility_in_subset(
        self,
        subset: Subset,
        cbf_lagrangian_x_degree: int,
        cbf_lagrangian_y_degree: int,
        lambda_y_lagrangian_x_degree: int,
        lambda_y_lagrangian_y_degree: int,
        xi_y_lagrangian_x_degree: int,
        xi_y_lagrangian_y_degree: int,
        eta: float,
        epsilon: float,
    ) -> bool:
        prog = self.construct_feasibility_check_prog(
            subset=subset,
            cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
            cbf_lagrangian_y_degree=cbf_lagrangian_y_degree,
            lambda_y_lagrangian_x_degree=lambda_y_lagrangian_x_degree,
            lambda_y_lagrangian_y_degree=lambda_y_lagrangian_y_degree,
            xi_y_lagrangian_x_degree=xi_y_lagrangian_x_degree,
            xi_y_lagrangian_y_degree=xi_y_lagrangian_y_degree,
            eta=eta,
            epsilon=epsilon
        )
        result = solve_with_id(prog)
        return result.is_success()
    
    def check_simplified_feasibility(
        self,
        cbf_index: int,
        cbf_lagrangian_x_degree: int,
        cbf_lagrangian_y_degree: int,
        lambda_y_lagrangian_x_degree: int,
        lambda_y_lagrangian_y_degree: int,
        xi_y_lagrangian_x_degree: int,
        xi_y_lagrangian_y_degree: int,
        eta: float,
        epsilon: float,
    ):
        """
        This function checks another simpler type of feasibility
        Given a subset of X_i = {x|h_i(x) ≥ 0}, we wanted to verify
        whether ∀ x ∈ X_i, ∃ u, with:
            Lfh_i(x) + Lgh_i(x) u + α_i*h_i(x) >= η;
            Au ≤ c - ϵ
        where Au ≤ c identifies the control input limits and
        η, ϵ>0 are given constants.
        Formally, this is equivalent to verify that ∀ x ∈ X_i,
        ∃ u, with:
            Λ_i(x) u ≤ ξ_i(x)
        where:
            Λ_i(x) = [-Lgh_i(x);
                        A     ],

            ξ_i(x) = [Lfh_i(x) + α*h_i(x)-η ;
                        c-ϵ   ]
        By Farkas' lemma, the above statement is equivalent to:
            ∀ x ∈ X_i, ∄ y ∈ Rⁿ, with:
                Λ_i(x)ᵀy² = 0 and ξ_i(x)ᵀy² + 1 = 0
        Hence it is to verify that the following set is empty:
        {(x, y)| h_i(x) >= 0,
            Λ_i(x)ᵀy² = 0, ξ_i(x)ᵀy² + 1 = 0}
        Now we can get the SOS program to the verification.
        -1 - s₀(x,y)ᵀ *[h_i(x)]ᵀ
            - s₁(x,y)ᵀΛ_i(x)ᵀy²
            - q₁(x,y)(ξ_i(x)ᵀy² + 1) is sos
        where s0 is an SOS polynomial, and all the others are polynomials.
        """

        """
        All the defined lagragian degree and lagrangian classes can still
        be used for this verification. Looking back at the text at the top,
        we can see that this verification is a special case where the
        subset is in the form S_N = {x|h_i(x) ≥ 0}, i.e., the subset has only
        one activated CBF and no deactivated CBF.
        """
        assert cbf_index >= 0 and cbf_index < self.n_h
        subset = Subset(
            x=self.x,
            cbfs = np.array([self.cbfs[cbf_index]]),
            activation_index=np.array([1])
        )
        return self.check_feasibility_in_subset(
            subset=subset,
            cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
            cbf_lagrangian_y_degree=cbf_lagrangian_y_degree,
            lambda_y_lagrangian_x_degree=lambda_y_lagrangian_x_degree,
            lambda_y_lagrangian_y_degree=lambda_y_lagrangian_y_degree,
            xi_y_lagrangian_x_degree=xi_y_lagrangian_x_degree,
            xi_y_lagrangian_y_degree=xi_y_lagrangian_y_degree,
            eta=eta,
            epsilon=epsilon
        )

    def verification_of_theorem_2(
        self,
        cbf_lagrangian_x_degree: int,
        cbf_lagrangian_y_degree: int,
        lambda_y_lagrangian_x_degree: int,
        lambda_y_lagrangian_y_degree: int,
        xi_y_lagrangian_x_degree: int,
        xi_y_lagrangian_y_degree: int,
        eta: float,
        epsilon: float,
    ):
        """
        This is the main function that will be for experiments.
        This verifies the conditions in Theorem 2 in the paper.
        """
        # 1. get all the possible subsets
        all_subsets_mask = self.all_possible_subsets()
        # 2. get all the non-empty subsets
        non_empty_subsets = self.get_non_empty_subsets(
            all_subsets_mask=all_subsets_mask,
            lagrangian_x_degree=2,
            lagrangian_act_c_degree=2,
            lagrangian_deact_c_degree=0,
            sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        )
        print("The following subsets are non-empty:")
        for subset in non_empty_subsets:
            print(subset.activation_index)
        print(f"In total, there are {len(non_empty_subsets)} non-empty subsets.")
        # 3. check the feasibility of all the non-empty subsets
        all_feasible = True
        for subset in non_empty_subsets:
            assert subset.cbfs.shape[0] == self.n_h
            assert np.sum(subset.activation_index) > 0
            start_time = time.time()
            is_feasible = self.check_feasibility_in_subset(
                subset=subset,
                cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
                cbf_lagrangian_y_degree=cbf_lagrangian_y_degree,
                lambda_y_lagrangian_x_degree=lambda_y_lagrangian_x_degree,
                lambda_y_lagrangian_y_degree=lambda_y_lagrangian_y_degree,
                xi_y_lagrangian_x_degree=xi_y_lagrangian_x_degree,
                xi_y_lagrangian_y_degree=xi_y_lagrangian_y_degree,
                eta=eta,
                epsilon=epsilon
            )
            end_time = time.time()
            if not is_feasible:
                all_feasible = False
                print("The following subset is found to be infeasible:")
                print(subset.activation_index)
            else:
                print("The following subset is verified to be feasible:")
                print(subset.activation_index)
                print(f"Time taken: {end_time - start_time} seconds")
        return all_feasible

    def verification_of_theorem_3(
        self,
        cbf_lagrangian_x_degree: int,
        cbf_lagrangian_y_degree: int,
        lambda_y_lagrangian_x_degree: int,
        lambda_y_lagrangian_y_degree: int,
        xi_y_lagrangian_x_degree: int,
        xi_y_lagrangian_y_degree: int,
        eta: float,
        epsilon: float,
    ):
        """
        This function verifies the conditions in Theorem 3 in the paper.
        """
        verification_succeeded = True
        for i in range(self.n_h):
            start_time = time.time()
            is_feasible = self.check_simplified_feasibility(
                cbf_index=i,
                cbf_lagrangian_x_degree=cbf_lagrangian_x_degree,
                cbf_lagrangian_y_degree=cbf_lagrangian_y_degree,
                lambda_y_lagrangian_x_degree=lambda_y_lagrangian_x_degree,
                lambda_y_lagrangian_y_degree=lambda_y_lagrangian_y_degree,
                xi_y_lagrangian_x_degree=xi_y_lagrangian_x_degree,
                xi_y_lagrangian_y_degree=xi_y_lagrangian_y_degree,
                eta=eta,
                epsilon=epsilon
            )
            end_time = time.time()
            if not is_feasible:
                verification_succeeded = False
                print(f"The CBF with index {i} fails the verification.")
            else:
                print(f"The CBF with index {i} passes the verification.")
                print(f"Time taken: {end_time - start_time} seconds")
        return verification_succeeded

    def bilinear_alternation():
        """
        This function implements the bilinear alternation algorithm
        to synthesize a single CBF. We define this function in the
        union of CBFs class since single CBF is a special case of
        union of CBFs. But since this function only synthesizes a
        single CBF, then we should check the input at the first few
        commands of this function.
        """

    def _lambda_xi(
        self,
        subset: Subset,
        eta: float,
        epsilon: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        This function computes the lambda and xi terms for all the
        activated CBFs in the subset.
        For the output, the fisrt term is the list of lambda matrices,
        the second term is the list of xi vectors.
        """
        activated_cbfs = subset.cbfs[subset.activation_index==1]
        lambda_list = []
        xi_list = []
        for h in activated_cbfs:
            Lfh = lie_derivative(
                poly=h,
                vector_feild=self.f,
                variables=self.x,
                pow=1
                )
            Lgh = lie_derivative(
                poly=h,
                vector_feild=self.g,
                variables=self.x,
                pow=1
                )
            if self.control_limits is not None:
                A = self.control_limits[0]
                c = self.control_limits[1]
                assert Lgh.shape[0] == self.n_u
                assert Lgh.shape[0] == A.shape[1]
                Lambda = np.vstack((-Lgh, A))
                xi = np.hstack((Lfh + self.alpha*h - eta,
                                c - epsilon))
            else:
                Lambda = -Lgh.reshape((1, self.n_u))
                xi = np.array([Lfh + self.alpha*h - eta])
            assert Lambda.shape[0] == self.xi_lambda_rows
            assert xi.shape[0] == self.xi_lambda_rows
            
            lambda_list.append(Lambda)
            xi_list.append(xi)
        return lambda_list, xi_list

    def _construct_lagrangian_degrees(
        self,
        subset: Subset,
        cbf_lagrangian_x_degree: int,
        cbf_lagrangian_y_degree: int,
        lambda_y_lagrangian_x_degree: int,
        lambda_y_lagrangian_y_degree: int,
        xi_y_lagrangian_x_degree: int,
        xi_y_lagrangian_y_degree: int,
    ) -> SubsetFeasibilityLagrangianDegrees:
        """
        This functiion constructs the lagrangian degrees
        for the feasibility verification of the given subset.
        """
        num_activated = int(np.sum(subset.activation_index))
        s0_degree = [
            Degree(
                x=cbf_lagrangian_x_degree,
                y=cbf_lagrangian_y_degree,
                c=0
            ) for _ in range(subset.cbfs.shape[0])
        ]
        s_lambda_y_degree = [
            [
                Degree(
                    x=lambda_y_lagrangian_x_degree,
                    y=(lambda_y_lagrangian_y_degree
                       if num_activated > 1 else 0),
                    c=0
                ) for _ in range(self.n_u)
            ]
            for _ in range(num_activated)
        ]
        q_xi_y_degree = [
            Degree(
                x=xi_y_lagrangian_x_degree,
                y=(xi_y_lagrangian_y_degree
                   if num_activated > 1 else 0),
                c=0
            ) for _ in range(num_activated)
        ]
        return SubsetFeasibilityLagrangianDegrees(
            s0=s0_degree,
            s_lambda_y=s_lambda_y_degree,
            q_xi_y=q_xi_y_degree
        )

    def _construct_x_y_sets(
        self,
        subset: Subset,
    ) -> Tuple[
        sym.Variables,
        sym.Variables,
        List[sym.Variables],
        List[np.ndarray]
        ]:
        """
        This function constructs the set of indeterminates for
        creating the lagrangian multipliers and the SOS feasibility
        constraint in the SOS program.
        This function only creats teh set of indeterminates for the
        SOS program of verifying a single subset. The output of this
        function has three parts:
        1. xy_set as the total indeterminates for SOS program.
        2. x_set for creating lagrangian multipliers.
        3. y_set for creating lagrangian multipliers.
            For the inner structure of y_sets, please check the
            explanation for function "to_lagrangian" in class
            "SubsetFeasibilityLagrangianDegrees".
        4. the squared y polynomial variables for creating
            the SOS feasibility constraint.
        """
        num_activated = int(np.sum(subset.activation_index))
        y_i_size = self.xi_lambda_rows
        y_i_groups = []
        y_all = np.array([])
        y_sets = []
        y_squared_polys = []
        for i in range(num_activated):
            y_i = sym.MakeVectorContinuousVariable(
                y_i_size, f"y_{i+1}")
            y_i_groups.append(y_i)
            y_squared_poly_i = np.array([
                sym.Polynomial(sym.Monomial(y_i[j], 2)) 
                for j in range(y_i_size)
            ])
            y_squared_polys.append(y_squared_poly_i)
            y_all = np.hstack((y_all, y_i))
        y_all_set = sym.Variables(y_all)
        if num_activated == 1:
            y_sets.append(None)
        elif num_activated > 1:   
            for i in range(num_activated):
                y_i_groups_copy = y_i_groups.copy()
                del y_i_groups_copy[i]
                y_i_set = sym.Variables(
                    np.concatenate(y_i_groups_copy)
                )
                y_sets.append(y_i_set)
        else:
            raise RuntimeError("The subset should have at least one activated CBF.")
        y_sets.append(y_all_set)
        x_set = sym.Variables(self.x)
        xy = np.hstack((self.x, y_all))
        xy_set = sym.Variables(xy)

        return (xy_set, x_set, y_sets, y_squared_polys)

    def _add_sos_constraints(
        self,
        prog: solvers.MathematicalProgram,
        subset: Subset,
        subset_lagrangian: SubsetFeasibilityLagrangian,
        lambda_list: List[np.ndarray],
        xi_list: List[np.ndarray],
        y_squared_polys: List[np.ndarray],
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        ):
        # basic checks:
        num_activated = int(np.sum(subset.activation_index))
        assert len(subset_lagrangian.lambda_y) == num_activated
        assert len(subset_lagrangian.xi_y) == num_activated
        assert len(lambda_list) == num_activated
        assert len(xi_list) == num_activated
        assert len(y_squared_polys) == num_activated
        for i in range(num_activated):
            assert subset_lagrangian.lambda_y[i].shape[0] == self.n_u
            assert lambda_list[i].shape[1] == self.n_u
            assert isinstance(subset_lagrangian.xi_y[i], sym.Polynomial)
            assert xi_list[i].shape == (self.xi_lambda_rows,)
            assert y_squared_polys[i].shape == (self.xi_lambda_rows,)

        # construct the sos constraint
        poly_one = sym.Polynomial(sym.Monomial())
        poly_constraint = -poly_one
        # the s0 term
        cbf_vector = (
            np.hstack((
            subset.cbfs[subset.activation_index==1],
            -subset.cbfs[subset.activation_index==0]
             ))
            if num_activated < subset.cbfs.shape[0]
            else subset.cbfs
        )
        poly_constraint -= np.dot(
            subset_lagrangian.cbfs,
            cbf_vector
        )
        # the s_lambda_y terms
        for i in range(num_activated):
            lambda_y_term = lambda_list[i].T @ y_squared_polys[i]
            poly_constraint -= np.dot(
                subset_lagrangian.lambda_y[i],
                lambda_y_term
                )
        # the q_xi_y terms
        for i in range(num_activated):
            xi_y_term = xi_list[i].T @ y_squared_polys[i] + 1
            poly_constraint -= (
                subset_lagrangian.xi_y[i] * (xi_y_term)
                )
        # add the sos constraint to the program
        prog.AddSosConstraint(poly_constraint, sos_type)

    