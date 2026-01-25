# """
# This script provides the code base for verification of union of control 
# barrier functions under switching policy I. Namely Verif-I in the paper.

# The formal discription of the pipeline can be found in Section V-A of 
# the paper.
# Given a set of CBFs {h_1(x), h_2(x), ..., h_{N_h}(x)}. 
# Define the set P = {1, 2, ..., N_h}.
# We let SP be the collection of all non-empty subsets of P.

# The Verif-I has the following pipeline:
# 1. Subset paritioning (Corresponds to Step 1 in Section V-A):
#     Partition the set X_c = {x|h1(x)≥ 0 or h2(x)≥ 0 or ... or hN(x)≥ 0}
#     as the union of disjoint sets 
#     X_N = {x|h_i(x)≥ 0, ∀i∈N; h_j(x) < 0, ∀j∈P\N} for all N in SP.
#     Each subset X_N can be build as an object of the Subset class defined
#     in script "non_empty-subsets.py".

# 2. Empty set pruning (Corresponds to Step 2 in Section V-A):
#     For each N in SP, check if X_N is empty or not. We only keep all the
#     non-empty X_N sets and store them in an array.

# 3. CBF condition verification (Corresponds to Step 3 in Section V-A):
#     For an non-empty subset X_N, we verify if there exists a control input
#     u and an index p in N such that:
#         L_f h_p(x) + L_g h_p(x) u + alpha * h_p(x) > 0, ∀x in X_N
#         Au - b < 0
#     We let Lambda_i(x) and xi_i(x) be the following:
#         Lambda_i(x) = [ L_f h_i(x) + alpha * h_i(x) - eta;
#                         b - eps]
#         xi_i(x) = [ -L_g h_i(x);
#                     A ]
#     For simplicity, we are verifying:
#     For all x in X'_N = {x|h_i(x)≥ 0, ∀i∈N; h_j(x) ≤ 0, ∀j∈P\N},
#     there exists an index p in N and u such that:
#         Lambda_p(x) u ≤ xi_p(x)
    
#     For convenience, we use the following example to show the verification
#     method for the statement above.
#     Example: Consider we have 4 CBFs: h_1, h_2, h_3, h_4, h_5.
#     Let N = {1, 2, 3}, then we are verifying the following:
#         For all x in X'_N = {x|h_1(x)≥ 0, h_2(x)≥ 0, h_3(x)≥ 0, 
#                             h_4(x) ≤ 0, h_5(x) ≤ 0},
#         there exists an index p in {1, 2, 3} and u such that:
#             Λₚ(x) u ≤ ξₚ(x)
    
#     The negation of the above statement is:
#         ∃ x ∈ X'_N such that ∀ p ∈ {1,2,3}, ∄ u, with:
#             Λₚ(x) u ≤ ξₚ(x)
#     By Farkas' lemma, the above statement is equivalent to:
#         ∃ x ∈ X'_N such that ∀ p ∈ {1,2,3}, ∃ yₚ, with:
#             Λₚ(x)ᵀyₚ² = 0 and ξₚ(x)ᵀyₚ² + 1 = 0
#     where yₚ² means each element in the vector yₚ is squared, and yₚ is
#     a vector with dimension = number of rows of ξₚ(x) (also Λₚ(x)).
#     The above statement can be verified by verifying the non-emptiness for
#     the following set:
#     {(x, y₁, y₂, y₃)| 
#         h₁(x) >= 0, h₂(x) >= 0, h₃(x) >= 0, -h₄(x) ≥ 0, -h₅(x) ≥ 0,
#         Λ₁(x)ᵀy₁² = 0, ξ₁(x)ᵀy₁² + 1 = 0,
#         Λ₂(x)ᵀy₂² = 0, ξ₂(x)ᵀy₂² + 1 = 0,
#         Λ₃(x)ᵀy₃² = 0, ξ₃(x)ᵀy₃² + 1 = 0,
#     }
#     Now we can negate the above statement and get the original statement.
#     We wanted to verify the set above is EMPTY by using SOS programming.
#     let y = [y1; y2; y3], the above set is empty if:
#     -1 - s₀(x,y)ᵀ *[h1, h2, h3]ᵀ
#         - s₁(x,y)ᵀΛ₁(x)ᵀy₁² - s₂(x,y)ᵀΛ₂(x)ᵀy₂² - s₃(x,y)ᵀΛ₃(x)ᵀy₃²
#         - q₁(x,y)(ξ₁(x)ᵀy₁² + 1) 
#         - q₂(x,y)(ξ₂(x)ᵀy₂² + 1) 
#         - q₃(x,y)(ξ₃(x)ᵀy₃² + 1)
#     where s0 is a vector of SOS, all the others are Polynomials.
#     If we also have some state equation constraints in the state space of
#     the system, then we will also have one more term (-p(x, y)ᵀeq(x)) in
#     the SOS constraint above, where p(x, y) is a vector of polynomials

#     Now, we summarize the number of SOS and polynomials that are needed to
#     verify the subset X_N in the general case:
#     Assume we have N_h number of CBFs in total. 
#         n number of CBFs are activated in X_N
#         m number of CBFs are deactivated in X_N (m+n=N_h)
#         u is the control input with dimension n_u
#     Then:
#         - s_0: is an SOS vector with N_h elements.
#         - For all p in {1,2,3}, there are n number of polynomial vectors s_p,
#           each with dimension n_u.
#         - there are n number of polynomial q_p, each is a scalar polynomial.
#         - the number of dimentions of y = n * (rows of Λₚ(x))
#     In total, the number of variables is:
#         n_x + n * (rows of Λₚ(x))
# """

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    Degree,
    truth_table,
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    to_lagrangian_impl,
)
from union_cbf_base.non_empty_subset import (
    Subset
)

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
        return SubsetFeasibilityLagrangian(
            cbfs=cbfs_result,
            lambda_y=lambda_y_result,
            xi_y=xi_y_result
        )


@dataclass
class SubsetFeasibilityLagrangianDegrees:
    # the length of the following list is equal to the
    # number of CBFs in the subset.
    cbfs: List[Degree]
    # the length of the outer list is equal to the number of
    # activated CBFs in the subset. The length of the inner list
    # is equal to the control input dimension.
    lambda_y: List[List[Degree]]
    # the length of the following list is equal to the number of
    # activated CBFs in the subset.
    xi_y: List[Degree]

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
            degree=self.cbfs,
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
                degree=self.lambda_y[i],
                lagrangian=(None if lagrangian_lambda_y is None
                            else lagrangian_lambda_y[i])
            ) for i in range(len(self.lambda_y))
        ]
        xi_y = [
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[i], # y_seti
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.xi_y[i],
                lagrangian=(None if lagrangian_xi_y is None
                            else lagrangian_xi_y[i])
            ) for i in range(len(self.xi_y))
        ]
        return SubsetFeasibilityLagrangian(
            cbfs=cbfs,
            lambda_y=lambda_y,
            xi_y=xi_y,
        )


@dataclass
class SubsetGeneralLagrangianDegrees:
    num_control_inputs: int
    cbfs_lagrangian_x_degree: int
    cbfs_lagrangian_y_degree: int
    lambda_lagrangian_x_degree: int
    lambda_lagrangian_y_degree: int
    xi_lagrangian_x_degree: int
    xi_lagrangian_y_degree: int

    def construct_lagrangian_degrees(
        self,
        subset: Subset,
    ) -> SubsetFeasibilityLagrangianDegrees:
        num_total_cbfs = subset.all_polys.shape[0]
        num_non_negative_cbfs = np.sum(subset.activation_index)
        cbfs_lagrangian_degrees = [
            Degree(
                x=self.cbfs_lagrangian_x_degree,
                y=self.cbfs_lagrangian_y_degree,
                c=0
            ) for _ in range(num_total_cbfs)
        ]
        lambda_lagrangian_degrees = [[
            Degree(
                x=self.lambda_lagrangian_x_degree,
                y=self.lambda_lagrangian_y_degree,
                c=0
            ) for _ in range(self.num_control_inputs)
            ] for _ in range(num_non_negative_cbfs)
        ]
        xi_lagrangian_degrees = [
            Degree(
                x=self.xi_lagrangian_x_degree,
                y=self.xi_lagrangian_y_degree,
                c=0
            ) for _ in range(num_non_negative_cbfs)
        ]
        return SubsetFeasibilityLagrangianDegrees(
            cbfs=cbfs_lagrangian_degrees,
            lambda_y=lambda_lagrangian_degrees,
            xi_y=xi_lagrangian_degrees
        )


class UnionCbfI:
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        alpha: float,
        control_limits: Optional[Tuple[np.ndarray, np.ndarray]]
    ):
        """
        arguments:
        - x: the state variable vector
        - f, g: the affine dynamics
        - alpha: the class K function parameter for the CBFs in the union. 
            The data type should be just a float. Since all the CBFs in the union
            share the same alpha parameter. 
        - control_limits: a tuple of (A, c) for control input limits
        """

        assert isinstance(x, np.ndarray)
        self.x = x
        assert f.shape[0] == x.shape[0]
        assert g.shape[0] == x.shape[0]
        self.f = f
        self.g = g
        self.n_x = x.shape[0]
        self.n_u = g.shape[1]
            
        assert alpha is not None
        assert isinstance(alpha, float)
        self.alpha = alpha
        
        if control_limits is not None:
            assert control_limits[0].shape[0] == control_limits[1].shape[0]
            assert g.shape[1] == control_limits[0].shape[1]
            self.control_limits = control_limits
        else:
            self.control_limits = None
    
    def all_possible_subsets(
        self,
        cbfs: np.ndarray,
    ) -> List[Subset]:
        """
        Step 1: 
        Given a set of CBFs {h_1(x), h_2(x), ..., h_{N_h}(x)}.
        generate all possible X_N subsets.
        arguments:
        - cbfs: an array of CBF polynomials with shape (N_h, )
        """
        num_cbfs = cbfs.shape[0]
        # we use the turth_table() function to generate the indecators
        # for all possible non-empty subsets of P = {1, 2, ..., N_h}.
        index_table = truth_table(num_cbfs)
        # The output index_table is an array with shape (2^N_h, N_h), 
        # each row of index_table is a 0-1 vector that indicates a 
        # possible N ⊆ P.
        # For example, if N_h = 3, P={1,2,3}, for the index set N={1,3}
        # the corresponding 0-1 indicator is [1, 0, 1].
        # Noted that the first row of index_table is all zeros, which 
        # indicates the empty index set N = {}. Hence, we remove it from
        # the index_table.
        index_table = np.delete(index_table, obj=0, axis=0)
        # Now we get all the possible X_N with N in SP.
        subsets = []
        for i in range(0, 2**num_cbfs-1):
            activation_index = index_table[i]
            subset = Subset(
                x=self.x,
                all_polys=cbfs,
                activation_index=activation_index
            )
            subsets.append(subset)
        return subsets

    def get_non_empty_subsets(
        self,
        all_subsets: List[Subset],
        *,
        lagrangian_x_degree: int = 2,
        lagrangian_act_c_degree: int = 2,
        lagrangian_deact_c_degree: int = 0,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> List[Subset]:
        """
        Step 2: For all N in SP, check if X_N is empty or not.
        and only keep all the non-empty X_N sets and store them
        in an array.
        """
        non_empty_subsets = []
        for subset in all_subsets:
            if not subset.is_empty(
                lagrangian_x_degree=lagrangian_x_degree,
                lagrangian_act_c_degree=lagrangian_act_c_degree,
                lagrangian_deact_c_degree=lagrangian_deact_c_degree,
                sos_type=sos_type,
            ):
                non_empty_subsets.append(subset)
        return non_empty_subsets

    def construct_feasibility_in_subset_prog(
        self,
        subset: Subset,
        lagrangian_degrees: SubsetGeneralLagrangianDegrees,
        eta: float,
        eps: float,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> solvers.MathematicalProgram:
        """
        This function constructs the SOS program for Step 3
        """
        # construct feasibility lagrangian degrees
        (
            feasibility_lagrangian_degrees
        ) = lagrangian_degrees.construct_lagrangian_degrees(subset=subset)
        # construct necessary variable sets:
        (
            xy_set,
            x_set,
            y_sets,
            y_squared_polys
        ) = self._construct_x_y_sets(subset=subset)
        assert len(y_sets) == int(np.sum(subset.activation_index)) + 1
        # 3. compute the xis and lambdas
        lambda_list,xi_list = self._lambda_xi(
            subset=subset,
            eta=eta,
            eps=eps
        )
        # construct the SOS program
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(xy_set)
        (
            feasibility_lagrangians 
        )= feasibility_lagrangian_degrees.to_lagrangian(
            prog=prog,
            x_set=x_set,
            y_sets=y_sets,
            sos_type=sos_type,
        )
        self._add_feasibility_in_subset_constraint(
            prog=prog,
            subset=subset,
            subset_lagrangian=feasibility_lagrangians,
            lambda_list=lambda_list,
            xi_list=xi_list,
            y_squared_polys=y_squared_polys,
            sos_type=sos_type,
        )
        return prog

    def check_feasibility_in_subset(
        self,
        subset: Subset,
        lagrangian_degrees: SubsetGeneralLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        prog = self.construct_feasibility_in_subset_prog(
            subset=subset,
            lagrangian_degrees=lagrangian_degrees,
            eta=eta,
            eps=eps,
            sos_type=sos_type,
        )
        result = solve_with_id(prog)
        return result.is_success()

    def verfication_fesibility_condition_I(
        self,
        union_cbfs: np.ndarray,
        lagrangian_degrees: SubsetGeneralLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> bool:
        """
        This function realizes the pipeline of Verif-I
        """
        # Step 1: generate all possible subsets
        all_subsets = self.all_possible_subsets(cbfs=union_cbfs)
        # Step 2: filter out empty subsets
        non_empty_subsets = self.get_non_empty_subsets(
            all_subsets=all_subsets,
        )
        # Step 3: check feasibility for each non-empty subset
        for subset in non_empty_subsets:
            if not self.check_feasibility_in_subset(
                subset=subset,
                lagrangian_degrees=lagrangian_degrees,
                eta=eta,
                eps=eps,
                sos_type=sos_type,
            ):
                return False
        return True

    def _lambda_xi(
        self,
        subset: Subset,
        eta: float,
        eps: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        This function computes the lambda and xi terms for all the
        non-negative CBFs in the subset.
        In the output, the fisrt term is the list of lambda matrices,
        the second term is the list of xi vectors.
        """
        non_negative_cbfs = subset.all_polys[subset.activation_index==1]
        lambda_list = []
        xi_list = []
        for h in non_negative_cbfs:
            Lfh = lie_derivative(
                poly=h,
                vector_field=self.f,
                variables=self.x,
                pow=1
                )
            Lgh = lie_derivative(
                poly=h,
                vector_field=self.g,
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
                                c - eps))
            else:
                Lambda = -Lgh.reshape((1, self.n_u))
                xi = np.array([Lfh + self.alpha*h - eta])
            assert (Lambda.shape[0] == self.control_limits[0].shape[0] + 1 
                    if self.control_limits is not None 
                    else 1
                    )
            assert xi.shape[0] == Lambda.shape[0]
            
            lambda_list.append(Lambda)
            xi_list.append(xi)

        return lambda_list, xi_list
    
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
        y_i_size = (
            self.control_limits[0].shape[0] + 1
            if self.control_limits is not None 
            else 1
            )
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

    def _add_feasibility_in_subset_constraint(
        self,
        prog: solvers.MathematicalProgram,
        subset: Subset,
        subset_lagrangian: SubsetFeasibilityLagrangian,
        lambda_list: List[np.ndarray],
        xi_list: List[np.ndarray],
        y_squared_polys: List[np.ndarray],
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        """
        This function adds the SOS feasibility constraint for Step 3,
        which is:
        -1 - s₀(x,y)ᵀ *[h1, h2, h3]ᵀ
            - s₁(x,y)ᵀΛ₁(x)ᵀy₁² - s₂(x,y)ᵀΛ₂(x)ᵀy₂² - s₃(x,y)ᵀΛ₃(x)ᵀy₃²
            - q₁(x,y)(ξ₁(x)ᵀy₁² + 1) 
            - q₂(x,y)(ξ₂(x)ᵀy₂² + 1) 
            - q₃(x,y)(ξ₃(x)ᵀy₃² + 1) is SOS,
        where s0 is a vector of SOS, all the others are Polynomials.
        """
        # basic checks:
        num_non_neative_cbfs = int(np.sum(subset.activation_index))
        assert len(subset_lagrangian.lambda_y) == num_non_neative_cbfs
        assert len(subset_lagrangian.xi_y) == num_non_neative_cbfs
        assert len(lambda_list) == num_non_neative_cbfs
        assert len(xi_list) == num_non_neative_cbfs
        assert len(y_squared_polys) == num_non_neative_cbfs
        for i in range(num_non_neative_cbfs):
            assert subset_lagrangian.lambda_y[i].shape[0] == self.n_u
            assert lambda_list[i].shape[1] == self.n_u
            assert isinstance(subset_lagrangian.xi_y[i], sym.Polynomial)
            assert xi_list[i].shape[0] == lambda_list[i].shape[0]
            assert y_squared_polys[i].shape == (xi_list[i].shape[0],)

        # construct the sos constraint
        poly_one = sym.Polynomial(sym.Monomial())
        poly_constraint = -poly_one
        # the s0 term
        cbf_vector = (
            np.hstack((
            subset.all_polys[subset.activation_index==1],
            -subset.all_polys[subset.activation_index==0]
             ))
            if num_non_neative_cbfs < subset.all_polys.shape[0]
            else subset.all_polys
        )
        poly_constraint -= np.dot(
            subset_lagrangian.cbfs,
            cbf_vector
        )
        # the s_lambda_y terms
        for i in range(num_non_neative_cbfs):
            lambda_y_term = lambda_list[i].T @ y_squared_polys[i]
            poly_constraint -= np.dot(
                subset_lagrangian.lambda_y[i],
                lambda_y_term
                )
        # the q_xi_y terms
        for i in range(num_non_neative_cbfs):
            xi_y_term = xi_list[i].T @ y_squared_polys[i] + 1
            poly_constraint -= (
                subset_lagrangian.xi_y[i] * (xi_y_term)
                )
        # add the sos constraint to the program
        prog.AddSosConstraint(poly_constraint, sos_type)
        



