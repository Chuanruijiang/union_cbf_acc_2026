"""
This script provides the code base for verification of union of control
barrier functions under the swtiching policy II. Namely, the Verif-II
in the paper. 

The formal description of the pipeline of the verification can be found
in the Section V-B of the conference paper (relative degree 1) and 
Section 5.2 of the journal paper (high relative degrees).

Let the index set of the switching CBFs b(x) be P, and let the index set 
of the static CBFs h(x) be I. We would like to verify that for all p in P,
for all x in the set 
    { h_1(x)>=0,...,h_{N_h}(x)>=0, b_p(x)>=0 }, 
there exists a control u in U such that Λ_p(x)u <= ξ_p(x) - epsilon.

If the CBFs are high relative degrees, then every h_i(x) in the set should
become a vector of phi_i(x) polynomials, and the b_p(x) should be \psi_p(x)
polynomials. Hence, the verif-II framework, like verif-I, also verifies 
feasibility for each Subset class object. The main difference is that the
Subset objects for verif-II always have only activated poly groups, and we
don't need to consider deactivated polys.

The following lagranigan and lagrangian degrees are applicable to both CBFs 
and HOCBFs. Since CBFs are also HOCBFs with relative degree 1. Hence, the
Program (9) in Theorem 5 of the conference paper is a special case of the 
Program (13) in Theorem 4 of the journal extenstion. So, in this script,
we stick to the Program (13) of the journal paper.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import time 
import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    Degree,
    to_lagrangian_impl,
    elementary_symetric_polynomials
)
from union_cbf_base.union_cbf_I import UnionCbfI
from union_cbf_base.non_empty_subset import Subset

@dataclass
class CbfFeasibilityLagrangian:
    """
    This class stores the s_i(x, y), s_p(x, y), q(x, y), r(x, y) 
    lagrangian polynomials.
    """
    # s_i(x, y) for all i in I plus
    # s_p(x, y) is the vector of lagrangians for current psi_p(x)
    # in the subset. 
    # The list length should be |I| + 1 and each array's
    # size should be the relative degree of the corresponding
    # HOCBF (since the vector of a lower lie derivative phi arrary
    # should be phi^(0) to phi^(r-1)).
    phis: List[np.ndarray]
    # q(x, y) for the lambda term of the current p in the subset:
    # the array's size should be number of control inputs for 
    # the verified system.
    lambda_y: np.ndarray
    # r(x, y) for the xi term of the current p in the subset.
    xi_y: sym.Polynomial
    # if we have state equation constriants.
    state_eq: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        phis_result = [
            get_polynomial_result(
                result=result,
                p=each_phi,
                coefficient_tol=coefficient_tol
            ) for each_phi in self.phis
        ]
        lambda_y_result = get_polynomial_result(
            result=result,
            p=self.lambda_y,
            coefficient_tol=coefficient_tol
        )
        xi_y_result = get_polynomial_result(
            result=result,
            p=self.xi_y,
            coefficient_tol=coefficient_tol
        )
        if self.state_eq is not None:
            state_eq_result = get_polynomial_result(
                result=result,
                p=self.state_eq,
                coefficient_tol=coefficient_tol
            )
        else:
            state_eq_result = None
        
        return CbfFeasibilityLagrangian(
            phis=phis_result,
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            state_eq=state_eq_result
        )

@dataclass
class CbfFeasibilityLagrangianDegrees:
    phis: List[List[Degree]]
    lambda_y: List[Degree]
    xi_y: Degree
    state_eq: Optional[List[Degree]]
    def to_lagrangian(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        y_set: sym.Variables,
        *,
        lagrangian_phis: Optional[List[np.ndarray]] = None,
        lagrangian_lambda_y: Optional[np.ndarray] = None,
        lagrangian_xi_y: Optional[sym.Polynomial] = None,
        lagrangian_state_eq: Optional[np.ndarray] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        phis = [
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_set,
                c=None,
                sos_type=sos_type,
                is_sos=True,
                degree=each_phi_degree,
                lagrangian=(None if lagrangian_phis is None
                            else lagrangian_phis[i])
                ) for i, each_phi_degree in enumerate(self.phis)
            ]
        lambda_y = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y
        )
        xi_y = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.xi_y,
            lagrangian=lagrangian_xi_y
        )
        state_eq = (
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_set,
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.state_eq,
                lagrangian=lagrangian_state_eq
            )
            if self.state_eq is not None
            else None
        )
        return CbfFeasibilityLagrangian(
            phis=phis,
            lambda_y=lambda_y,
            xi_y=xi_y,
            state_eq=state_eq
        )

class UnionCbfII(UnionCbfI):
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        alpha: List[List[float]],
        relative_degree:List[int],
        control_limits: Optional[Tuple[np.ndarray, np.ndarray]],
        state_eq_constr: Optional[np.ndarray] = None
    ):
        """
        arguments:
        - x: the state variable vector
        - f, g: the affine dynamics
        - alphas: the class K function parameters for the switching HOCBFs and
            static CBFs. The data type should be a List of lists. 
            The outter length is the number of all static CBFs plus one switching
            CBF. The inner length is equal to the relative degree of each static
            HOCBF and switching HOCBF.
            We let the first list in the outter list to be the alpha parameters
            for the switching CBF, the other lists are the alpha parameters for 
            the static CBFs. 
        - relative_degrees: The shared relative degree of all the switching and
            static CBFs or HOCBFs. The length of the list should be number
            of all the static CBFs plus one switching CBF.
            We let the first element to be the relative degree of the switching
            CBF, the other elements are the relative degrees of static CBFs.
        - control_limits: a tuple of (A, c) for control input limits
        - state_eq_const: an array of polynomials that defines the
            state equation constraints in the state space.
        Note: The basic members of this class is similar to UnionCbfI, so we 
        inherate it to create this class.
        """
        super().__init__(
            x=x,
            f=f,
            g=g,
            alpha=alpha,
            relative_degree=relative_degree,
            control_limits=control_limits,
            state_eq_constr=state_eq_constr
        )
        
    def all_subsets_to_verify(
        self,
        switching_cbfs: np.ndarray,
        static_cbfs: Optional[np.ndarray] = None,
    ) -> List[Subset]: 
        """
        Given all the switching CBFs and the static CBFs, this 
        function returns a list of all subsets to be verified. 
        Each of the subsets should be in the form of:
            {x | phi_1(x)>=0, ..., phi_{N_h}(x)>=0, psi_p(x)>=0}
        Hence each subset only have activatied poly groups and 
        is defined for each p in P. Hence, the length of the
        output list should be the number of all switching CBFs
        |P|.
        Note: for each subset, the deactivated_polys and the 
        subset_component_indecator (bar_i) are None since these
        concepts are only defined for Verif-I. 
        """
        num_switching_cbfs = switching_cbfs.shape[0]
        num_static_cbfs = (
            static_cbfs.shape[0] 
            if static_cbfs is not None else 0
        )
        assert len(self.relative_degree) == num_static_cbfs + 1

        # started to create the list of subsets
        subsets = []
        for i in range(0, num_switching_cbfs):
            (   all_psi_switching,
                all_phi_static
            ) = self._all_phi_polys_for_all_cbfs(
                switching_cbfs=switching_cbfs,
                static_cbfs=static_cbfs
            )
            activated_poly_groups = [all_psi_switching[i,:]]
            if all_phi_static is not None:
                assert static_cbfs is not None
                activated_poly_groups.extend(all_phi_static)

            mask_switching_cbfs = np.zeros(num_switching_cbfs, dtype=int)
            mask_switching_cbfs[i] = 1  
            new_subset = Subset(
                x=self.x,
                activated_poly_groups=activated_poly_groups,
                deactivated_polys=None,
                equation_constraints=self.state_eq_const,
                num_avaliable_switching_cbfs=1,
                mask_avaliable_switching_cbfs=mask_switching_cbfs,
                subset_component_indicator=None,
            )
            subsets.append(new_subset)
        return subsets

    def construct_cbf_feasibility_prog(
        self,
        subset: Subset,
        lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        output_lagrangian_tags: bool = False,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Union[
            solvers.MathematicalProgram,
            Tuple[solvers.MathematicalProgram, CbfFeasibilityLagrangian]
            ]:
        """
        This function constructs the verification program (13) in the
        journal extension.
        """
        # basic checks
        num_cbf = len(subset.activated_poly_groups)
        assert len(lagrangian_degrees.phis) == num_cbf
        assert len(lagrangian_degrees.lambda_y) == self.n_u
        for i in range(num_cbf):
            assert len(lagrangian_degrees.phis[i]) == self.relative_degree[i]
        if self.state_eq_const is not None:
            assert lagrangian_degrees.state_eq is not None
            assert len(lagrangian_degrees.state_eq) == self.state_eq_const.shape[0]
        else:
            assert lagrangian_degrees.state_eq is None

        # construct indeterminate variables
        y_num = self.control_limit_rows + num_cbf
        y = sym.MakeVectorContinuousVariable(rows=y_num, name="y")
        y_squared_poly = np.array([
            sym.Polynomial(sym.Monomial(y[i], 2)) 
            for i in range(y_num)
            ])
        x_set = sym.Variables(self.x)
        y_set = sym.Variables(y)
        xy = np.concatenate((self.x, y))
        xy_set = sym.Variables(xy)
        
        # construct the program:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(xy_set)
        unsolved_lagrangians = lagrangian_degrees.to_lagrangian(
            prog=prog,
            x_set=x_set,
            y_set=y_set,
            sos_type=sos_type,
        )
        # compute lambda and xi
        lambda_matrix, xi_vector = self._xi_lambda_(
            subset=subset,
            eta=eta,
            eps=eps,
        )
        # construct the intersection sos constraint.
        # this function builds the constraint (13b)
        # in the journal extension
        self._add_cbf_feasibility_constraints_(
            prog=prog,
            subset=subset,
            lambda_mat=lambda_matrix,
            xi_vector=xi_vector,
            lagrangians=unsolved_lagrangians,
            y_squared_poly=y_squared_poly
        )
        if output_lagrangian_tags:
            return (prog, unsolved_lagrangians)
        else:
            return prog    

    def check_cbf_feasibility(
        self,
        subset: Subset,
        lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        output_computed_lagrangians: bool = False,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Union[Tuple[bool, float], Tuple[bool, float, CbfFeasibilityLagrangian]]:
        if output_computed_lagrangians:
            (
                prog, unsolved_lagrangians
            ) = self.construct_cbf_feasibility_prog(
                subset=subset,
                lagrangian_degrees=lagrangian_degrees,
                eta=eta,
                eps=eps,
                sos_type=sos_type
            )
            start_time = time.time()
            result = solve_with_id(prog)
            end_time = time.time()

            if result.is_success():
                computed_lagrangians = unsolved_lagrangians.get_result(
                    result=result,
                    coefficient_tol=1e-6
                )
                return (True, end_time - start_time, computed_lagrangians)
            else:
                return (False, end_time - start_time, None)
        else:
            prog = self.construct_cbf_feasibility_prog(
                subset=subset,
                lagrangian_degrees=lagrangian_degrees,
                eta=eta,
                eps=eps,
                sos_type=sos_type
            )
            start_time = time.time()
            result = solve_with_id(prog)
            end_time = time.time()
            
            return (result.is_success(), end_time - start_time)
        
    def verification_feasibility_condition_II(
        self,
        switching_cbfs: np.ndarray,
        static_cbfs: Optional[np.ndarray],
        lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        show_output: bool = True,
        show_computation_time: bool = True,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> bool:
        """
        This function defines the overall framework of the Verif-II.
        The pseudo code is shown by Algorithm 2 in the paper.
        """
        if show_output:
            print("Starting verification of all subsets...")
            if static_cbfs is not None:
                print(f"Number of static CBFs: {static_cbfs.shape[0]}")
            print(f"Number of switching CBFs: {switching_cbfs.shape[0]}")
        
        # get all the subsets to be verified.
        all_subsets = self.all_subsets_to_verify(
            switching_cbfs=switching_cbfs,
            static_cbfs=static_cbfs
        )
        total_computation_time = 0.0

        # for each subset, do CBF-QP feasibility verification
        for i in range(len(all_subsets)):
            subset = all_subsets[i]
            if show_output:
                print(f"Verifying subset {i+1} / {len(all_subsets)}...")
            (
                feasible_flag, computation_time
            ) = self.check_cbf_feasibility(
                subset=subset,
                lagrangian_degrees=lagrangian_degrees,
                eta=eta,
                eps=eps,
                sos_type=sos_type
            )
            total_computation_time += computation_time
            if not feasible_flag:
                if show_output:
                    print(f"Subset {i+1} is not feasible.")
                return False
        if show_output:    
            print("All subsets are feasible.")
        if show_computation_time:
            print(f"Total computation time for all subsets: {total_computation_time} seconds.")
        return True
        
    def _xi_lambda_(
        self,
        subset: Subset,
        eta: float,
        eps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_cbfs = len(subset.activated_poly_groups)
        for i in range(num_cbfs):
            assert len(subset.activated_poly_groups[i]) == self.relative_degree[i] 
        assert subset.deactivated_polys is None

        num_rows = self.control_limit_rows + num_cbfs
        lambda_mat = np.empty((num_rows, self.n_u), dtype=object)
        xi_vector = np.empty((num_rows,), dtype=object)

        for i in range(num_cbfs):
            current_cbf = subset.activated_poly_groups[i][0]
            current_r = self.relative_degree[i]
            # xi part
            alpha_vector = elementary_symetric_polynomials(self.alpha[i])
            lie_derivative_vector = np.empty(
                shape=(current_r + 1,), dtype=sym.Polynomial
            )
            lie_derivative_vector[-1] = current_cbf
            for j in range(current_r, 0, -1):
                lie_derivative_vector[j - 1] = lie_derivative(
                    poly=lie_derivative_vector[j],
                    vector_field=self.f,
                    variables=self.x,
                    pow=1,
                )
            xi_element = np.dot(lie_derivative_vector, alpha_vector)
            xi_vector[i] = xi_element - eta
            # lambda part
            # compute Lf⁽ʳ⁻¹⁾Lgh(x) = ∂Lf⁽ʳ⁻¹⁾h(x)/∂x * g(x):
            LfLgb = lie_derivative(
                poly=lie_derivative_vector[1],
                vector_field=self.g,
                variables=self.x,
                pow=1,
            )
            lambda_element = -LfLgb
            lambda_mat[i] = lambda_element
        # add control limit parts if any
        if self.control_limits is not None:
            lambda_mat[num_cbfs:, :] = self.control_limits[0]
            xi_vector[num_cbfs:] = self.control_limits[1] - eps
        
        return lambda_mat, xi_vector

    def _add_cbf_feasibility_constraints_(
        self,
        prog: solvers.MathematicalProgram,
        subset: Subset,
        lambda_mat: np.ndarray,
        xi_vector: np.ndarray,
        lagrangians: CbfFeasibilityLagrangian,
        y_squared_poly: np.ndarray,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        """
        Add SOS constraints (13b) to program
        """
        poly_one = sym.Polynomial(1)
        poly = -poly_one
        # add the cbf and lower order time derivative terms
        num_cbf = len(subset.activated_poly_groups)
        for i in range(num_cbf):
            phis_terms = subset.activated_poly_groups[i]
            poly -= lagrangians.phis[i].dot(phis_terms)
        # add the lambda_y terms
        lambda_y_term = lambda_mat.T @ y_squared_poly
        poly -= lagrangians.lambda_y.dot(lambda_y_term)
        # add the xi_y terms
        xi_y_term = xi_vector.dot(y_squared_poly)
        poly -= lagrangians.xi_y * (xi_y_term + poly_one)
        # if there are state equations:
        if self.state_eq_const is not None:
            poly -= lagrangians.state_eq.dot(self.state_eq_const)
        # add the sos constraint
        prog.AddSosConstraint(p=poly, type=sos_type)












