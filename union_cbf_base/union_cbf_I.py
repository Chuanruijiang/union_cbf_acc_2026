"""
This script provides the code base for verification of union of control 
barrier functions under switching policy I. Namely Verif-I in the paper 
(both the journal and conference paper).

The formal discription of the pipeline can be found in Section V-A of 
the confrence paper (normal relative degree CBF) and the Section 5.1 of
the journal extension (high relative degree HOCBF).

The Verif-I framework verifies the feasibilty of switching HOCBF-QP using
the following steps:
1. Set partition:
    We partition the forward invariant set into sever subsets with each 
    subset has a unique switching CBF avaliability (activation) condition,
    meaning that, at all the state points in the a subset, the same group
    of switching CBFs are non-negative. Such a kind of subset is modeled 
    by the class "Subset" in non-empty-subset.py. 
2. Emptiness checking:
    We list out all possible switching CBF avaliability cases, for each case, 
    we check whether the corresponding subset is empty. We keep the subsets 
    that are non-empty and discard all the empty subsets. 
    The finial non-empty subsets form an valid partition result for the
    forward invariant set.
3. Feasibility verification:
    We verify feasibilty of the switching HOCBF-QP in each of the non-empty
    subset. The feasibility checking program is shown by Theorem 4, 
    Program (8) in the conference paper (reltive degree 1 CBFs). 
    Or the Theorem 3, program (12) in the journal extension (HOCBFs).

The following lagranigan and lagrangian degrees are applicable to both CBFs 
and HOCBFs. Since CBFs are also HOCBFs with relative degree 1. Hence, the
Program (8) in Theorem 4 of the conference paper is a special case of the 
Program (12) in Theorem 3 of the journal extenstion. So, in this script,
we stick to the Program (12) of the journal paper.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import time
import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    Degree,
    truth_table,
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    lower_lie_derivatives,
    to_lagrangian_impl,
    all_possible_sequences,
    elementary_symetric_polynomials
)
from union_cbf_base.non_empty_subset import (
    Subset
)

@dataclass
class SubsetFeasibilityLagrangian:
    """
    Note: in the definition of the Subset class, we said
    the deactivated_polys are the polys that are p(x)<0
    but we regard it as p(x) <=0 during the verification
    to establish a simpler and sufficient condition. See
    statements T2 and T3 in the Section V-A of the 
    conference paper and statements T1 and T2 in Section
    5.1 of the journal paper.
    We define the Lagrangians in Program (12) in this
    class.
    """
    
    # s_i(x, y) for all i in I plus
    # s_p(x, y) for all p in N:
    # the list length should be |I| + |N| and each array's
    # size should be the relative degree of the corresponding
    # HOCBF.
    activated_poly_groups: List[np.ndarray]
    # s_p'(x, y) for all p' in bar_N:
    # if the subset has deactivated HOCBFs, then size of this
    # array should be |bar_N|. 
    deactivated_polys: Optional[np.ndarray]
    # q_p(x, y) for all p in N:
    # the list size should be |N|, each array's size should be
    # the number of control inputs for the verified system.
    lambda_y: List[np.ndarray]
    # r_p(x, y) for all p in N:
    # the list size should be |N|.
    xi_y: List[sym.Polynomial]
    # if we have state quation constraints e(x)=0:
    # the array size should be the number of state equality
    # constraints.
    state_eq: Optional[np.ndarray]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        activated_poly_groups_result = [
            get_polynomial_result(
                result=result,
                p=self.activated_poly_groups[i],
                coefficient_tol=coefficient_tol
            ) for i in range(len(self.activated_poly_groups))
        ]        
        deactivated_polys_result = (
            get_polynomial_result(
                result=result,
                p=self.deactivated_polys,
                coefficient_tol=coefficient_tol
            ) 
            if self.deactivated_polys is not None else None
        )
        lambda_y_result = [
            get_polynomial_result(
                result=result,
                p=self.lambda_y[i],
                coefficient_tol=coefficient_tol
            ) for i in range(len(self.lambda_y))
        ]
        xi_y_result = [
            get_polynomial_result(
                result=result,
                p=self.xi_y[i],
                coefficient_tol=coefficient_tol
            ) for i in range(len(self.xi_y))
        ]
        state_eq_result = (
            get_polynomial_result(
                result=result,
                p=self.state_eq,
                coefficient_tol=coefficient_tol
                )
        if self.state_eq is not None else None
        )
        return SubsetFeasibilityLagrangian(
            activated_poly_groups=activated_poly_groups_result,
            deactivated_polys=deactivated_polys_result,
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            state_eq=state_eq_result
        )

@dataclass
class SubsetFeasibilityLagrangianDegrees:
    activated_poly_groups_degree: List[List[Degree]]
    deactivated_polys_degree: Optional[List[Degree]]
    lambda_y_degree: List[List[Degree]]
    xi_y_degree: List[Degree]
    state_eq_degree: Optional[List[Degree]]

    def to_lagrangian(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        y_sets: List[sym.Variables],
        *,
        lagrangian_activated_poly_groups: Optional[List[np.ndarray]] = None,
        lagrangian_deactivated_polys: Optional[List[np.ndarray]] = None,
        lagrangian_lambda_y: Optional[List[np.ndarray]] = None,
        lagrangian_xi_y: Optional[List[sym.Polynomial]] = None,
        lagrangian_state_eq: Optional[np.ndarray] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        activated_poly_groups = [
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[-1],
                c=None,
                sos_type=sos_type,
                is_sos=True,
                degree=self.activated_poly_groups_degree[i],
                lagrangian=(None if lagrangian_activated_poly_groups is None
                            else lagrangian_activated_poly_groups[i])
            ) for i in range(len(self.activated_poly_groups_degree))
        ]
        deactivated_polys = to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[-1],
                c=None,
                sos_type=sos_type,
                is_sos=True,
                degree=self.deactivated_polys_degree,
                lagrangian=(None if lagrangian_deactivated_polys is None
                            else lagrangian_deactivated_polys)
        )
        lambda_y = [
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[i],
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.lambda_y_degree[i],
                lagrangian=(None if lagrangian_lambda_y is None
                            else lagrangian_lambda_y[i])
            ) for i in range(len(self.lambda_y_degree))
        ]
        xi_y = [
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[i], 
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.xi_y_degree[i],
                lagrangian=(None if lagrangian_xi_y is None
                            else lagrangian_xi_y[i])
                ) for i in range(len(self.xi_y_degree))
        ]
        state_eq = (
            to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_sets[-1],
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.state_eq_degree,
                lagrangian=(None if lagrangian_state_eq is None
                            else lagrangian_state_eq)
            )
            if self.state_eq_degree is not None else None
        )
        return SubsetFeasibilityLagrangian(
            activated_poly_groups=activated_poly_groups,
            deactivated_polys=deactivated_polys,
            lambda_y=lambda_y,
            xi_y=xi_y,
            state_eq=state_eq
        )

@dataclass
class SubsetGeneralLagrangianDegrees:
    # this class provides a simpler Lagrangian
    # degree specification interface.
    num_control_inputs: int
    activated_lagrangian_x_degree: int
    activated_lagrangian_y_degree: int
    deactivated_lagrangian_x_degree: int
    deactivated_lagrangian_y_degree: int
    lambda_lagrangian_x_degree: int
    lambda_lagrangian_y_degree: int
    xi_lagrangian_x_degree: int
    xi_lagrangian_y_degree: int
    state_eq_lagrangian_x_degree: Optional[int]
    state_eq_lagrangian_y_degree: Optional[int]

    def construct_lagrangian_degrees(
        self,
        subset: Subset,
    ) -> SubsetFeasibilityLagrangianDegrees:
        num_activated_switching_cbfs = subset.num_avaliable_switching_cbfs
        num_activated_groups = len(subset.activated_poly_groups)
        num_deactivated_polys = (
            0 if subset.deactivated_polys is None 
            else subset.deactivated_polys.shape[0]
        )
        activated_degrees = [[
            Degree(
                x=self.activated_lagrangian_x_degree,
                y=self.activated_lagrangian_y_degree,
                c=0
            ) for _ in range(subset.activated_poly_groups[i].shape[0])
            ] for i in range(num_activated_groups)
        ]
        deactivated_degrees = ([
            Degree(
                x=self.deactivated_lagrangian_x_degree,
                y=self.deactivated_lagrangian_y_degree,
                c=0
            ) for _ in range(num_deactivated_polys)
            ] if num_deactivated_polys > 0 else None
        )
        lambda_degrees = [[
            Degree(
                x=self.lambda_lagrangian_x_degree,
                y=self.lambda_lagrangian_y_degree,
                c=0
            ) for _ in range(self.num_control_inputs)
            ] for _ in range(num_activated_switching_cbfs)
        ]
        xi_degrees = [
            Degree(
                x=self.xi_lagrangian_x_degree,
                y=self.xi_lagrangian_y_degree,
                c=0
            ) for _ in range(num_activated_switching_cbfs)
        ]
        state_eq_degrees = ([
            Degree(
                x=self.state_eq_lagrangian_x_degree,
                y=self.state_eq_lagrangian_y_degree,
                c=0
                ) for _ in range(subset.equation_constraints.shape[0])
            ] if self.state_eq_lagrangian_x_degree is not None and \
                    self.state_eq_lagrangian_y_degree is not None 
            else None
        )
        return SubsetFeasibilityLagrangianDegrees(
            activated_poly_groups_degree=activated_degrees,
            deactivated_polys_degree=deactivated_degrees,
            lambda_y_degree=lambda_degrees,
            xi_y_degree=xi_degrees,
            state_eq_degree=state_eq_degrees
        )


class UnionCbfI:
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
        We categorize the CBFs as static CBFs and switching CBF.
        Both of them are defined as array of polynomials. These polynomials
        are not required in the definition of this class, but they will be input
        as arguments to method functions of this class.

        This class supports the following type of Union CBF verification:
        Consider the QP has multiple CBF constraints and one of them switches
        between different CBFs. (Of course, it is also possible to have only one
        switching HOCBF/CBF in the QP constraints and no static HOCBF/CBF
        constraints hence the static CBFs are optional.)
        In this case, we verify a single union of CBFs. We specify the relative 
        degree of both static and switching CBFs and also assume that all the 
        switching CBFs share the same relative degree and the same set of alpha
        parameters. 
        
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
        assert isinstance(alpha, list)
        assert relative_degree is not None
        assert isinstance(relative_degree, list)
        assert len(alpha) == len(relative_degree)
        for i in range(len(alpha)):
            assert isinstance(alpha[i], list)
            assert len(alpha[i]) == relative_degree[i]
        self.alpha = alpha
        self.relative_degree = relative_degree
        
        if control_limits is not None:
            assert control_limits[0].shape[0] == control_limits[1].shape[0]
            assert g.shape[1] == control_limits[0].shape[1]
            self.control_limits = control_limits
            self.control_limit_rows = control_limits[0].shape[0]
        else:
            self.control_limits = None
            self.control_limit_rows = 0

        if state_eq_constr is not None:
            assert isinstance(state_eq_constr, np.ndarray)
            assert len(state_eq_constr.shape) == 1
        self.state_eq_const = state_eq_constr
    
    # The following methods are designed to get all the possible 
    # subsets to be verified, namely the X_N^(bar_i) in the journal.
    # each X_N^(bar_i) is modeled by the Subst set class.
    def _all_phi_polys_given_h(
        self,
        cbf: sym.Polynomial,
        relative_degree: int,
        alphas: List[float],
        ) -> np.ndarray:
        """
        Given an HOCBF h(x) with relative degree r,
        All the lower lie derivative polys are:
            phi^(0) = h(x)
            phi^(1) = Lfphi^(0) + alpha(1)phi^(0)
            phi^(2) = Lfphi^(1) + alpha(2)phi^(1)
            ...
            phi^(r-1) = Lfphi^(r-2) + alpha(r-1)phi^(r-2)
        This function returns the array of polynomials:
            [ phi^(0), phi^(1),...,phi^(r-1) ]
        """
        phis_with_out_h = lower_lie_derivatives(
            poly=cbf,
            vector_field=self.f,
            variables=self.x,
            relative_degree=relative_degree,
            betas=alphas
        )
        phis = np.concatenate(
            (np.array([cbf]), phis_with_out_h), axis=0
        )
        return phis

    def _all_phi_polys_for_all_cbfs(
        self,
        switching_cbfs: np.ndarray,
        static_cbfs: Optional[np.ndarray],
        ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        This function computes all the lower lie derivatives
        for all the given static and switching HOCBFs. We let
        all the lower lie derivatives of the static HOCBFs be phi(x),
        all the lower lie derivatives of the switching HOCBFs be psi(x)
        
        The output should be a tuple of two elements:
        1. an array of all the psi(x) of all the switching CBFs.
        2. a list of all the phi(x) for the static CBFs
        
        Since all the switching HOCBFs shares the same relative
        degree and same alpha parameters, then the array size should
        be (num_switching_HOCBF, r_b) where r_b is the shared
        relative degree.

        Since the relative degrees and alpha parameters of the
        static HOCBFs are different, then the phi(x) polys should be
        stored in a list of arrays. The list length should be the 
        number of static HOCBFs, while each array element of the list
        has size r_i, which is the relative degree of the i-th static
        HOCBF.
        """
        num_switching_cbfs = switching_cbfs.shape[0]
        all_psi_polys = np.empty(
            (num_switching_cbfs, self.relative_degree[0]), 
            dtype=sym.Polynomial
        )
        for i in range(num_switching_cbfs):
            all_psi_polys[i, :] = self._all_phi_polys_given_h(
                cbf=switching_cbfs[i],
                relative_degree=self.relative_degree[0],
                alphas=self.alpha[0]
            )
        
        if static_cbfs is None:
            return (all_psi_polys, None)
        else: 
            all_phi_polys = []
            num_static_cbfs = static_cbfs.shape[0]
            for j in range(num_static_cbfs):
                all_phi_polys.append(self._all_phi_polys_given_h(
                    cbf=static_cbfs[j],
                    relative_degree=self.relative_degree[j + 1],
                    alphas=self.alpha[j + 1]
                ))
            return (all_psi_polys, all_phi_polys)

    def all_possible_subsets(
        self,
        switching_cbfs: np.ndarray,
        static_cbfs: Optional[np.ndarray],
        ) -> List[np.ndarray]:
        """
        This function corresponds to the step 1 introduced in the section 5.1
        of the journal paper.
        This function returns a list of arrays of Subset objects. Each single
        Subset object corresponds to a X_N^(bar_i) in the paper. The array 
        represents all the X_N^(bar_i) for all bar_i in S_N^(i), while the list
        corresponds to all the N in S^P. 
        The length of the list is equivalent to the number of all the possible
        subsets of P={1,2,3,...,N_b}. 
        The size of each array is equal to all possible bar_i in S_N^(i).
        """

        # Generating all possible X_N in the step 1 of section
        # 5.1 of the journal paper.
        # We achieve this using a truth table of a specific length
        # Given N_b number of switching CBFs, the index set of the
        # switching CBFs is P={1,2,3,...,N_b}. Each possible subset
        # N contains some indecies from P. We use each line of the
        # truth table (which is a 0-1 vector) to represent which
        # index in P is in the current N and which are not. Hence,
        # we call each line of the truth table a "mask" for an N.
        # "all_switching_mask" is a reprentative of S^P in the paper.
        # since N should not be empty, then we discard the 0 line
        # of the truth table.
        num_switching_cbfs = switching_cbfs.shape[0]
        all_switching_mask = truth_table(num_switching_cbfs)
        all_switching_mask = np.delete(all_switching_mask, obj=0, axis=0)

        # The complete Verif-I requires verification under every
        # possible N. The number of indeterminates in the SOS program
        # increases as the number of chosen indecies in N increases,
        # so as the computation time of the verification. 
        # Hence, we first check the subset that requires the most 
        # number of indeterminates in the SOS program. This can help
        # us know whether the verification is feasible at early stage.
        all_switching_mask = np.flip(all_switching_mask, axis=0)

        # Generating all the lower lie derivative functions for all
        # static and switching HOCBFs.
        (   all_switching_psi_polys, 
            all_static_phi_polys
        ) = self._all_phi_polys_for_all_cbfs(
            switching_cbfs=switching_cbfs,
            static_cbfs=static_cbfs
        )

        # As shown in Section 5.1, each subset X_N should be further
        # viewed as a union of X_N^(bar_i). It is each X_N^(bar_i) that
        # verifiable for SOS programs. Hence, we next construct every
        # X_N^(bar_i) for every X_N using the Subset class.
        all_possible_subsets = []
        for each_mask in all_switching_mask: 
            # for each N in S^P, we generate all the X_N^(bar_i) for 
            # each X_N:

            # total number of elements in N: 
            num_activated = int(np.sum(each_mask))
            # number of elements in P\N:
            num_deactivated = num_switching_cbfs - num_activated
            # get the actual switching CBF indices N from the mask:
            activated_indices = np.where(each_mask == 1)[0]
            
            # If the set P\N is empty:
            if num_deactivated == 0:
                # In this case, bar_N = P\N is empty and N=P, there
                # does not exits p' in P\N, hence i_p'and bar_i are 
                # undefined. We only have the X_N.
                activated_poly_groups = [
                    all_switching_psi_polys[index] 
                    for index in activated_indices
                ]
                if static_cbfs is not None:
                    for static_phi_polys in all_static_phi_polys:
                        activated_poly_groups.append(static_phi_polys)
                subset_component = Subset(
                    x=self.x,
                    activated_poly_groups=activated_poly_groups,
                    deactivated_polys=None,
                    equation_constraints=self.state_eq_const,
                    num_avaliable_switching_cbfs=num_activated,
                    mask_avaliable_switching_cbfs=each_mask,
                    subset_component_indicator=None,
                )
                subset_components = np.array(
                    [subset_component], 
                    dtype=Subset
                )
            # If the set P\N is not empty:
            elif num_deactivated > 0:
                # In this case, bar_N = P\N is non-empty and we have
                # some p' in bar_N, hence i_p' and bar_i should have
                # a valid definition.
                deactivated_indices = np.where(each_mask == 0)[0]

                # Generate the set S_N^(i), which is the range space of bar_i.
                # we use "subset_component_indecator" to represent a "bar_i"
                # the definition of S_N^(i), bar_i and i_p' are in the section
                # 5.1, step 1 of the journal paper.
                (all_subset_components_indecators
                ) = all_possible_sequences(
                    num_elements=num_deactivated,
                    num_possible_values_for_each_element=self.relative_degree[0]
                )

                # reserve space for all the subset X_N^(bar_i) of a specific X_N
                subset_components = np.empty(
                    (all_subset_components_indecators.shape[0],), 
                    dtype=Subset
                    )
                for i in range(all_subset_components_indecators.shape[0]):
                    # for each of the possible bar_i vector, we create the
                    # corresponding subset component. 
                    bar_i = all_subset_components_indecators[i]
                    
                    # we first put psi functions of every switching CBFs bp(x),
                    # for p∈ N, in to the activated_poly_group.
                    activated_poly_groups = [
                        all_switching_psi_polys[index] 
                        for index in activated_indices
                    ]
                    
                    # if we also have static CBFs, then we need to
                    # also add their phi functions to the activated_poly_group
                    if static_cbfs is not None:
                        for phis_static in all_static_phi_polys :
                            activated_poly_groups.append(phis_static)
                    
                    # the deactivated polynomials are:
                    # [ψₚ'⁽ⁱᵖ'⁾(x) < 0, ∀p'∈ P\N], 
                    # hence we also reserve the space
                    deactivated_poly_array = np.empty(
                        (num_deactivated,), 
                        dtype=sym.Polynomial
                        )
                    
                    # Now, we go through all the indices p'∈ P\N,
                    # load the corresponding ϕₚ'⁽ⁱᵖ'⁾(x) from the all_switching_psi
                    # array using the index p' and the bar_i sequence.
                    for j in range(num_deactivated):
                        # add the deactivated polynomials: 
                        # ψₚ'⁽ⁱᵖ'⁾(x) < 0,
                        deactivated_poly_array[j] = all_switching_psi_polys[
                            deactivated_indices[j], bar_i[j]
                        ]
                    
                    # create the X_N^(bar_i) using Subset class:
                    subset_component = Subset(
                        x=self.x,
                        activated_poly_groups=activated_poly_groups,
                        deactivated_polys=deactivated_poly_array,
                        equation_constraints=self.state_eq_const,
                        num_avaliable_switching_cbfs=num_activated,
                        mask_avaliable_switching_cbfs=each_mask,
                        subset_component_indicator=bar_i,
                    )

                    # Collecting every X_N^(bar_i) for the specific X_N
                    # create the array of X_N^(bar_i) for all ipp in S_N^(i).
                    subset_components[i] = subset_component
            
            # collect all the arrays of subsets X_N^(bar_i) for all X_N.
            all_possible_subsets.append(subset_components)
            
        return all_possible_subsets

    def get_non_empty_subsets(
        self,
        all_subsets: List[np.ndarray],
        *,
        lagrangian_x_degree: int = 2,
        lagrangian_act_c_degree: int = 2,
        lagrangian_deact_c_degree: int = 0,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> List[Subset]:
        """
        This function corresponds to the step 2 of the Verif-I.
        For each of the subset objects, we use the .is_empty()
        method to verify the emptiness.
        We break the pack of list of arrays of subsets, and
        return just a list of all non-empty subset objects.
        """
        all_non_empty_subsets = []
        for each_subset_array in all_subsets:
            for each_subset in each_subset_array:
                empty_flag = each_subset.is_empty(
                    lagrangian_x_degree = lagrangian_x_degree,
                    lagrangian_act_c_degree = lagrangian_act_c_degree,
                    lagrangian_deact_c_degree = lagrangian_deact_c_degree,
                )
                if not empty_flag:
                    all_non_empty_subsets.append(each_subset)                   
        if len(all_non_empty_subsets) == 0:
            raise ValueError("All the subsets are empty!")
        
        return all_non_empty_subsets

    # The following methods defines verificaiton of a single subset
    # for Verif-I
    def construct_feasibility_check_prog(
        self,
        subset: Subset,
        general_degrees: SubsetGeneralLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        output_lagragians: bool = False,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Union[
            solvers.MathematicalProgram,
            Tuple[solvers.MathematicalProgram, SubsetFeasibilityLagrangian]
            ]:
        """
        This function constructs the feasibility verification program (12)
        in Theorem 3, Section 5.1 of the journal extension. The program 
        does not achieve the funtionality of Verif-I, but to verify the 
        feasibility in a single subset X_N^(bar_i).
        """
        # 1. construct the lagrangian degrees
        (
            lagrangian_degrees
        ) = general_degrees.construct_lagrangian_degrees(subset=subset)
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
            eps=eps
        )

        # create the program and sepcify the indeterminates
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(xy_set)

        # checking dimentions:
        num_activated_switching_cbf = subset.num_avaliable_switching_cbfs
        assert len(y_sets) == num_activated_switching_cbf + 1

        # 4. create the lagrangian multipliers
        subset_lagrangian = lagrangian_degrees.to_lagrangian(
            prog=prog,
            x_set=x_set,
            y_sets=y_sets,
            sos_type=sos_type
        )
        # 5. add_sos_constraints
        # This constraint is shown in (12b) of program (12)
        # in Theorem 3 of the journal paper. 
        self._add_subset_feasibility_constraint(
            prog=prog,
            subset=subset,
            subset_lagrangian=subset_lagrangian,
            lambda_list=lambda_list,
            xi_list=xi_list,
            y_squared_polys=y_squared_polys
        )
        if output_lagragians:
            return (prog, subset_lagrangian)
        else:
            return prog

    def check_feasibility_in_subset(
        self,
        subset: Subset,
        general_degrees: SubsetGeneralLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        record_time: bool = True,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Tuple[bool, Optional[float]]:
        prog = self.construct_feasibility_check_prog(
            subset=subset,
            general_degrees=general_degrees,
            eta=eta,
            eps=eps,
            sos_type=sos_type,
        )
        if record_time:
            start_time = time.time()
        result = solve_with_id(prog)
        if record_time:
            end_time = time.time()
            print(f"Time taken for feasibility check: {end_time - start_time} seconds")
            return result.is_success(), end_time - start_time
        return result.is_success(), None

    def verification_feasibility_condition_I(
        self,
        switching_cbfs: np.ndarray,
        static_cbfs: Optional[np.ndarray],
        general_degrees: SubsetGeneralLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        show_output: bool = True,
        show_verification_time: bool = True,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> Tuple[bool, Optional[float]]:
        """
        This function realizes the pipeline of Verif-I the sudo code is
        shown in Algorithm 1 of the journal paper.
        """
        # Step 1: generate all possible subsets
        all_subsets = self.all_possible_subsets(
            switching_cbfs=switching_cbfs,
            static_cbfs=static_cbfs
        )
        
        # Step 2: filter out empty subsets
        non_empty_subsets = self.get_non_empty_subsets(
            all_subsets=all_subsets,
        )
        if show_output:
            print(
                f"Total number of non-empty subsets: {len(non_empty_subsets)}"
            )
        
        # Step 3: check feasibility for each non-empty subset
        total_verif_time = 0
        for subset in non_empty_subsets:
            if show_output:
                print(
                    f"Checking feasibility condition in subset X_N^(bar_i) with \
                    N mask: {subset.mask_avaliable_switching_cbfs}"
                )
                print(
                    f"The bar_i of the current subset is: \
                    {subset.subset_component_indicator}"
                )
            (feasible_in_subset, verif_time) = self.check_feasibility_in_subset(
                subset=subset,
                general_degrees=general_degrees,
                eta=eta,
                eps=eps,
                record_time=show_verification_time,
                sos_type=sos_type,
            )
            total_verif_time += (verif_time if verif_time is not None else 0)
            if not feasible_in_subset:
                if show_output:
                    print(
                        "Feasibility condition failed in this subset."
                    )
                return False, total_verif_time
        if show_output:
            print("Feasibility condition passed for all non-empty subsets.")
        return True, total_verif_time

    def _lambda_xi(
        self,
        subset: Subset,
        eta: float,
        eps: float
    ) -> Union[
            Tuple[List[np.ndarray], List[np.ndarray]],
            Tuple[np.ndarray, np.ndarray]
        ]:
        """
        This function computes the lambda and xi terms for all the
        activated CBFs in the subset.
        For the output, the fisrt term is the list of lambda matrices,
        the second term is the list of xi vectors.
        For each subset object, the activated_poly_groups has the following
        structure:
        [
            (first part of the list):
            phis for avaliable switching CBFs,
            (if any, second part of the list):
            phis for static CBFs,
            (if any, last part of the list):
            activated_phis for deactivated switching CBFs
        ]
        """
        num_activated_switching_cbfs = subset.num_avaliable_switching_cbfs
        activated_switching_cbfs = [
            subset.activated_poly_groups[i][0]
            for i in range(num_activated_switching_cbfs)
        ]
        Lambda_list = []
        xi_list = []
        for each_cbf in activated_switching_cbfs:
            Lr_1fh = lie_derivative(
                poly=each_cbf,
                vector_field=self.f,
                variables=self.x,
                pow=self.relative_degree[-1] - 1
            )
            phi_r_lambda_part = lie_derivative(
                poly=Lr_1fh,
                vector_field=self.g,
                variables=self.x,
                pow=1 
            )
            Lambda_p=-phi_r_lambda_part.reshape(1, -1)

            # compute xi element for HOCBF
            alphas_vector = elementary_symetric_polynomials(self.alpha[-1])
            lie_derivatives = np.array([
                lie_derivative(
                    poly=each_cbf,
                    vector_field=self.f,
                    variables=self.x,
                    pow=j
                )
                for j in range(self.relative_degree[-1], -1, -1)
            ])
            xi_p = np.dot(alphas_vector, lie_derivatives)

            # add static cbf parts into the lambda and xi
            num_active_poly_groups = len(subset.activated_poly_groups)
            num_static_cbfs = num_active_poly_groups - num_activated_switching_cbfs
            if num_static_cbfs != 0:
                for i in range(num_static_cbfs):
                    # the first few groups of polys in 
                    # subset.activated_poly_groups
                    # are polys for switching CBFs.
                    # the second part of activated_poly_groups
                    # are polys for static CBFs.
                    static_cbf_poly = subset.activated_poly_groups[
                        num_activated_switching_cbfs + i
                        ][0]
                    Lr_1fh_normal = lie_derivative(
                        poly=static_cbf_poly,
                        vector_field=self.f,
                        variables=self.x,
                        pow=self.relative_degree[i] - 1
                    )
                    phi_r_lambda_part_normal = lie_derivative(
                        poly=Lr_1fh_normal,
                        vector_field=self.g,
                        variables=self.x,
                        pow=1 
                    )
                    Lambda_p = np.vstack((
                        Lambda_p,
                        -phi_r_lambda_part_normal.reshape(1, -1)
                    ))

                    # compute xi element for static CBF
                    alphas_vector_static = elementary_symetric_polynomials(self.alpha[i])
                    lie_derivatives_static = np.array([
                        lie_derivative(
                            poly=static_cbf_poly,
                            vector_field=self.f,
                            variables=self.x,
                            pow=j
                        )
                        for j in range(self.relative_degree[i], -1, -1)
                    ])
                    xi_p = np.hstack((
                        xi_p,
                        np.dot(alphas_vector_static, lie_derivatives_static)
                    ))

            # add control limits into the lambda and xi
            if self.control_limits is not None:
                A = self.control_limits[0]
                c = self.control_limits[1]
                assert Lambda_p.shape[1] == self.n_u
                assert Lambda_p.shape[1] == A.shape[1]
                Lambda = np.vstack((Lambda_p, A))
                xi = np.hstack((xi_p - eta, c - eps))
                assert Lambda.shape[0] == (
                    self.control_limit_rows + num_static_cbfs + 1
                    )
                assert xi.shape[0] == (
                    self.control_limit_rows + num_static_cbfs + 1
                    )
                assert Lambda.shape[1] == self.n_u
            else:
                Lambda = Lambda_p
                xi = xi_p - eta
                assert Lambda.shape[0] == num_static_cbfs + 1
                assert Lambda.shape[1] == self.n_u
            
            Lambda_list.append(Lambda)
            xi_list.append(xi)
        return Lambda_list, xi_list
    
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
        function has 4 parts:
        1. xy_set as the total indeterminates for SOS program.
        2. x_set for creating lagrangian multipliers.
        3. y_set for creating lagrangian multipliers.
            For the inner structure of y_sets, please check the
            explanation for function "to_lagrangian" in class
            "SubsetFeasibilityLagrangianDegrees".
        4. the squared y polynomial variables for Λ_p(x)ᵀy_p^2 and
            ξ_p(x)ᵀy_p^2 + 1 terms.
        """
        num_active_poly_group = len(subset.activated_poly_groups)
        num_activated_switching_cbf = subset.num_avaliable_switching_cbfs
        num_static_cbfs = num_active_poly_group - num_activated_switching_cbf
        
        y_i_size = self.control_limit_rows + 1
        if num_static_cbfs != 0:
            y_i_size += num_static_cbfs
        
        y_i_groups = []
        y_all = np.array([])
        y_sets = []
        y_squared_polys = []
        for i in range(num_activated_switching_cbf):
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
        if num_activated_switching_cbf == 1:
            y_sets.append(None)
        elif num_activated_switching_cbf > 1:   
            for i in range(num_activated_switching_cbf):
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

    def _add_subset_feasibility_constraint(
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
        This function add the sos constraint (12b) to the program (12)
        """
    
        # basic checks:
        num_activated_switching_cbf = subset.num_avaliable_switching_cbfs
        num_activated_groups = len(subset.activated_poly_groups)
        assert len(subset_lagrangian.activated_poly_groups) == num_activated_groups
        assert len(subset_lagrangian.lambda_y) == num_activated_switching_cbf
        assert len(subset_lagrangian.xi_y) == num_activated_switching_cbf
        assert isinstance(subset_lagrangian.activated_poly_groups, list)
        for each_poly_group in subset_lagrangian.activated_poly_groups:
            assert isinstance(each_poly_group, np.ndarray)       
        if subset.deactivated_polys is not None:
            assert subset_lagrangian.deactivated_polys is not None
        else:
            assert subset_lagrangian.deactivated_polys is None
        assert len(lambda_list) == num_activated_switching_cbf
        assert len(xi_list) == num_activated_switching_cbf
        assert len(y_squared_polys) == num_activated_switching_cbf
        for i in range(num_activated_switching_cbf):
            assert subset_lagrangian.lambda_y[i].shape == (self.n_u,)

        
        # construct the sos constraint
        poly_one = sym.Polynomial(sym.Monomial())
        poly_constraint = -poly_one
        
        # include the term sum_activated_polys_lagrangian * activated_polys:
        for i in range(num_activated_groups):
            activated_lagrangians = subset_lagrangian.activated_poly_groups[i]
            activated_polys = subset.activated_poly_groups[i]
            poly_constraint -= activated_lagrangians.dot(activated_polys)
        
        # include the term sum_deactivated_polys_lagrangian * deactivated_polys:
        if subset.deactivated_polys is not None:
            deactivated_lagrangians = subset_lagrangian.deactivated_polys
            deactivated_polys = subset.deactivated_polys
            poly_constraint += deactivated_lagrangians.dot(deactivated_polys)

        # include the term ∑_{p ∈ N}( qₚ(x, y)ᵀΛₚᵀ(x)yₚ² )
        for i in range(num_activated_switching_cbf):
            lambda_term = lambda_list[i].T @ y_squared_polys[i]
            poly_constraint -= subset_lagrangian.lambda_y[i].dot(lambda_term)
        
        # include the term ∑_{p ∈ N}( rₚ(x, y)ᵀ(ξₚᵀ(x)yₚ² + 1) )
        for i in range(num_activated_switching_cbf):
            xi_y_term = xi_list[i].dot(y_squared_polys[i]) + 1
            poly_constraint -= subset_lagrangian.xi_y[i] * (xi_y_term)
        
        # if we also have state equation constraints in the state space,
        # we should also add one more terms 
        if self.state_eq_const is not None:
            poly_constraint -= subset_lagrangian.state_eq.dot(
                self.state_eq_const
            )
        # add the sos constraint to the program
        prog.AddSosConstraint(poly_constraint, sos_type)
        


