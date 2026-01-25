# """
# This script provides the code base for verification of union of control
# barrier functions under the swtiching policy II. Namely, the Verif-II
# in the paper. 

# The formal description of the pipeline of the verification can be found
# in the Section V-B of the paper.
# Given a set of CBFs {h_1(x), ..., h_{N_h}(x)}
# We define the set P = {1, 2, ..., N_h}.
# For each p in P, we define the set X_p = {x | h_p(x) >= 0}.
# Also for each p in P, we define lambda_p(x) and xi_p(x) as:

# lambda_p(x) = [-L_gh_p(x); A]
# xi_p(x) = [L_fh_p(x) + alpha(h_p(x)) - eta; b - epsilon]

# The verification of pipeline is the following:
# For each p in P:
#     Verify: For all x in X_p, there exists a control input u in U such
#     that
#         lambda_p(x) * u <= xi_p(x)

# For each p in P, the verification above can be done by solving the SOS:
#     -1 - s(x, y)*h_p(x) - p(x, y)^T * lambda_p(x)^T * y^2  
#                         - q(x, y) * (xi_p(x)^T * y^2 + 1) is SOS
# where s(x, y) is SOS, p(x, y) is a vector of polynomials, and q(x, y)
# is a free polynomial.
# """

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    lower_lie_derivatives,
    elementary_symetric_polynomials,
    Degree,
    to_lagrangian_impl,
)
from union_cbf_base.non_empty_subset import (
    Subset
)

@dataclass
class CbfFeasibilityLagrangian:
    """
    This class stores the s(x, y), p(x, y), q(x, y) 
    lagrangian polynomials.
    """
    cbf: sym.Polynomial
    lambda_y: np.ndarray
    xi_y: sym.Polynomial

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        cbf_result = get_polynomial_result(
            result, self.cbf, coefficient_tol
        )
        lambda_y_result = get_polynomial_result(
            result, self.lambda_y, coefficient_tol
        )
        xi_y_result = get_polynomial_result(
            result, self.xi_y, coefficient_tol
        )
        return CbfFeasibilityLagrangian(
            cbf=cbf_result,
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
        )

@dataclass
class CbfFeasibilityLagrangianDegrees:
    """
    This class stores the degree information of 
    s(x, y), p(x, y), q(x, y) lagrangian polynomials.
    """
    cbf: Degree
    lambda_y: List[Degree]
    xi_y: Degree

    def to_lagrangian(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        y_set: sym.Variables,
        *,
        lagrangian_cbf: Optional[sym.Polynomial] = None,
        lagrangian_lambda_y: Optional[np.ndarray] = None,
        lagrangian_xi_y: Optional[sym.Polynomial] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> CbfFeasibilityLagrangian:
        cbf_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.cbf,
            lagrangian=lagrangian_cbf
        )
        lambda_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y
        )
        xi_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.xi_y,
            lagrangian=lagrangian_xi_y
        )
        return CbfFeasibilityLagrangian(
            cbf=cbf_lagrangian,
            lambda_y=lambda_y_lagrangian,
            xi_y=xi_y_lagrangian,
        )

class UnionCbfII:
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

    def construct_cbf_feasibility_prog(
        self,
        cbf: sym.Polynomial,
        lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
        eta: float,
        eps: float,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> solvers.MathematicalProgram:
        prog = solvers.MathematicalProgram()
        (
            x_set, y_set, xy_set, y_squared_polys
        ) = self._construct_x_y_sets()

        prog.AddIndeterminates(xy_set)

        lagrangians = lagrangian_degrees.to_lagrangian(
            prog=prog,
            x_set=x_set,
            y_set=y_set,
            sos_type=sos_type
        )
        lambda_mat, xi_vector = self._xi_lambda(
            alpha=self.alpha,
            cbf=cbf,
            eta=eta,
            eps=eps,
        )
        self._add_cbf_feasibility_constraint(
            prog=prog,
            cbf=cbf,
            lambda_mat=lambda_mat,
            xi_vector=xi_vector,
            lagrangians=lagrangians,
            y_squared_polys=y_squared_polys,
            sos_type=sos_type,
        )
        return prog

    def check_cbf_feasibility(
        self,
        cbf: sym.Polynomial,
        lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
        eta: float,
        eps: float,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        ):
        prog = self.construct_cbf_feasibility_prog(
            cbf=cbf,
            lagrangian_degrees=lagrangian_degrees,
            eta=eta,
            eps=eps,
            sos_type=sos_type,
        )
        result = solve_with_id(prog)
        return result.is_success()
        
    def verification_feasibility_condition_II(
        self,
        union_cbfs: np.ndarray,
        lagrangian_degrees: CbfFeasibilityLagrangianDegrees,
        eta: float,
        epsilon: float,
        *,
        show_output: bool = True,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        for idx, cbf in enumerate(union_cbfs):
            is_feasible = self.check_cbf_feasibility(
                cbf=cbf,
                lagrangian_degrees=lagrangian_degrees,
                eta=eta,
                epsilon=epsilon,
                sos_type=sos_type,
            )
            if not is_feasible:
                if show_output:
                    print(f"CBF feasibility condition failed for CBF index {idx}.")
                return False
        if show_output:
            print("CBF feasibility condition passed for all CBFs.")
        return True
        
    def _xi_lambda(
        self,
        alpha: float,
        cbf: sym.Polynomial,
        eta: float,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:

        num_rows = (
            self.control_limits[0].shape[0] + 1
            if self.control_limits is not None
            else 1
        )
        lambda_mat = np.empty((num_rows, self.n_u), dtype=object)
        xi_vector = np.empty((num_rows,), dtype=object)

        # xi part
        # compute [Lf h(x) + alpha * h(x) - eta]:
        xi_vector[0] = (
            lie_derivative(
                poly=cbf,
                vector_field=self.f,
                variables=self.x,
                pow=1,
            )
            + alpha * cbf
            - eta
        )
        # lambda part
        # compute [-Lg h(x)]:
        lambda_mat[0, :] = -lie_derivative(
            poly=cbf,
            vector_field=self.g,
            variables=self.x,
            pow=1,
        )
        # add control limit parts if any
        if self.control_limits is not None:
            lambda_mat[1:, :] = self.control_limits[0]
            xi_vector[1:] = self.control_limits[1] - eps
        
        return lambda_mat, xi_vector

    def _construct_x_y_sets(
        self,
    ) -> Tuple[
        sym.Variables,
        sym.Variables,
        sym.Variables,
        np.ndarray
    ]:
        """
        This function constructs the x and y indeterminate variable 
        sets. The ouput should be:
        Tuple[x_set, y_set, xy_sets, y_squared_polys]
        """
        y_size = (
            self.control_limits[0].shape[0] + 1
            if self.control_limits is not None
            else 1
        )
        y= sym.MakeVectorContinuousVariable(y_size, "y")
        xy = np.hstack((self.x, y))
        x_set = sym.Variables(self.x)
        y_set = sym.Variables(y)
        xy_set = sym.Variables(xy)
        y_squared_polys = np.array([
            sym.Polynomial(sym.Monomial(y[i], 2)) 
            for i in range(y_size)
            ])
        return x_set, y_set, xy_set, y_squared_polys

    def _add_cbf_feasibility_constraint(
        self,
        prog: solvers.MathematicalProgram,
        cbf: sym.Polynomial,
        lambda_mat: np.ndarray,
        xi_vector: np.ndarray,
        lagrangians: CbfFeasibilityLagrangian,
        y_squared_polys: np.ndarray,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ):
        """
        Add the following SOS constraint:
        -1 - s(x, y)*h(x) - p(x, y)^T * lambda(x)^T * y^2  
                            - q(x, y) * (xi(x)^T * y^2 + 1) is SOS
        """
        # Construct the polynomial expression for the constraint
        poly_one = sym.Polynomial(1)
        poly = -poly_one

        # s(x, y) * h(x) term
        poly -= lagrangians.cbf * cbf

        # p(x, y)^T * lambda(x)^T * y^2 term
        poly -= lagrangians.lambda_y.dot(
            lambda_mat.T @ y_squared_polys
        )
        # q(x, y) * (xi(x)^T * y^2 + 1) term
        poly -= lagrangians.xi_y * (
            xi_vector.dot(y_squared_polys) + poly_one
        )
        # Add the SOS constraint to the program
        prog.AddSosConstraint(p=poly, type=sos_type)












