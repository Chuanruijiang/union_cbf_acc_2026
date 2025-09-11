from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing_extensions import Self

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym

from union_cbf_base.utils import (
    truth_table,
    get_polynomial_result,
    solve_with_id,
    lie_derivative,
    lower_lie_derivatives,
    elementary_symetric_polynomials,
    Degree,
    to_lagrangian_impl,
    is_sos,
    check_polynomial_arrays_equal,
    BackoffScale
)

"""
In this file, we consider relative degree two CBFs and verify the
union of these kind of CBFs under the switching policy of Theorem 3
in the paper.
First of all if we have a relative degree two CBF h(x), then we have
the following condition for the CBF:
∀ x ∈ {x| h(x)≥0, ̇h(x) + α₁h(x)≥0}, ∃ u ∈ U s.t. 
    ̈h(x,u) + α₁̇h(x) + α₂̇h(x) + α₁α₂h(x)≥0
assume the system dynamics is: ẋ = f(x) + g(x)u, also because h(x) is
a relative degree two CBF, then we have:
    ̇h(x) = L_f h(x) + L_gh(x)u = L_f h(x)
    ̈h(x,u) = L_f² h(x) + L_fL_gh(x)u
Thus the verification condition above is equivalent to:
∀ x ∈ {x| h(x)≥0, L_f h(x) + α₁h(x)≥0}, ∃ u ∈ U s.t. 
    L_f² h(x) + L_fL_gh(x)u + α₁L_f h(x) + α₂L_f h(x) + α₁α₂h(x)≥0
Putting them into the form of Λ and ξ, we have:
Λ(x) = [
    -L_fL_gh(x)
    A
]
ξ(x) = [
    L_f² h(x) + α₁L_f h(x) + α₂L_f h(x) + α₁α₂h(x)-η 
    c - ϵ
]
where Au≤c is the admissible control space and η,ϵ are the given positive
constants.
Hence, the verification is:
∀ x ∈ {x| h(x)≥0, L_f h(x) + α₁h(x)≥0}, ∃ u s.t. Λ(x)u≤ξ(x)
By Farkas' lemma, the above condition is equivalent to:
∀ x ∈ {x| h(x)≥0, L_f h(x) + α₁h(x)≥0}, ∄ y s.t. Λ(x)ᵀy²=0, ξ(x)ᵀy²+1=0
this is equivalent to say that the following set is empty:

S = { (x,y) | h(x)≥0, L_f h(x) + α₁h(x)≥0,
                Λ(x)ᵀy²=0, ξ(x)ᵀy²+1=0 }

The SOS conditions for that is:

-1 - s_cbf(x,y)h(x) - s_cbf_dot(x,y)(L_f h(x) + α₁h(x))
    - s(x,y)ᵀΛ(x)ᵀy² - q(x,y)(ξ(x)ᵀy²+1) is SOS

    where s_cbf(x,y), s_cbf_dot(x,y) are SOS polynomials,
s(x,y) is a vector of polynomials, and q(x,y) is a polynomial.
    (1). s(x,y) is a vector of polynomials with m number of elements,
        where m is the number of control inputs.
    (2). y is also a vector of variables with l number of elements,
        where l is the number of linear constraints in the Λ(x)u≤ξ(x)
        (i.e., the number of rows in Λ(x) and ξ(x)).
"""

@dataclass
class OrderTwoCBFLagrangians:
    cbf: sym.Polynomial
    cbf_dot: sym.Polynomial
    # an array of polynomials of shape (m, ), where m is the number of
    # control inputs
    s: np.ndarray
    q: sym.Polynomial
    state_eq: Optional[sym.Polynomial]

    def get_result(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        cbf_result = get_polynomial_result(
            result=result,
            p=self.cbf,
            coefficient_tol=coefficient_tol,
        )
        cbf_dot_result = get_polynomial_result(
            result=result,
            p=self.cbf_dot,
            coefficient_tol=coefficient_tol,
        )
        s_result = get_polynomial_result(
            result=result,
            p=self.s,
            coefficient_tol=coefficient_tol,
        )
        q_result = get_polynomial_result(
            result=result,
            p=self.q,
            coefficient_tol=coefficient_tol,
        )
        return OrderTwoCBFLagrangians(
            cbf=cbf_result,
            cbf_dot=cbf_dot_result,
            s=s_result,
            q=q_result,
        )
    
@dataclass
class OrderTwoCBFLagrangianDegrees:
    cbf: Degree
    cbf_dot: Degree
    # an list of degrees of length m, where m is the number of
    # control inputs
    s: List[Degree]
    q: Degree
    state_eq: Optional[Degree]

    def to_lagrangian(
        self,
        prog: solvers.MathematicalProgram,
        x_set: sym.Variables,
        y_set: sym.Variables,
        *,
        lagragian_cbf: Optional[sym.Polynomial] = None,
        lagragian_cbf_dot: Optional[sym.Polynomial] = None,
        lagragian_s: Optional[np.ndarray] = None,
        lagragian_q: Optional[sym.Polynomial] = None,
        lagrangian_state_eq: Optional[sym.Polynomial] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> OrderTwoCBFLagrangians:
        cbf = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.cbf,
            lagrangian=lagragian_cbf
        )
        cbf_dot = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.cbf_dot,
            lagrangian=lagragian_cbf_dot
        )
        s = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.s,
            lagrangian=lagragian_s
        )
        q = to_lagrangian_impl(
            prog=prog,
            x=x_set,
            y=y_set,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.q,
            lagrangian=lagragian_q
        )
        if self.state_eq is not None:
            state_eq = to_lagrangian_impl(
                prog=prog,
                x=x_set,
                y=y_set,
                c=None,
                sos_type=sos_type,
                is_sos=False,
                degree=self.state_eq,
                lagrangian=lagrangian_state_eq
            )
        else:
            state_eq = None
        return OrderTwoCBFLagrangians(
            cbf=cbf,
            cbf_dot=cbf_dot,
            s=s,
            q=q,
            state_eq=state_eq
        )

class UnionHighOrderCBF:
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        cbfs: np.ndarray,
        alpha1: float,
        alpha2: float,
        control_limits: Tuple[np.ndarray, np.ndarray],
        state_eq_constraint: Optional[sym.Polynomial]
    ):
        # basic checks
        assert isinstance(x, np.ndarray)
        assert isinstance(cbfs, np.ndarray)
        assert isinstance(cbfs[0], sym.Polynomial)
        assert f.shape[0] == x.shape[0]
        assert g.shape[0] == x.shape[0]
        assert control_limits[0].shape[0] == control_limits[1].shape[0]        
        # although A and c are control input limits presented by constants,
        # we should also encode them as polynomials to be consistent with
        # Lfh and Lgh when computing Lambda and xi.
        assert isinstance(control_limits[0][0][0], sym.Polynomial)
        assert isinstance(control_limits[1][0], sym.Polynomial)
        assert g.shape[1] == control_limits[0].shape[1]

        self.x = x
        self.f = f
        self.g = g
        self.cbfs = cbfs
        self.alphas = [alpha1, alpha2]
        self.control_limits = control_limits
        self.state_eq_constraint = state_eq_constraint
        self.n_x = x.shape[0]
        self.n_u = g.shape[1]
        self.n_h = cbfs.shape[0]

    def construct_check_single_cbf_prog(
        self,
        cbf_index: int,
        lagrangian_cbf_x_degree: int,
        lagrangian_cbf_y_degree: int,
        lagrangian_cbf_dot_x_degree: int,
        lagrangian_cbf_dot_y_degree: int,
        lagrangian_s_x_degree: int,
        lagrangian_s_y_degree: int,
        lagrangian_q_x_degree: int,
        lagrangian_q_y_degree: int,
        eta: float,
        epsilon: float,
        *,
        lagrangian_state_eq_x_degree: Optional[int] = None,
        lagrangian_state_eq_y_degree: Optional[int] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> solvers.MathematicalProgram:
        
        x_set = sym.Variables(self.x)
        (y_set, y_squared, xy_set) = self._construct_xy_set()
        assert len(y_set) == 1 + self.control_limits[0].shape[0]
        assert y_squared.shape[0] == 1 + self.control_limits[0].shape[0]
        
        lagrangian_degrees = self._construct_lagrangian_degrees(
            lagrangian_cbf_x_degree=lagrangian_cbf_x_degree,
            lagrangian_cbf_y_degree=lagrangian_cbf_y_degree,
            lagrangian_cbf_dot_x_degree=lagrangian_cbf_dot_x_degree,
            lagrangian_cbf_dot_y_degree=lagrangian_cbf_dot_y_degree,
            lagrangian_s_x_degree=lagrangian_s_x_degree,
            lagrangian_s_y_degree=lagrangian_s_y_degree,
            lagrangian_q_x_degree=lagrangian_q_x_degree,
            lagrangian_q_y_degree=lagrangian_q_y_degree,
            lagrangian_state_eq_x_degree=lagrangian_state_eq_x_degree,
            lagrangian_state_eq_y_degree=lagrangian_state_eq_y_degree
        )
        
        lambda_mat, xi = self._lambda_xi(
            cbf=self.cbfs[cbf_index],
            eta=eta,
            epsilon=epsilon
        )

        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(xy_set)
        lagrangians = lagrangian_degrees.to_lagrangian(
            prog=prog,
            x_set=x_set,
            y_set=y_set,
        )
        
        cbf_dot_term = lower_lie_derivatives(
            poly=self.cbfs[cbf_index],
            vector_field=self.f,
            variables=self.x,
            relative_degree=2,
            betas=self.alphas
        )
        self._add_sos_constraint(
            prog=prog,
            cbf=self.cbfs[cbf_index],
            cbf_dot=cbf_dot_term[0],
            lambda_mat=lambda_mat,
            xi=xi,
            y_squared=y_squared,
            lagrangians=lagrangians,
            sos_type=sos_type,
        )
        return prog

    def check_single_cbf(
        self,
        cbf_index: int,
        lagrangian_cbf_x_degree: int,
        lagrangian_cbf_y_degree: int,
        lagrangian_cbf_dot_x_degree: int,
        lagrangian_cbf_dot_y_degree: int,
        lagrangian_s_x_degree: int,
        lagrangian_s_y_degree: int,
        lagrangian_q_x_degree: int,
        lagrangian_q_y_degree: int,
        eta: float,
        epsilon: float,
        *,
        lagrangian_state_eq_x_degree: Optional[int] = None,
        lagrangian_state_eq_y_degree: Optional[int] = None,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> bool:
        prog = self.construct_check_single_cbf_prog(
            cbf_index=cbf_index,
            lagrangian_cbf_x_degree=lagrangian_cbf_x_degree,
            lagrangian_cbf_y_degree=lagrangian_cbf_y_degree,
            lagrangian_cbf_dot_x_degree=lagrangian_cbf_dot_x_degree,
            lagrangian_cbf_dot_y_degree=lagrangian_cbf_dot_y_degree,
            lagrangian_s_x_degree=lagrangian_s_x_degree,
            lagrangian_s_y_degree=lagrangian_s_y_degree,
            lagrangian_q_x_degree=lagrangian_q_x_degree,
            lagrangian_q_y_degree=lagrangian_q_y_degree,
            eta=eta,
            epsilon=epsilon,
            lagrangian_state_eq_x_degree=lagrangian_state_eq_x_degree,
            lagrangian_state_eq_y_degree=lagrangian_state_eq_y_degree,
            sos_type=sos_type,
        )
        result = solve_with_id(prog)
        return result.is_success()

    def verification_union_cbf_theorem_3(
        self,
        lagrangian_cbf_x_degree: int,
        lagrangian_cbf_y_degree: int,
        lagrangian_cbf_dot_x_degree: int,
        lagrangian_cbf_dot_y_degree: int,
        lagrangian_s_x_degree: int,
        lagrangian_s_y_degree: int,
        lagrangian_q_x_degree: int,
        lagrangian_q_y_degree: int,
        eta: float,
        epsilon: float,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    )->bool:
        num_cbfs = self.cbfs.shape[0]
        verification_flag = True
        for i in range(num_cbfs):
            flag = self.check_single_cbf(
                cbf_index=i,
                lagrangian_cbf_x_degree=lagrangian_cbf_x_degree,
                lagrangian_cbf_y_degree=lagrangian_cbf_y_degree,
                lagrangian_cbf_dot_x_degree=lagrangian_cbf_dot_x_degree,
                lagrangian_cbf_dot_y_degree=lagrangian_cbf_dot_y_degree,
                lagrangian_s_x_degree=lagrangian_s_x_degree,
                lagrangian_s_y_degree=lagrangian_s_y_degree,
                lagrangian_q_x_degree=lagrangian_q_x_degree,
                lagrangian_q_y_degree=lagrangian_q_y_degree,
                eta=eta,
                epsilon=epsilon,
                sos_type=sos_type,
            )
            if flag:
                print(f"The {i}-th CBF is verified.")
            else:
                print(f"The {i}-th CBF fails to be verified.")
                verification_flag = False
        return verification_flag

    def _lambda_xi(
        self,
        cbf: sym.Polynomial,
        eta: float,
        epsilon: float
    )-> Tuple[np.ndarray, np.ndarray]:
        # check whether the current CBF is relative degree two.
        L_gh = lie_derivative(
            poly=cbf,
            vector_feild=self.g,
            variables=self.x,
            pow=1
        )
        expected_L_gh = np.array(
            [sym.Polynomial(0)]*self.n_u
        )
        check_polynomial_arrays_equal(
            p=L_gh, q=expected_L_gh, tol=1e-10
            )
        
        L_fh = lie_derivative(
            poly=cbf,
            vector_feild=self.f,
            variables=self.x,
            pow=1
        )
        L_f2h = lie_derivative(
            poly=cbf,
            vector_feild=self.f,
            variables=self.x,
            pow=2
        )
        L_fL_gh = lie_derivative(
            poly=L_fh,
            vector_feild=self.g,
            variables=self.x,
            pow=1
        )
        # construct Lambda and xi:
        # lambda:
        A = self.control_limits[0]
        lambda_mat = np.vstack((
            -L_fL_gh,
            A
        ))
        # xi:
        c = self.control_limits[1]
        alpha_symm_poly = elementary_symetric_polynomials(
            input=self.alphas
            )
        xi = np.hstack((
            alpha_symm_poly.dot(
                np.array([L_f2h, L_fh, cbf])
            ) - eta,
            c - epsilon
        ))
        return lambda_mat, xi

    def _construct_xy_set(
        self,
    ) -> Tuple[sym.Variables, np.ndarray, sym.Variables]:
        # the first item of the output should be the y_set
        # and the second item should be the y² vector
        # the third item is the xy_set for the sos program
        y_size = 1 + self.control_limits[0].shape[0]
        y = sym.MakeVectorContinuousVariable(y_size, "y")
        y_squared_poly = np.array([
            sym.Polynomial(sym.Monomial(y[i], 2))
            for i in range(y_size)
        ])
        y_set = sym.Variables(y)
        xy_vars = np.hstack((self.x, y))
        xy_set = sym.Variables(xy_vars)
        return (y_set, y_squared_poly, xy_set)

    def _construct_lagrangian_degrees(
        self,
        lagrangian_cbf_x_degree: int,
        lagrangian_cbf_y_degree: int,
        lagrangian_cbf_dot_x_degree: int,
        lagrangian_cbf_dot_y_degree: int,
        lagrangian_s_x_degree: int,
        lagrangian_s_y_degree: int,
        lagrangian_q_x_degree: int,
        lagrangian_q_y_degree: int,
        *,
        lagrangian_state_eq_x_degree: Optional[int] = None,
        lagrangian_state_eq_y_degree: Optional[int] = None
    ) -> OrderTwoCBFLagrangianDegrees:
        cbf_degree = Degree(
            x=lagrangian_cbf_x_degree,
            y=lagrangian_cbf_y_degree,
            c=0,
        )
        cbf_dot_degree = Degree(
            x=lagrangian_cbf_dot_x_degree,
            y=lagrangian_cbf_dot_y_degree,
            c=0,
        )
        s_degrees = [
            Degree(
                x=lagrangian_s_x_degree,
                y=lagrangian_s_y_degree,
                c=0,
            )
            for _ in range(self.n_u)
        ]
        q_degree = Degree(
            x=lagrangian_q_x_degree,
            y=lagrangian_q_y_degree,
            c=0,
        )
        if lagrangian_state_eq_x_degree is not None and \
           lagrangian_state_eq_y_degree is not None:
            state_eq_degree = Degree(
                x=lagrangian_state_eq_x_degree,
                y=lagrangian_state_eq_y_degree,
                c=0,
            )
        else:
            state_eq_degree = None
        return OrderTwoCBFLagrangianDegrees(
            cbf=cbf_degree,
            cbf_dot=cbf_dot_degree,
            s=s_degrees,
            q=q_degree,
            state_eq=state_eq_degree
        )

    def _add_sos_constraint(
        self,
        prog: solvers.MathematicalProgram,
        cbf: sym.Polynomial,
        cbf_dot: sym.Polynomial,
        lambda_mat: np.ndarray,
        xi: np.ndarray,
        y_squared: np.ndarray,
        lagrangians: OrderTwoCBFLagrangians,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> None:
        poly_one = sym.Polynomial(1)
        poly = -poly_one

        poly -= lagrangians.cbf * cbf

        poly -= lagrangians.cbf_dot * cbf_dot

        lambda_y_term = lambda_mat.T @ y_squared
        poly -= lagrangians.s.dot(lambda_y_term)

        xi_y_term = xi @ y_squared
        poly -= lagrangians.q * (xi_y_term + poly_one)

        if self.state_eq_constraint is not None and \
           lagrangians.state_eq is not None:
            poly -= lagrangians.state_eq * self.state_eq_constraint

        prog.AddSosConstraint(poly, type=sos_type)







