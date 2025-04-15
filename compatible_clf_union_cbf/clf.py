from dataclasses import dataclass
import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from typing import List, Optional, Tuple
from typing_extensions import Self
from compatible_clf_union_cbf.utils import (
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    lie_derivative,
    BackoffScale,
    check_polynomials_pass_origin,
    find_no_linear_term_variables,
    new_free_polynomial_pass_origin,
    new_sos_polynomial,
    solve_with_id,
)
from compatible_clf_union_cbf.inclusion import (
    BallInclusionLagrangianDegree,
    BallInclusionLagrangian,
    BallInclusion,
    PointsInclusionConstriants,
)


"""
This file includes the bilinear alternation of a CLF synthesis.
This is the problem formulation:
Given a set of points x1,...,xn, find a CLF V(x) such that:
1. V(x) is an SOS polynomial that goes through the origin, V(x=0)=0.
2. The sublevel set {x | V(x) <= 1} contains all the points x1,...,xn.
3. For all x in the sublevel set, there exists a u such that
        dVdx * f(x) + dVdx * g(x) * u <= -kappa * V.
4. The superlevel set {x | V(x) >= 1} should include a ball with radius r.

From the SOS perspective, we have the following SOS constraints:
let: p(x) = 1 - V(x)
1. The sublevel set contains the ball:
    -1 - s_1(x)*(r²-xᵀx) + s_2(x)*p(x) is SOS
    s_1(x) and s_2(x) are SOS polynomials.
2. The CLF constraint:
Let
Λ(x) = [LgV(x),
        Au]
ξ(x) = [-αV(x) - LfV(x),
        bu]
-1 - s0(x)ᵀΛ(x)ᵀy² - s1(x)(ξ(x)y²+1) - s2(x,y)(ρ-V(x)) is SOS
s2(x,y) is SOS, s0(x) and s1(x) are free polynomials.

"""


@dataclass
class ClfLagrangian:
    lambda_y: np.ndarray
    xi_y: sym.Polynomial
    rho_minus_V: sym.Polynomial
    state_eq_constraints: Optional[np.ndarray]

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float],
    ) -> Self:
        lambda_y_result = get_polynomial_result(result, self.lambda_y, coefficient_tol)
        xi_y_result = get_polynomial_result(result, self.xi_y, coefficient_tol)
        rho_minus_V_result = get_polynomial_result(
            result, self.rho_minus_V, coefficient_tol
        )
        state_eq_constraints_result = (
            get_polynomial_result(result, self.state_eq_constraints, coefficient_tol)
            if self.state_eq_constraints is not None
            else None
        )
        return ClfLagrangian(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            rho_minus_V=rho_minus_V_result,
            state_eq_constraints=state_eq_constraints_result,
        )


@dataclass
class ClfLagrangianDegree:
    lambda_y: List[Degree]
    xi_y: Degree
    rho_minus_V: Degree
    state_eq_constraints: Optional[List[Degree]]

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        y: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_lambda_y: Optional[List[np.ndarray]] = None,
        lagrangian_xi_y: Optional[np.ndarray] = None,
        lagrangian_rho_minus_V: Optional[np.ndarray] = None,
        lagrangian_state_eq_constraints: Optional[List[np.ndarray]] = None,
    ) -> ClfLagrangian:
        lambda_y_lagrangians = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y,
        )

        xi_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.xi_y,
            lagrangian=lagrangian_xi_y,
        )

        rho_minus_V_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.rho_minus_V,
            lagrangian=lagrangian_rho_minus_V,
        )

        if self.state_eq_constraints is not None:
            state_eq_constraints_lagrangians = to_lagrangian_impl(
                prog=prog,
                x=x,
                y=y,
                c=None,
                sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
                is_sos=True,
                degree=self.state_eq_constraints,
                lagrangian=lagrangian_state_eq_constraints,
            )
        else:
            state_eq_constraints_lagrangians = None

        return ClfLagrangian(
            lambda_y=lambda_y_lagrangians,
            xi_y=xi_y_lagrangian,
            rho_minus_V=rho_minus_V_lagrangian,
            state_eq_constraints=state_eq_constraints_lagrangians,
        )


class ClfSynthesis:
    def __init__(
        self,
        x: np.ndarray,
        sys_dyn_f: np.ndarray,
        sys_dyn_g: np.ndarray,
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray],
        state_eq_constraint: Optional[np.ndarray],
    ):
        """
        Noted that since we are synthesizing a CLF, the CLF would be changing during
        the running of some methods of this class. Therefore, we don't design the CLF
        as a class member. Instead, we use the CLF as a parameter of the methods.
        """
        self.x = x
        self.sys_dyn_f = sys_dyn_f
        self.sys_dyn_g = sys_dyn_g
        self.Au = Au
        self.bu = bu
        self.state_eq_constraint = state_eq_constraint

        if Au is not None:
            assert bu is not None
            assert Au.shape[1] == sys_dyn_g.shape[1]
        assert sys_dyn_f.shape[0] == self.x.size
        assert len(sys_dyn_g.shape) == 2
        assert sys_dyn_g.shape[0] == self.x.size

        self.x_set = sym.Variables(x)
        num_y = 1 + (Au.shape[0] if Au is not None else 0)
        self.y = sym.MakeVectorContinuousVariable(num_y, "y")
        self.y_poly = np.array(
            [sym.Polynomial(sym.Monomial(self.y[i], 1)) for i in range(num_y)]
        )
        self.y_squared_poly = np.array(
            [sym.Polynomial(sym.Monomial(self.y[i], 2)) for i in range(num_y)]
        )
        self.y_set = sym.Variables(self.y)
        self.xy_set = sym.Variables(np.concatenate([self.x, self.y]))

    def _calc_xi_lambda(
        self,
        kappaV: float,
        clf: sym.Polynomial,
    ) -> Tuple[np.ndarray, np.ndarray]:
        LfV = lie_derivative(
            poly=clf, vector_feild=self.sys_dyn_f, variables=self.x, pow=1
        )
        LgV = lie_derivative(
            poly=clf, vector_feild=self.sys_dyn_g, variables=self.x, pow=1
        )

        lambda_row = 1 + (self.Au.shape[0] if self.Au is not None else 0)
        lambda_col = self.sys_dyn_g.shape[1]
        lambda_mat = np.empty((lambda_row, lambda_col), dtype=object)
        lambda_mat[0, :] = LgV
        if self.Au is not None:
            lambda_mat[1:, :] = self.Au

        xi = np.empty(lambda_row, dtype=object)
        xi[0] = -kappaV * clf - LfV
        if self.bu is not None:
            xi[1:] = self.bu

        return lambda_mat, xi

    def _add_clf_constraint(
        self,
        prog: solvers.MathematicalProgram,
        lagrangian: ClfLagrangian,
        lambda_mat: np.ndarray,
        xi: np.ndarray,
        clf: sym.Polynomial,
        rho: float,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> sym.Polynomial:
        poly_one = sym.Polynomial(1)
        poly = -poly_one
        poly -= lagrangian.lambda_y.dot(lambda_mat.T @ self.y_squared_poly)
        poly -= lagrangian.xi_y * (xi.dot(self.y_squared_poly) + poly_one)
        poly -= lagrangian.rho_minus_V * (rho - clf)
        if self.state_eq_constraint is not None:
            poly -= lagrangian.state_eq_constraints.dot(self.state_eq_constraint)

        prog.AddSosConstraint(poly, sos_type)
        return poly

    def _add_ball_inclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        ball_radius: float,
        clf: sym.Polynomial,
        rho: float,
        ball_inclusion_lagrangian: BallInclusionLagrangian,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> sym.Polynomial:
        ball_inclusion = BallInclusion(radius=ball_radius, h=(rho - clf), x=self.x)
        poly = ball_inclusion.add_ball_inclusion_constraint(
            prog=prog,
            ball_inclusion_lagrangian=ball_inclusion_lagrangian,
            sos_type=sos_type,
        )
        return poly

    def _add_points_inclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        clf: sym.Polynomial,
        rho: float,
        included_points: np.ndarray,
        points_inclusion_weights: np.ndarray,
        anchor_points: Optional[np.ndarray],
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    ):
        points_inclusion = PointsInclusionConstriants(
            x=self.x,
            p=rho - clf,
            points_to_include=included_points,
            point_inclusion_weights=points_inclusion_weights,
            anchor_points=anchor_points,
            p_anchor_bounds=anchor_bounds,
        )
        points_inclusion.add_to_prog(prog=prog)
        points_inclusion.add_anchor_bound_to_prog(prog=prog)

    def _evalueate_clf_at_points(
        self, clf: sym.Polynomial, evaluate_at_points: np.ndarray
    ) -> np.ndarray:
        clf_at_points = clf.EvaluateIndeterminates(
            indeterminates=self.x, indeterminates_values=evaluate_at_points.T
        )
        return clf_at_points

    def _create_lagrangian_degrees(
        self,
        ball_inclusion_ball_x_degree: int,
        ball_inclusion_h_x_degree: int,
        clf_lagrangain_lambda_y_x_degree: List[int],
        clf_lagrangain_xi_y_x_degree: int,
        clf_lagrangain_rho_minus_V_x_degree: int,
        state_eq_constraints_x_degree: Optional[List[int]],
    ) -> Tuple[BallInclusionLagrangianDegree, ClfLagrangianDegree]:
        ball_inclusion_lagrangian_degree = BallInclusionLagrangianDegree(
            r_minus_xTx=Degree(x=ball_inclusion_ball_x_degree, y=0, c=0),
            h=Degree(x=ball_inclusion_h_x_degree, y=0, c=0),
        )
        clf_lagrangian_degree = ClfLagrangianDegree(
            lambda_y=[
                Degree(x=clf_lagrangain_lambda_y_x_degree[i], y=0, c=0)
                for i in range(len(clf_lagrangain_lambda_y_x_degree))
            ],
            xi_y=Degree(x=clf_lagrangain_xi_y_x_degree, y=0, c=0),
            rho_minus_V=Degree(x=clf_lagrangain_rho_minus_V_x_degree, y=2, c=0),
            state_eq_constraints=(
                [
                    Degree(x=state_eq_constraints_x_degree[i], y=2, c=0)
                    for i in range(len(state_eq_constraints_x_degree))
                ]
                if state_eq_constraints_x_degree is not None
                else None
            ),
        )
        return (ball_inclusion_lagrangian_degree, clf_lagrangian_degree)

    def search_lagrangian_given_clf(
        self,
        ball_inclusion_lagrangian_degree: BallInclusionLagrangianDegree,
        clf_lagrangian_degree: ClfLagrangianDegree,
        clf: sym.Polynomial,
        kappaV: float,
        rho: float,
        ball_radius: float,
        *,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coeff_tol: Optional[float] = None,
    ) -> Tuple[Optional[BallInclusionLagrangian], Optional[ClfLagrangian]]:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)

        (lambda_matrix, xi_vec) = self._calc_xi_lambda(kappaV=kappaV, clf=clf)
        # ball_inclusion_lagrangian = ball_inclusion_lagrangian_degree.to_lagrangians(
        #     prog=prog,
        #     x=self.x_set
        #     )
        clf_lagrangian = clf_lagrangian_degree.to_lagrangians(
            prog=prog, x=self.x_set, y=self.y_set
        )
        # add CLF constriants:
        self._add_clf_constraint(
            prog=prog,
            lagrangian=clf_lagrangian,
            lambda_mat=lambda_matrix,
            xi=xi_vec,
            clf=clf,
            rho=rho,
        )
        # # add ball inclusion constraints:
        # self._add_ball_inclusion_constraint(
        #     prog=prog,
        #     ball_radius=ball_radius,
        #     clf=clf,
        #     rho=rho,
        #     ball_inclusion_lagrangian=ball_inclusion_lagrangian
        # )

        result = solve_with_id(
            prog=prog, solver_id=solver_id, solver_options=solver_options
        )

        if result.is_success():
            # ball_inclusion_lagrangian_result = ball_inclusion_lagrangian.get_results(
            #     result=result,
            #     coefficient_tol=lagrangian_coeff_tol
            # )
            clf_lagrangian_result = clf_lagrangian.get_results(
                result=result, coefficient_tol=lagrangian_coeff_tol
            )
            return (
                # ball_inclusion_lagrangian_result,
                None,
                clf_lagrangian_result,
            )
        else:
            return (None, None)

    def seach_clf_given_lagrangians(
        self,
        ball_inclusion_lagrangian: BallInclusionLagrangian,
        clf_lagrangian: ClfLagrangian,
        V_x_degree: int,
        kappaV: float,
        rho: float,
        ball_radius: float,
        included_points: np.ndarray,
        points_inclusion_weights: np.ndarray,
        anchor_points: Optional[np.ndarray],
        acnhor_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        *,
        ball_inclusion_degrees: Optional[BallInclusionLagrangianDegree] = None,
        clf_lagrangian_degree: Optional[ClfLagrangianDegree] = None,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        backoff_rel_scale: Optional[float] = None,
        backoff_abs_scale: Optional[float] = None,
        clf_coeff_tol: Optional[float] = None,
    ) -> Optional[sym.Polynomial]:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)

        # if ball_inclusion_degrees is not None:
        #     (
        #         new_ball_inclusion_lagrangian
        #     ) = ball_inclusion_degrees.to_lagrangians(
        #         prog=prog,
        #         x=self.x_set,
        #         lagrangian_h=ball_inclusion_lagrangian.h,
        #     )
        # else:
        #     new_ball_inclusion_lagrangian = ball_inclusion_lagrangian
        if clf_lagrangian_degree is not None:
            (new_clf_lagrangian) = clf_lagrangian_degree.to_lagrangians(
                prog=prog,
                x=self.x_set,
                y=self.y_set,
                lagrangian_lambda_y=clf_lagrangian.lambda_y,
                lagrangian_xi_y=clf_lagrangian.xi_y,
                lagrangian_rho_minus_V=clf_lagrangian.rho_minus_V,
            )
        else:
            new_clf_lagrangian = clf_lagrangian

        V_unsolved, _ = new_sos_polynomial(
            prog=prog, x_set=self.x_set, degree=V_x_degree, zero_at_origin=True
        )

        lambda_matrix, xi_vec = self._calc_xi_lambda(kappaV=kappaV, clf=V_unsolved)
        self._add_clf_constraint(
            prog=prog,
            lagrangian=new_clf_lagrangian,
            lambda_mat=lambda_matrix,
            xi=xi_vec,
            clf=V_unsolved,
            rho=rho,
        )
        # self._add_ball_inclusion_constraint(
        #     prog=prog,
        #     ball_radius=ball_radius,
        #     clf=V_unsolved,
        #     rho=rho,
        #     ball_inclusion_lagrangian=new_ball_inclusion_lagrangian
        # )
        self._add_points_inclusion_constraint(
            prog=prog,
            clf=V_unsolved,
            rho=rho,
            included_points=included_points,
            points_inclusion_weights=points_inclusion_weights,
            anchor_points=anchor_points,
            anchor_bounds=acnhor_bounds,
        )

        result = solve_with_id(
            prog=prog,
            solver_id=solver_id,
            solver_options=solver_options,
            backoff_rel_scale=backoff_rel_scale,
            backoff_abs_scale=backoff_abs_scale,
        )

        if result.is_success():
            V_result = get_polynomial_result(
                result=result, p=V_unsolved, coefficient_tol=clf_coeff_tol
            )
            return V_result
        else:
            return None

    def bilinear_alternation(
        self,
        clf_init: sym.Polynomial,
        rho: float,
        kappaV: float,
        ball_radius: float,
        ball_inclusion_ball_x_degree: int,
        ball_inclusion_h_x_degree: int,
        clf_lagrangain_lambda_y_x_degree: List[int],
        clf_lagrangain_xi_y_x_degree: int,
        clf_lagrangain_rho_minus_V_x_degree: int,
        V_x_degree: int,
        state_eq_constraints_x_degree: Optional[List[int]],
        included_points: np.ndarray,
        points_inclusion_weights: np.ndarray,
        anchor_points: Optional[np.ndarray],
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        max_iter: int,
        *,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coeff_tol: Optional[float] = None,
        clf_coeff_tol: Optional[float] = None,
        backoff_scale: Optional[List[BackoffScale]] = None,
    ) -> Optional[sym.Polynomial]:
        (ball_inclusion_lagragian_degree, clf_lagrangian_degree) = (
            self._create_lagrangian_degrees(
                ball_inclusion_ball_x_degree=ball_inclusion_ball_x_degree,
                ball_inclusion_h_x_degree=ball_inclusion_h_x_degree,
                clf_lagrangain_lambda_y_x_degree=clf_lagrangain_lambda_y_x_degree,
                clf_lagrangain_xi_y_x_degree=clf_lagrangain_xi_y_x_degree,
                clf_lagrangain_rho_minus_V_x_degree=clf_lagrangain_rho_minus_V_x_degree,
                state_eq_constraints_x_degree=state_eq_constraints_x_degree,
            )
        )
        iter_num = 1
        clf = clf_init

        if backoff_scale is not None:
            assert len(backoff_scale) == max_iter
        while iter_num <= max_iter:
            (ball_inclusion_lagragian_result, clf_lagrangian_result) = (
                self.search_lagrangian_given_clf(
                    ball_inclusion_lagrangian_degree=ball_inclusion_lagragian_degree,
                    clf_lagrangian_degree=clf_lagrangian_degree,
                    clf=clf,
                    kappaV=kappaV,
                    rho=rho,
                    ball_radius=ball_radius,
                    solver_id=solver_id,
                    solver_options=solver_options,
                    lagrangian_coeff_tol=lagrangian_coeff_tol,
                )
            )

            # assert ball_inclusion_lagragian_result is not None
            assert clf_lagrangian_result is not None

            clf_updated = self.seach_clf_given_lagrangians(
                ball_inclusion_lagrangian=ball_inclusion_lagragian_result,
                clf_lagrangian=clf_lagrangian_result,
                V_x_degree=V_x_degree,
                kappaV=kappaV,
                rho=rho,
                ball_radius=ball_radius,
                included_points=included_points,
                points_inclusion_weights=points_inclusion_weights,
                anchor_points=anchor_points,
                acnhor_bounds=anchor_bounds,
                ball_inclusion_degrees=ball_inclusion_lagragian_degree,
                clf_lagrangian_degree=clf_lagrangian_degree,
                solver_id=solver_id,
                solver_options=solver_options,
                backoff_rel_scale=(
                    None if backoff_scale is None else backoff_scale[iter_num - 1].rel
                ),
                backoff_abs_scale=(
                    None if backoff_scale is None else backoff_scale[iter_num - 1].abs
                ),
                clf_coeff_tol=clf_coeff_tol,
            )

            assert clf_updated is not None

            clf_evaluation = self._evalueate_clf_at_points(
                clf=clf_updated, evaluate_at_points=included_points
            )
            print_values = rho - clf_evaluation
            print(f"iteration: {iter_num}")
            print(f"[rho - V](states_to_be_included): {print_values}")

            clf = clf_updated
            iter_num += 1

        return clf


class ClfConstraint:
    """
    Add the linear constraint dVdx * f(x) + dVdx * g(x) * u <= -kappa * V on u.
    """

    def __init__(
        self,
        V: sym.Polynomial,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        kappa: float,
    ):
        self.V = V
        dVdx = V.Jacobian(x)
        dVdx_times_f = dVdx.dot(f)
        dVdx_times_g = dVdx @ g
        self.rhs = -kappa * V - dVdx_times_f
        self.lhs_coeff = dVdx_times_g
        self.x = x

    def add_to_prog(
        self, prog: solvers.MathematicalProgram, x_val: np.ndarray, u: np.ndarray
    ) -> solvers.Binding[solvers.LinearConstraint]:
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        constraint = prog.AddLinearConstraint(lhs_coeff, -np.inf, rhs, u)
        return constraint


def find_candidate_regional_lyapunov(
    x: np.ndarray,
    dynamics: np.ndarray,
    V_degree: int,
    positivity_eps: float,
    d: int,
    kappa: float,
    state_eq_constraints: np.ndarray,
    positivity_ceq_lagrangian_degrees: List[int],
    derivative_ceq_lagrangian_degrees: List[int],
    state_ineq_constraints: np.ndarray,
    positivity_cin_lagrangian_degrees: List[int],
    derivative_cin_lagrangian_degrees: List[int],
) -> Tuple[solvers.MathematicalProgram, sym.Polynomial]:
    """
    Constructs a program to find Lyapunov candidate V for a closed-loop system,
    that satisfy the following constraints within a region cin(x) <= 0

    Find V(x), p1(x), p2(x), q1(1), q2(x)
    s.t V - ε1*(xᵀx)ᵈ + p1(x) * cin(x) - p2(x) * ceq(x) is sos  (1)
       -Vdot - κ * V + q1(x) * cin(x) - q2(x) * ceq(x) is sos  (2)
       p1(x) is sos, q1(x) is sos.

    Namely SOS can verify that on the semialgebraic set
    {x | cin(x) <= 0, ceq(x) = 0}, we have V(x) >= 0 and Vdot <= -κ * V

    Args:
      dynamics: An array of polynomials of x. The closed-loop dynamics.
      state_eq_constraints: An array of polynomials of x. The left hand side of
      ceq(x) = 0
      state_ineq_constraints: An array of polynomials of x. The left hand side
      of cin(x) <= 0
    """
    # We assume that the goal state is x = 0.
    check_polynomials_pass_origin(dynamics)
    prog = solvers.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    # We know that ceq(0) = 0 (because the goal state 0 should satisfy the
    # equality constraint). Combining this with
    # V - ε1*(xᵀx)ᵈ + p1(x) * cin(x) - p2(x) * ceq(x) is sos     (1)
    # we know that the left hand side of (1) is 0 at the x=0. The left hand side
    # is p1(0) * cin(0) at x=0.
    # We know that cin(0) < 0 (assume that the goal state 0 is in the strict interior
    # of the region). p1(x) is a sos, hence p1(0) = 0. A sos polynomial with
    # constant term equal to 0 also means that all its linear terms should be 0.
    # Hence p1(x) doesn't have a linear or constant terms. Hence the only linear
    # term from the left hand side of (1) can come from V(x) and and ceq(x), and
    # the linear terms have to cancel out.
    no_linear_term_variables = find_no_linear_term_variables(
        x_set, state_eq_constraints
    )
    V = new_free_polynomial_pass_origin(
        prog, x_set, V_degree, "V", no_linear_term_variables
    )
    # Add the constraint V - ε1*(xᵀx)ᵈ + p1(x) * cin(x) - p2(x) * ceq(x) is sos
    positivity_sos_condition = V
    if positivity_eps > 0:
        positivity_sos_condition -= positivity_eps * sym.Polynomial(
            sym.pow(x.dot(x), d)
        )
    p1 = np.array(
        [
            prog.NewSosPolynomial(x_set, degree)[0]
            for degree in positivity_cin_lagrangian_degrees
        ]
    )
    positivity_sos_condition += p1.dot(state_ineq_constraints)
    p2 = np.array(
        [
            prog.NewFreePolynomial(x_set, degree)
            for degree in positivity_ceq_lagrangian_degrees
        ]
    )
    positivity_sos_condition -= p2.dot(state_eq_constraints)
    prog.AddSosConstraint(positivity_sos_condition)

    # Impose the constraint -Vdot - κ * V + q1(x) * cin(x) - q2(x) * ceq(x) is sos
    Vdot = V.Jacobian(x).dot(dynamics)
    derivative_sos_condition = -Vdot - kappa * V
    q1 = np.array(
        [
            prog.NewSosPolynomial(x_set, degree)[0]
            for degree in derivative_cin_lagrangian_degrees
        ]
    )
    derivative_sos_condition += q1.dot(state_ineq_constraints)
    q2 = np.array(
        [
            prog.NewFreePolynomial(x_set, degree)
            for degree in derivative_ceq_lagrangian_degrees
        ]
    )
    derivative_sos_condition -= q2.dot(state_eq_constraints)
    prog.AddSosConstraint(derivative_sos_condition)
    return prog, V
