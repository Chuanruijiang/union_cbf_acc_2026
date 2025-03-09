from dataclasses import dataclass
import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from typing import List, Optional, Tuple, Union
from typing_extensions import Self
from compatible_clf_union_cbf.utils import (
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    lie_derivative,
    solve_with_id,
    new_sos_polynomial,
)
from compatible_clf_union_cbf.inclusion import(
    BallInclusionLagrangianDegree,
    BallInclusionLagrangian,
    BallInclusion,
    PointsInclusionConstriants
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
    
    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficitent_tol: Optional[float]
    ) -> Self:
        lambda_y_result = get_polynomial_result(
            result, self.lambda_y, coefficitent_tol
            )
        xi_y_result = get_polynomial_result(
            result, self.xi_y, coefficitent_tol
            )
        rho_minus_V_result = get_polynomial_result(
            result, self.rho_minus_V, coefficitent_tol
            )
        return ClfLagrangian(
            lambda_y=lambda_y_result,
            xi_y=xi_y_result,
            rho_minus_V=rho_minus_V_result
            )


@dataclass
class ClfLagrangianDegree:
    lambda_y: List[Degree]
    xi_y: Degree
    rho_minus_V: Degree

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
    ) -> ClfLagrangian:
        lambda_y_lagrangians = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.lambda_y,
            lagrangian=lagrangian_lambda_y
        )

        xi_y_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=False,
            degree=self.xi_y,
            lagrangian=lagrangian_xi_y
        )

        rho_minus_V_lagrangian = to_lagrangian_impl(
            prog=prog,
            x=x,
            y=y,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.rho_minus_V,
            lagrangian=lagrangian_rho_minus_V
        )

        return ClfLagrangian(
            lambda_y=lambda_y_lagrangians,
            xi_y=xi_y_lagrangian,
            rho_minus_V=rho_minus_V_lagrangian
            )


class ClfSynthesis:
    def __init__(
        self,
        x: np.ndarray,
        sys_dyn_f: np.ndarray,
        sys_dyn_g: np.ndarray,
        Au: Optional[np.ndarray],
        bu: Optional[np.ndarray]
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

        if Au is not None:
            assert bu is not None
            assert Au.shape[0] == sys_dyn_g.shape[0]
        assert sys_dyn_f.shape[0] == self.x.size
        assert len(sys_dyn_g.shape) == 2
        assert sys_dyn_g.shape[0] == self.x.size

        self.x_set = sym.Variables(x)
        num_y = 1+(Au.shape[0] if Au is not None else 0)
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
            poly=clf,
            vector_feild=self.sys_dyn_f,
            variables=self.x,
            pow=1
        )
        LgV = lie_derivative(
            poly=clf,
            vector_feild=self.sys_dyn_g,
            variables=self.x,
            pow=1
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
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> sym.Polynomial:
        poly_one = sym.Polynomial(1)
        poly = -poly_one
        poly -= lagrangian.lambda_y.dot(lambda_mat.T @ self.y_squared_poly)
        poly -= lagrangian.xi_y * (xi.dot(self.y_squared_poly) + poly_one)
        poly -= lagrangian.rho_minus_V * (rho - clf)

        prog.AddSosConstraint(poly, sos_type)
        return poly

    def _add_ball_inclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        ball_radius: float,
        clf: sym.Polynomial,
        rho: float,
        ball_inclusion_lagrangian: BallInclusionLagrangian,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> sym.Polynomial:
        ball_inclusion = BallInclusion(
            radius=ball_radius,
            h=(rho - clf),
            x=self.x
        )
        poly = ball_inclusion.add_ball_inclusion_constraint(
            prog=prog,
            ball_inclusion_lagrangian=ball_inclusion_lagrangian,
            sos_type=sos_type
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
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]]
    ): 
        points_inclusion = PointsInclusionConstriants(
            x=self.x,
            p=rho-clf,
            points_to_include=included_points,
            point_inclusion_weights=points_inclusion_weights,
            anchor_points=anchor_points,
            p_anchor_bounds=anchor_bounds
        )
        points_inclusion.add_to_prog(prog=prog)
        points_inclusion.add_anchor_bound_to_prog(prog=prog)

    def _evalueate_clf_at_points(
        self,
        clf: sym.Polynomial,
        evaluate_at_points: np.ndarray
    ) -> np.ndarray:
        clf_at_points = clf.EvaluateIndeterminates(
            indeterminates=self.x,
            indeterminates_values=evaluate_at_points.T
        )
        return clf_at_points
        
    def _create_lagrangian_degrees(
        self,
        ball_inclusion_ball_x_degree: int,
        ball_inclusion_h_x_degree: int,
        clf_lagrangain_lambda_y_x_degree: List[int],
        clf_lagrangain_xi_y_x_degree: int,
        clf_lagrangain_rho_minus_V_x_degree: int
    ) -> Tuple[
        BallInclusionLagrangianDegree,
        ClfLagrangianDegree
        ]:
        ball_inclusion_lagrangian_degree = BallInclusionLagrangianDegree(
            r_minus_xTx=Degree(x=ball_inclusion_ball_x_degree, y=0, c=0),
            h=Degree(x=ball_inclusion_h_x_degree, y=0, c=0)
        )
        clf_lagrangian_degree = ClfLagrangianDegree(
            lambda_y=[
                Degree(x=clf_lagrangain_lambda_y_x_degree[i], y=0, c=0)
                for i in range(len(clf_lagrangain_lambda_y_x_degree))
            ],
            xi_y=Degree(x=clf_lagrangain_xi_y_x_degree, y=0, c=0),
            rho_minus_V=Degree(x=clf_lagrangain_rho_minus_V_x_degree, y=2, c=0)
        )
        return (
            ball_inclusion_lagrangian_degree,
            clf_lagrangian_degree
        )

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
    )-> Tuple[
        Optional[BallInclusionLagrangian],
        Optional[ClfLagrangian]
    ]:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)

        (lambda_matrix, xi_vec
        ) = self._calc_xi_lambda(
            kappaV=kappaV,
            clf=clf
            )
        ball_inclusion_lagrangian = ball_inclusion_lagrangian_degree.to_lagrangians(
            prog=prog,
            x=self.x_set
            )
        clf_lagrangian = clf_lagrangian_degree.to_lagrangians(
            prog=prog,
            x=self.x_set,
            y=self.y_set
            )
        # add CLF constriants:
        self._add_clf_constraint(
            prog=prog,
            lagrangian=clf_lagrangian,
            lambda_mat=lambda_matrix,
            xi=xi_vec,
            clf=clf,
            rho=rho
        )
        # add ball inclusion constraints:
        self._add_ball_inclusion_constraint(
            prog=prog,
            ball_radius=ball_radius,
            clf=clf,
            rho=rho,
            ball_inclusion_lagrangian=ball_inclusion_lagrangian
        )

        result = solve_with_id(
            prog=prog,
            solver_id=solver_id,
            solver_options=solver_options
        )

        if result.is_success():
            ball_inclusion_lagrangian_result = ball_inclusion_lagrangian.get_results(
                result=result,
                coefficitent_tol=lagrangian_coeff_tol
            )
            clf_lagrangian_result = clf_lagrangian.get_results(
                result=result,
                coefficitent_tol=lagrangian_coeff_tol
            )
            return (
                ball_inclusion_lagrangian_result, 
                clf_lagrangian_result
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
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        clf_coeff_tol: Optional[float] = None,
    ) -> Optional[sym.Polynomial]:
        prog = solvers.MathematicalProgram()
        prog.AddIndeterminates(self.xy_set)
        
        V_unsolved,_ = new_sos_polynomial(
            prog=prog,
            x_set=self.x_set,
            degree=V_x_degree,
            zero_at_origin=True
        )

        lambda_matrix, xi_vec = self._calc_xi_lambda(
            kappaV=kappaV,
            clf=V_unsolved
        )
        self._add_clf_constraint(
            prog=prog,
            lagrangian=clf_lagrangian,
            lambda_mat=lambda_matrix,
            xi=xi_vec,
            clf=V_unsolved,
            rho=rho
        )
        self._add_ball_inclusion_constraint(
            prog=prog,
            ball_radius=ball_radius,
            clf=V_unsolved,
            rho=rho,
            ball_inclusion_lagrangian=ball_inclusion_lagrangian
        )
        self._add_points_inclusion_constraint(
            prog=prog,
            clf=V_unsolved,
            rho=rho,
            included_points=included_points,
            points_inclusion_weights=points_inclusion_weights,
            anchor_points=anchor_points,
            anchor_bounds=acnhor_bounds
        )

        result = solve_with_id(
            prog=prog,
            solver_id=solver_id,
            solver_options=solver_options
        )

        if result.is_success():
            V_result = get_polynomial_result(
                result=result,
                p=V_unsolved,
                coefficient_tol=clf_coeff_tol
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
        included_points: np.ndarray,
        points_inclusion_weights: np.ndarray,
        anchor_points: Optional[np.ndarray],
        anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        max_iter: int,
        *,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
        lagrangian_coeff_tol: Optional[float] = None,
        clf_coeff_tol: Optional[float] = None
    ) -> Optional[sym.Polynomial]:
        (
            ball_inclusion_lagragian_degree,
            clf_lagrangian_degree
        ) = self._create_lagrangian_degrees(
            ball_inclusion_ball_x_degree=ball_inclusion_ball_x_degree,
            ball_inclusion_h_x_degree=ball_inclusion_h_x_degree,
            clf_lagrangain_lambda_y_x_degree=clf_lagrangain_lambda_y_x_degree,
            clf_lagrangain_xi_y_x_degree=clf_lagrangain_xi_y_x_degree,
            clf_lagrangain_rho_minus_V_x_degree=clf_lagrangain_rho_minus_V_x_degree
        )
        iter_num = 1
        clf = clf_init

        while(iter_num <= max_iter):
            (
                ball_inclusion_lagragian_result,
                clf_lagrangian_result
            ) = self.search_lagrangian_given_clf(
                ball_inclusion_lagrangian_degree=ball_inclusion_lagragian_degree,
                clf_lagrangian_degree=clf_lagrangian_degree,
                clf=clf,
                kappaV=kappaV,
                rho=rho,
                ball_radius=ball_radius,
                solver_id=solver_id,
                solver_options=solver_options,
                lagrangian_coeff_tol=lagrangian_coeff_tol
            )
            
            assert ball_inclusion_lagragian_result is not None
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
                solver_id=solver_id,
                solver_options=solver_options,
                clf_coeff_tol=clf_coeff_tol
            )

            assert clf_updated is not None

            clf_eavaluation = self._evalueate_clf_at_points(
                clf=clf_updated,
                evaluate_at_points=included_points
            )
            print_values = rho - clf_eavaluation
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


