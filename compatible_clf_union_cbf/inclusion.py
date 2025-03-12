from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union
from typing_extensions import Self
import pydrake.solvers as solvers
import pydrake.symbolic as sym
from compatible_clf_union_cbf.utils import (
    Degree,
    to_lagrangian_impl,
    get_polynomial_result,
    solve_with_id,
)

"""
The following contains the verification of the ball inclusion in the interior
of a 0-super-level set of a polynomial function. We wanted to verify that:
for all x in the ball B_r(0) = {xᵀx <= r}, h(x) > 0.
This is equivalent to verifying that the set:
{ x | xᵀx <= r, h(x) <= 0 }
is empty. We can use the S-procedure to verify this. We can write the above

-1 - s_1(x) * (r - xᵀx) + s_2(x)h(x) is SOS 
where s_1(x) and s_2(x) are SOS polynomials.
"""
@dataclass
class BallInclusionLagrangian:
    r_minus_xTx: sym.Polynomial
    h: sym.Polynomial

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float]
    )-> Self:
        r_minus_xTx_result = get_polynomial_result(
            result, self.r_minus_xTx, coefficient_tol
        )
        h_result = get_polynomial_result(
            result, self.h, coefficient_tol
        )
        return BallInclusionLagrangian(
            r_minus_xTx=r_minus_xTx_result,
            h=h_result
            )


@dataclass
class BallInclusionLagrangianDegree:
    r_minus_xTx: Degree
    h: Degree

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        lagrangian_r_minus_xTx: Optional[sym.Polynomial] = None,
        lagrangian_h: Optional[sym.Polynomial] = None,
    ) -> BallInclusionLagrangian:
        assert self.r_minus_xTx.y == 0
        assert self.h.y == 0
        r_minus_xTx = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.r_minus_xTx,
            lagrangian=lagrangian_r_minus_xTx
        )
        h = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.h,
            lagrangian=lagrangian_h
        )
        return BallInclusionLagrangian(
            r_minus_xTx=r_minus_xTx,
            h=h
        )


class BallInclusion:
    def __init__(
        self,
        radius: float,
        h: sym.Polynomial,
        x: np.ndarray
    ):
        self.radius = radius
        self.h = h
        self.x = x

    def add_ball_inclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        ball_inclusion_lagrangian: BallInclusionLagrangian,
        sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> sym.Polynomial:
        x = self.x
        r = self.radius
        h = self.h
        s1 = ball_inclusion_lagrangian.r_minus_xTx
        s2 = ball_inclusion_lagrangian.h
        poly = -1 - s1 * (r - x.dot(x)) + s2 * h
        prog.AddSosConstraint(poly, sos_type)
        return poly
    
    def verify_ball_inclusion(
        self,
        ball_x_degree: int,
        h_x_degree: int,
        sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos,
    ) -> bool:
        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.x)
        prog.AddIndeterminates(x_set)

        ball_inclusion_degree = BallInclusionLagrangianDegree(
            r_minus_xTx=Degree(x = ball_x_degree, y= 0, c=0),
            h=Degree(x = h_x_degree, y= 0, c=0)
        )
        ball_inclusion_lagrangian = ball_inclusion_degree.to_lagrangians(
            prog, x_set, sos_type=sos_type
        )

        self.add_ball_inclusion_constraint(
            prog, ball_inclusion_lagrangian, sos_type=sos_type
        )
        result = solve_with_id(prog)
        return result.is_success()


"""
The following module creates SOS constraints for letting a 0-super-level set of a
polynomial to include some specified points. Hence, given a set of sepcified points: 
x_1,...,x_N. We would like to find the coefficients of the polynomial p(x) such that
the following cost is zero:
cost = sumᵢ₌₁ᴺ wᵢ*Relu(-p(xᵢ))
where wᵢ is the weight for each point xᵢ.

In order to get the SOS constraints, we can define a new group of decision variables
p_relu to denote the Relu(-p(xᵢ)). We can then write the following SOS constraint:
cost = sumᵢ₌₁ᴺ p_reluᵢ(x)
constriant1: -p(x_i) <= p_relu_i(x) for all i
constriant2: 0 <= p_relu_i(x) <= inf for all i

For constraint1, since each x_i is known and p(x) is a polynomial, then each p(x_i)
can be written as: p(x_i) = aᵢᵀθₚ + bᵢ where aᵢ is a contant vector, bᵢ is a
constant scalar, and θₚ are the coefficients of p(x).
Noted that we have N number of x_i, then we have:
[p(x₁), p(x₂), ..., p(x_N)]ᵀ = A * θₚ + b 
where A is a matrix of with each row being aᵢᵀ and b is a vector of bᵢ. 
Hence, letting p_relu = [p_relu₁, p_relu₂, ..., p_relu_N]ᵀ, we can write the
constraint1 as:
-A * θₚ - b <= p_relu
since p_relu and θₚ are both decision variables, we can put them
together as:
-b < = [A, I] * [θₚᵀ, p_reluᵀ]ᵀ

According to the constraint2, each p_relu_i has inf as the upper bound. We can put 
contraint2 and constriant1 together as:
-b <= [A, I] * [θₚᵀ, p_reluᵀ]ᵀ <= inf
where inf here is a vector with the same size as p_relu and vector b.

To summarize, the conditions for the polynomial p(x) to include the points
x_1,...,x_N are:
1. The cost function is zero
    sumᵢ₌₁ᴺ wᵢ*p_relu_i = 0
2. The constraints:
    b <= [A, I] * [θₚᵀ, p_reluᵀ]ᵀ <= inf

We may also want to let the p(x) not exceed a certain value at some points. We can
specify the upper and lower bound of p(x) at these points. We call these points 
anchor points. Assume there are M anchor points, for each anchor point x_j, we want
the polynomial p(x) to be in the range [lower_bound, upper_bound]. We can write the
following constraints:
lower_bound <= p(x_j) <= upper_bound
"""
@dataclass
class PointsInclusionConstriants:
    # the indeterminates of the polynomial:
    x: np.ndarray
    # the polynomial that includes theses points in the 0-super-level set
    p: sym.Polynomial
    # the points to be included, this array should only have 2 axis
    # axis 0: is the number of points
    # axis 1: is the number of dimensions of the points
    points_to_include: np.ndarray
    # the lower and upper bound of p(x) at the anchor points. This term should be a
    # tuple with 2 arrays. The first array is the lower bounds and the second is the
    # upper bounds. Each array is a 1D array with the same size as the number of 
    # anchor points.
    point_inclusion_weights: np.ndarray
    # the set of anchor points. This is an array with 2 axis
    # axis 0: is the number of anchor points
    # axis 1: is the number of dimensions of the anchor points
    anchor_points: Optional[np.ndarray]
    # these are the weights in the loss function for the points inclusion. The size
    # of this array should be the same as the number of points to be included.
    p_anchor_bounds: Optional[Tuple[np.ndarray, np.ndarray]]
    # if the following term is not zero, then we want the 0-super-level set of
    # p(x) + bias to include the points
    bias: float = 0
    # if the following term is not zero, then we want all the points to be included
    # in the interior of the 0-super-level set of p(x), i.e. p(x) > 0
    # thie means that we want the 0-super-level set of p(x) - margin to include
    # these points
    margin: float = 0
    # Hence, if the bias and margin are both non-zero, the then p_relu will be:
    # -(p(x) + bias - margin) <= p_relu

    def add_to_prog(
        self,
        prog: solvers.MathematicalProgram
    )-> Tuple[
        solvers.Binding[solvers.LinearCost],
        np.ndarray
    ]: 
        assert len(self.points_to_include.shape) == 2
        num_included_points = self.points_to_include.shape[0]
        assert self.point_inclusion_weights.shape[0] == num_included_points
        assert self.margin >= 0

        p_relu = prog.NewContinuousVariables(num_included_points, "p_relu")
        prog.AddBoundingBoxConstraint(0, np.inf, p_relu)
        (Ap, theta_p, bp) = self.p.EvaluateWithAffineCoefficients(
            indeterminates=self.x,
            indeterminates_values=self.points_to_include.T
        )

        prog.AddLinearConstraint(
            A=np.concatenate([Ap, np.eye(num_included_points)], axis=1),
            lb=-bp - self.bias + self.margin,
            ub=np.full_like(bp, np.inf),
            vars=np.concatenate([theta_p, p_relu])
        )

        cost_coeff = self.point_inclusion_weights
        cost_vars = p_relu
        cost = prog.AddLinearCost(cost_coeff, 0.0, cost_vars)

        return cost, p_relu
    
    def add_anchor_bound_to_prog(
        self,
        prog: solvers.MathematicalProgram,
    )-> Optional[solvers.Binding[solvers.LinearConstraint]]:
        if self.anchor_points is not None and self.p_anchor_bounds is not None:
            assert len(self.anchor_points.shape) == 2
            num_anchor_points = self.anchor_points.shape[0]
            assert len(self.p_anchor_bounds) == 2
            assert self.p_anchor_bounds[0].shape[0] == num_anchor_points
            assert self.p_anchor_bounds[1].shape[0] == num_anchor_points

            (Ap, theta_p, bp) = self.p.EvaluateWithAffineCoefficients(
                indeterminates=self.x,
                indeterminates_values=self.anchor_points.T
            )

            constraint = prog.AddLinearConstraint(
                A=Ap,
                lb=self.p_anchor_bounds[0] - bp,
                ub=self.p_anchor_bounds[1] - bp,
                vars=theta_p
            )
            return constraint
        else:
            return None


"""
The following module creates SOS constraints for excluding unsafe region
presented by the intersection of 0-super-level sets of a collection of
polynomial functions. Given a set of polynomial functions p₁(x), ..., pₙ(x)
the unsafe region is the region: {x| p₁(x) => 0, ..., pₙ(x) => 0}.
Now, assume we have a cbf h(x), we hope that: 
for all x in the unsafe region, h(x) < 0.
This is equivalent to verifying that the set:
{x | p₁(x) => 0, ..., pₙ(x) => 0, h(x) >= 0}
is empty. 
Using P-satz, we have the following SOS constraint:
-1 - s₁(x) * p₁(x) - ... - sₙ(x) * pₙ(x) - s_{n+1}(x) * h(x) is SOS
where s₁(x), ..., s_{n+1}(x) are SOS polynomials.
"""
@dataclass
class UnsafeRegionExclusionLagrangians:
    # The array of polynomials presenting lagrangains for p(x)s above
    unsafe_polys: np.ndarray
    # The lagrangian for the cbf function
    h: sym.Polynomial

    def get_results(
        self,
        result: solvers.MathematicalProgramResult,
        coefficient_tol: Optional[float]
    )-> Self:
        unsafe_polys_results = get_polynomial_result(
            result, self.unsafe_polys, coefficient_tol
        )
        h_result = get_polynomial_result(
            result, self.h, coefficient_tol
        )
        return UnsafeRegionExclusionLagrangians(
            unsafe_polys=unsafe_polys_results,
            h=h_result
        )


@dataclass
class UnsafeRegionExclusionLagrangianDegrees:
    unsafe_polys: List[Degree]
    h: Degree

    def to_lagrangians(
        self,
        prog: solvers.MathematicalProgram,
        x: sym.Variables,
        *,
        sos_type=solvers.MathematicalProgram.NonnegativePolynomial.kSos,
        unsafe_poly_lagrangians: Optional[np.ndarray] = None,
        h_lagrangian: Optional[sym.Polynomial] = None
    ) -> UnsafeRegionExclusionLagrangians:
        lagrangians_unsafe_polys = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.unsafe_polys,
            lagrangian=unsafe_poly_lagrangians
        )
        lagrangians_h = to_lagrangian_impl(
            prog,
            x,
            y=None,
            c=None,
            sos_type=sos_type,
            is_sos=True,
            degree=self.h,
            lagrangian=h_lagrangian
        )
        return UnsafeRegionExclusionLagrangians(
            unsafe_polys=lagrangians_unsafe_polys,
            h=lagrangians_h
        )


class UnsafeExclusion:
    def __init__(
        self,
        unsafe_polys: np.ndarray,
        h: sym.Polynomial,
        x: np.ndarray
    ):
        self.unsafe_polys = unsafe_polys
        self.h = h
        self.x = x

    def add_unsafe_exclusion_constraint(
        self,
        prog: solvers.MathematicalProgram,
        unsafe_exclusion_lagrangians: UnsafeRegionExclusionLagrangians,
        sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos
    )-> sym.Polynomial:
        polys = self.unsafe_polys
        h = self.h
        poly = -1
        for i in range(polys.shape[0]):
            poly -= unsafe_exclusion_lagrangians.unsafe_polys[i] * polys[i]
        poly -= unsafe_exclusion_lagrangians.h * h
        prog.AddSosConstraint(poly, sos_type)
        return poly

    def verify_unsafe_exclusion(
        self,
        unsafe_poly_x_degrees: List[int],
        h_x_degree: int,
        sos_type = solvers.MathematicalProgram.NonnegativePolynomial.kSos
    ) -> bool:
        assert len(unsafe_poly_x_degrees) == self.unsafe_polys.shape[0]

        prog = solvers.MathematicalProgram()
        x_set = sym.Variables(self.x)
        prog.AddIndeterminates(x_set)

        unsafe_exclusion_degree = UnsafeRegionExclusionLagrangianDegrees(
            unsafe_polys=[
                Degree(x = unsafe_poly_x_degrees[i], y= 0, c=0)
                for i in range(self.unsafe_polys.shape[0])
            ],
            h=Degree(x = h_x_degree, y= 0, c=0)
        )
        unsafe_exclusion_lagrangians = unsafe_exclusion_degree.to_lagrangians(
            prog, x_set, sos_type=sos_type
        )
        self.add_unsafe_exclusion_constraint(
            prog, unsafe_exclusion_lagrangians, sos_type=sos_type
        )

        result = solve_with_id(prog)
        return result.is_success()