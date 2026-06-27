from typing import List, Optional, Tuple, Union

import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.framework as drake_sys_frame

from union_cbf_base.utils import (
    elementary_symetric_polynomials,
    lie_derivative,
    lower_lie_derivatives,
    solve_with_id,
)


class CbfConstraint:
    """
    Add the linear constraint 
    for an HOCBF:
    If the realtive degree of h(x) is r, then
    the left hand side is:
    -L_{g}L_{f}^{r-1}h(x) u
    the right hand side is:
    elementary symetric polynomial vector of alphas
    times the vector 
    [L_{f}^{r-1}h(x), L_{f}^{r-2}h(x), ..., h(x)]^T
    At any given state value x_val, the constraint is:
    lhs(x_val) * u <= rhs(x_val)
    which is linear in u.
    (   In other words, this is to add one row of 
        Lambda(x)u <= xi(x) )
    """
    def __init__(
        self,
        h: Union[sym.Polynomial, sym.Expression],
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        alpha: list[float],
        relative_degree: int,
    ):
        assert len(alpha) == relative_degree

        # Compute the HOCBF constraint:
        # -L_g L_f^(r-1) h(x) u <= sum_i e_i(alpha) L_f^(r-i) h(x)
        # where r is the relative degree and e_i are elementary symmetric
        # polynomials of the alpha coefficients.
        Lf_r_1 = lie_derivative(
            poly=h,
            vector_field=f,
            variables=x,
            pow=relative_degree - 1,
        )
        Lg_Lf_r_1 = lie_derivative(
            poly=Lf_r_1,
            vector_field=g,
            variables=x,
            pow=1,
        )
        assert isinstance(Lg_Lf_r_1, np.ndarray)
        assert Lg_Lf_r_1.shape[0] == g.shape[1]
        self.lhs_u_coeff = -Lg_Lf_r_1

        alpha_vector = elementary_symetric_polynomials(alpha)
        lie_derivative_vector = np.empty(
            shape=(relative_degree + 1,),
            dtype=object,
        )
        lie_derivative_vector[-1] = h
        for j in range(relative_degree, 0, -1):
            lie_derivative_vector[j - 1] = lie_derivative(
                poly=lie_derivative_vector[j],
                vector_field=f,
                variables=x,
                pow=1,
            )
        self.rhs = np.dot(lie_derivative_vector, alpha_vector)
        self.lhs_coeff = self.lhs_u_coeff
        self.x = x
        self.h = h

    def add_to_prog(
        self,
        prog: solvers.MathematicalProgram,
        x_val: np.ndarray,
        u: np.ndarray
    ) -> solvers.Binding[solvers.LinearConstraint]:
        """
        Add the linear HOCBF constraint evaluated at the current state.
        """
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        for i in range(lhs_coeff.shape[0]):
            if abs(lhs_coeff[i]) < 1e-8:
                lhs_coeff[i] = 0.0
        if abs(rhs) < 1e-8:
            rhs = 0.0
        constraint = prog.AddLinearConstraint(
            a = lhs_coeff,
            lb = -np.inf,
            ub = rhs,
            vars = u
            )
        return constraint

    def add_to_prog_as_cost(
        self,
        prog: solvers.MathematicalProgram,
        x_val: np.ndarray,
        u: np.ndarray
    ):
        """
        Add the linearized HOCBF residual as a cost so the controller can
        probe feasibility margins under the current admissible control set.
        """
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        cost = lhs_coeff.dot(u) - rhs
        prog.AddLinearCost(e=cost)


class SwitchingCBFController(drake_sys_frame.LeafSystem):
    """
    If a system has trig-poly affine dynamics, then this controller
    can be applied in the extended state space. For polynomial-affine
    dynamics, pass the original state vector directly.
    """
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        static_cbfs: Optional[np.ndarray],
        switching_cbfs: Optional[np.ndarray],
        relative_degree: List[int],
        alpha: List[List[float]],
        control_limits: Tuple[np.ndarray, np.ndarray],
        switching_policy_id: int, # 1 or 2
        switching_policy_param: Tuple[float, float],
        initial_cbf_index: int=0,
        solver_id: Optional[solvers.SolverId] = None,
        solver_options: Optional[solvers.SolverOptions] = None,
    ):
        super().__init__()
        self.x = x
        self.f = f
        self.g = g
        self.static_cbfs = static_cbfs
        self.switching_cbfs = switching_cbfs
        self.control_limits = control_limits
        assert not (static_cbfs is None and switching_cbfs is None)

        num_static_cbf_constraints = (
            0 if static_cbfs is None else static_cbfs.shape[0]
        )
        num_switching_cbf_constraints = (
            0 if switching_cbfs is None else 1
        )
        num_cbf_constraints = (
            num_static_cbf_constraints + num_switching_cbf_constraints
        )
        assert len(alpha) == num_cbf_constraints
        assert len(relative_degree) == num_cbf_constraints
        assert 0 <= initial_cbf_index < switching_cbfs.shape[0]
        self.relative_degree = relative_degree
        self.alpha = alpha

        assert switching_policy_id == 1 or switching_policy_id == 2
        self.switching_policy_id = switching_policy_id
        assert isinstance(switching_policy_param, tuple)
        assert len(switching_policy_param) == 2
        self.switching_policy_param = switching_policy_param
        self.solver_id = solver_id
        self.solver_options = solver_options

        self.nu = g.shape[1]
        self.nx = x.shape[0]
        self.choose_switching_cbf_index = initial_cbf_index

        self.DeclareVectorInputPort("state", x.shape[0])
        self.DeclareVectorInputPort("nominal_action", self.nu)
        self.action_output_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()

        # If we have both static CBFs and a switching CBF,
        # then the first "num_static_cbf_constraints" elements in the
        # relative_degree and alpha list correspond to the static CBFs,
        # and the last one corresponds to the switching CBF.
        self.static_cbf_constraints = ([
            CbfConstraint(
                h=static_cbfs[i],
                f=f,
                g=g,
                x=x,
                relative_degree=relative_degree[i],
                alpha=alpha[i],
            )
            for i in range(num_static_cbf_constraints)
        ] if static_cbfs is not None else None
        )
        self.switching_cbf_constraints = ([
            CbfConstraint(
                h=switching_cbfs[i],
                f=f,
                g=g,
                x=x,
                relative_degree=relative_degree[num_static_cbf_constraints],
                alpha=alpha[num_static_cbf_constraints],
            )
            for i in range(switching_cbfs.shape[0])
        ] if switching_cbfs is not None else None
        )

    def action_output_port(self):
        return self.get_output_port(self.action_output_index)
    
    def switching_policy_1(
        self,
        x_val: np.ndarray,
    ):
        """
        This function defines the first switching policy discussed in the paper.
        Before we state the switching conditions, we first define the an admissible
        control set U(x) as follows:
        U(x) = {u| Au<= c, and all the normal CBF constraints hold }
        Since our verfication method also verifies the feasbility of the CBF-QP with
        control limits and the normal CBF constraints, then this U(x) should not be
        empty.
        The parameter is a tuple (eta_l, eta_h) where:
          eta_l: threshold for the first switching condition (can be positive for
                 proactive switching before infeasibility).
          eta_h: threshold for the second switching condition (from switching policy II),
                 requiring the candidate CBF's HOCBF derivatives to be >= eta_h.
        Now, we give the switching conditions. The switching signal will switch from
        i to j if the following conditions are satisfied:
         1. max_{u in U(x)} L_{g}h_i(x) u + L_f h_i(x) + alpha * h_i(x) <= eta_l
        meaning that the current switching CBF no longer has sufficient feasibility margin.
         2. max_{u in U(x)} L_{g}h_i(x) u + L_f h_i(x) + alpha * h_i(x) >= eta_h
        meaning that the candidate CBF j is a more feasible HOCBF.
         3. The current state is also in the region of HOCBF that we are switching to.
        Of course, if there are no normal CBFs, then U(x) only contains the control
        limits. Also, this switching policy function should not be called if there is 
        no switching CBF condition in the QP.
        """
        print("current state:", x_val)
        assert self.switching_cbfs is not None
        (eta_l, eta_h) = self.switching_policy_param
        switching = False
        optimal_cost = self._find_max_of_cbf_constriant(
            switching_cbf_index=self.choose_switching_cbf_index,
            x_val=x_val
        )
        if optimal_cost >= -eta_l:
            switching = True

        if switching:
            for i in range(self.switching_cbfs.shape[0]):
                if i == self.choose_switching_cbf_index:
                    continue
                optimal_cost = self._find_max_of_cbf_constriant(
                    switching_cbf_index=i,
                    x_val=x_val
                )
                in_target_region = self._current_state_in_candidate_hocbf_region(
                    switching_cbf_index=i,
                    x_val=x_val,
                    ref_val=0.0
                )
                if optimal_cost <= -eta_h and in_target_region:
                    self.choose_switching_cbf_index = i
                    break

    def switching_policy_2(
        self,
        x_val: np.ndarray,
    ):
        """
        This function defines the second switching policy discussed in the paper.
        When the relative degree of the switching CBF is 1, then we use the switching
        policy II in the ACC paper. Ohterwise we use the switching policy II in the
        Journla extension.
        
        if reltive degree == 1:
        The switching signal will switch from i to j if the following conditions
        are met:
        1. h_i(x) <= \eta_low,
        meaning that we already reach the boundary of the current CBF;
        2. h_j(x) >= \eta_high,
        and eta_high - eta_low = constant eps > 0
        
        else if relative degree > 1:
        The switching signal will switch from i to j if the following conditions
        are met:
        1. max {phi_i0(x), phi_i1(x),...phi_ir-1(x)} <= eta_l
        meaning that we already reach the boundary of the current HOCBF region;
        2. min {phi_j0(x), phi_j1(x),...phi_jr-1(x)} >= eta_h
        and eta_high - eta_low = constant eps > 0
        meaning that the candidate HOCBF has a further boundary.

        Both the functions "_current_state_near_hocbf_bounary()"
        and "_current_state_in_candidate_hocbf_region()" are adaptive to the
        relative degree of switching CBFs. See doc strings of these functions.
        """
        print("Current State: ", x_val)
        eta_low = self.switching_policy_param[0]
        eta_high = self.switching_policy_param[1]
        switching_condition_a = False
        
        # check whether condition a is true:
        near_hocbf_boundary = self._current_state_near_hocbf_boundary(
            switching_cbf_index=self.choose_switching_cbf_index,
            x_val=x_val,
            ref_val=eta_low
        )
        if near_hocbf_boundary:
            switching_condition_a = True

        # if a is true then we check condition b:
        if switching_condition_a:
            for i in range(self.switching_cbfs.shape[0]):
                if i == self.choose_switching_cbf_index:
                    continue
                optimal_cost = self._find_max_of_cbf_constriant(
                        switching_cbf_index=i,
                        x_val=x_val,
                    )
                in_candidate_hocbf_region = self._current_state_in_candidate_hocbf_region(
                    switching_cbf_index=i,
                    x_val=x_val,
                    ref_val=eta_high
                )
                if  in_candidate_hocbf_region and optimal_cost < 0:
                    self.choose_switching_cbf_index = i
                    break
        
    def calc_action(
        self,
        context: drake_sys_frame.Context,
        output
    ):
        x_val = self.get_input_port(0).Eval(context)
        u_d = self.get_input_port(1).Eval(context)

        if self.switching_policy_id == 1:
            self.switching_policy_1(
                x_val=x_val
            )
        elif self.switching_policy_id == 2:
            self.switching_policy_2(
                x_val=x_val
            )
        else:
            raise ValueError("switching_policy_id is invalid")
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        cost = 0.5 * (u - u_d).dot(u - u_d)
        prog.AddQuadraticCost(e=cost, is_convex=True)
        self.switching_cbf_constraints[self.choose_switching_cbf_index].add_to_prog(
            prog=prog,
            x_val=x_val,
            u=u
        )
        if self.static_cbf_constraints is not None:
            for static_cbf_constraint in self.static_cbf_constraints:
                static_cbf_constraint.add_to_prog(
                    prog=prog,
                    x_val=x_val,
                    u=u
                )
        (A, c) = self.control_limits
        prog.AddLinearConstraint(
            A=A, lb=np.array([-np.inf]*c.shape[0]), ub=c, vars=u
        )
        result = solve_with_id(
            prog=prog,
            solver_id=self.solver_id,
            solver_options=self.solver_options
        )
        assert result.is_success()
        optimal_u = result.GetSolution(u)
        output.set_value(optimal_u)
    
    def _find_max_of_cbf_constriant(
        self,
        switching_cbf_index: int,
        x_val: np.ndarray,
    ) -> float:
        """
        This function finds the following value:
        max_{u in U(x)} L_{g}h_i(x) u + L_f h_i(x) + alpha * h_i(x)
        where U(x) = {u| Au <= c and all the static CBF constraints hold }
        return the optimal value of the above maximization problem
        (Not the optimal u, but the optimal value of the objective function)
        This is equivalent to returning the minimum of the following cost:
        - L_{g}h_i(x) u - L_f h_i(x) - alpha * h_i(x)
        subject to the same constraints.
        """
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        current_switching_constraint = self.switching_cbf_constraints[
            switching_cbf_index
        ]
        current_switching_constraint.add_to_prog_as_cost(
            prog=prog,
            x_val=x_val,
            u=u
        )
        (A, c) = self.control_limits
        prog.AddLinearConstraint(
            A=A, lb=np.array([-np.inf]*c.shape[0]), ub=c, vars=u
        )
        if self.static_cbf_constraints is not None:
            for static_cbf_constraint in self.static_cbf_constraints:
                static_cbf_constraint.add_to_prog(
                    prog=prog,
                    x_val=x_val,
                    u=u
                )
        result = solve_with_id(
            prog=prog,
            solver_id=self.solver_id,
            solver_options=self.solver_options
        )
        assert result.is_success()
        optimal_cost = result.get_optimal_cost()
        return optimal_cost

    def _current_state_in_candidate_hocbf_region(
        self,
        switching_cbf_index: int,
        x_val: np.ndarray,
        ref_val: Optional[float]=0.0
    ) -> bool:
        """
        This function checks whether the current state x_val is in the
        region defined by the candidate HOCBF with index switching_cbf_index.
        In other words, we check whether 
        min {phi_j0(x), phi_j1(x),...phi_jr-1(x)} >= ref_val > 0,
        where phi_jk(x) is the k-th lower lie derivative of h_j(x).
        If the switching CBF's relative degree is 1, then this function only
        checks 
        phi_j0(x) = h_j(x) >= ref_val.
        """
        candidate_hocbf = self.switching_cbfs[switching_cbf_index]
        relative_degree = self.relative_degree[-1]
        alpha = self.alpha[-1]
        phi_functions = lower_lie_derivatives(
            poly=candidate_hocbf,
            vector_field=self.f,
            variables=self.x,
            relative_degree=relative_degree,
            betas=alpha
        )
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        candidate_hocbf_val = candidate_hocbf.Evaluate(env)
        phi_functions = np.concatenate(
            (np.array([candidate_hocbf]), phi_functions)
        )
        min_phi_value = np.inf
        for each_item in phi_functions:
            each_item_val = each_item.Evaluate(env)
            if each_item_val < min_phi_value:
                min_phi_value = each_item_val
        return (min_phi_value >= 0.0) and (candidate_hocbf_val >= ref_val)

    def _current_state_near_hocbf_boundary(
        self,
        switching_cbf_index: int,
        x_val: np.ndarray,
        ref_val: Optional[float]=0.0
    ) -> bool:
        """
        This function checks whether the current state x_val is near the
        boundary of the candidate HOCBF with index switching_cbf_index.
        In other words, we check whether
        max {phi_j0(x), phi_j1(x),...phi_jr-1(x)} <= ref_val,
        where phi_jk(x) is the k-th lower lie derivative of h_j(x).
        If the switching CBF's relative degree is 1, then this function only
        checks 
        phi_j0(x) = h_j(x) <= ref_val.
        """
        candidate_hocbf = self.switching_cbfs[switching_cbf_index]
        relative_degree = self.relative_degree[-1]
        alpha = self.alpha[-1]
        phi_functions = lower_lie_derivatives(
            poly=candidate_hocbf,
            vector_field=self.f,
            variables=self.x,
            relative_degree=relative_degree,
            betas=alpha
        )
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        phi_functions = np.concatenate(
            (np.array([candidate_hocbf]), phi_functions)
        )
        max_phi_value = -np.inf
        for each_item in phi_functions:
            each_item_val = each_item.Evaluate(env)
            if each_item_val > max_phi_value:
                max_phi_value = each_item_val
        return max_phi_value <= ref_val
