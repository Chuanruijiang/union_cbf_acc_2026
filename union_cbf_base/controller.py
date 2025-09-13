from typing import Optional, Tuple

import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.framework as drake_sys_frame

from union_cbf_base.utils import(
    solve_with_id
)


class CbfConstraint:
    """
    Add the linear constraint 
    dhdx * f(x) + dhdx * g(x)*u >= -alpha * h(x) on u.
    """
    def __init__(
        self,
        h: sym.Polynomial,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        alpha:float,
    ):
        dhdx = h.Jacobian(x)
        dhdx_times_f = dhdx.dot(f)
        dhdx_times_g = dhdx @ g
        self.rhs = -alpha * h - dhdx_times_f
        self.lhs_coeff = dhdx_times_g
        self.x = x

    def add_to_prog(
        self,
        prog: solvers.MathematicalProgram,
        x_val: np.ndarray,
        u: np.ndarray
    ) -> solvers.Binding[solvers.LinearConstraint]:
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        constraint = prog.AddLinearConstraint(lhs_coeff, rhs, np.inf, u)
        return constraint

    def add_as_cost(
        self,
        prog:solvers.MathematicalProgram,
        x_val: np.ndarray,
        u: np.ndarray
    ):
        """
        This function adds the following cost on u:
        dhdx * f(x) + dhdx * g(x)*u + alpha * h(x)
        """
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        cost = lhs_coeff.dot(u) + (-rhs)
        prog.AddLinearCost(e=cost)


class SwitchingCBFController(drake_sys_frame.LeafSystem):
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        cbfs: np.ndarray,
        alpha: float,
        control_limits: Tuple[np.ndarray, np.ndarray],
        waypoints: np.ndarray,
        switching_policy_id: int, # 1 or 2
        solver_id: Optional[solvers.SolverId],
        solver_options: Optional[solvers.SolverOptions],
    ):
        super().__init__()
        self.x=x
        self.f=f
        self.g=g
        self.cbfs=cbfs
        self.alpha=alpha
        self.control_limits=control_limits
        assert waypoints.shape[1] == x.shape[0]
        self.waypoints=waypoints
        assert switching_policy_id == 1 \
            or switching_policy_id == 2
        self.switching_policy_id=switching_policy_id
        self.solver_id=solver_id
        self.solver_options=solver_options
        
        self.nu = g.shape[1]
        self.nx = x.shape[0]
        self.choose_cbf_index = 0  # default to the first cbf
        
        self.DeclareVectorInputPort("state", x.shape[0])
        self.action_ouput_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()
        self.cbf_constraints = [
            CbfConstraint(h=cbf, f=f, g=g, x=x, alpha=alpha)
            for cbf in cbfs
        ]

    def action_output_port(self):
        return self.get_output_port(self.action_ouput_index)
    
    def nominal_action(
        self,
        x_val: np.ndarray
    ) -> np.ndarray:
        gain = 2.0
        assert x_val.shape == (self.nx,)
        if np.linalg.norm(self.waypoints[0] - x_val) < 0.01:
            if self.waypoints.shape[0] == 1:
                return np.zeros(shape=(self.nu,))
            elif self.waypoints.shape[0] > 1:
                x_target = self.waypoints[1]
                self.waypoints = self.waypoints[1:,:]
            else:
                raise ValueError("waypoints shape is invalid")
        else:
            x_target = self.waypoints[0]
        return gain * (x_target - x_val)

    def switching_policy_1(
        self,
        x_val: np.ndarray,
        u_d: np.ndarray,
    ):
        """
        This function defines the first switching policy discussed in the paper.
        Before we state the switching conditions, we first define the an admissible
        control set U(x) as follows:
        U(x) = {u| Au<= c and u_d^T u >= mu}
        this means that we hope every control action u is within the control limits
        and also has a positive projection on the nominal action u_d.
        Now, we give the switching conditions. The switching signal will switch from
        i to j if the following conditions are satisfied:
         1. max_{u in U(x)} L_{g}h_i(x) u + L_f h_i(x) + alpha * h_i(x) <= 0
        meaning that there is no admissible control that can make the CBF condition
        of the current active CBF hold;
         2. max_{u in U(x)} L_{g}h_j(x) u + L_f h_j(x) + alpha * h_j(x) >= η > 0
        meaning that there exists an admissible control that can make the CBF condition
        of the candidate CBF (with index j) hold.
         3. h_j(x) >= 0
        meaning that the candidate CBF is valid at the current state.
        """
        mu = 1e-3
        eta = 1e-3
        # check the first switching condition
        swtiching_condition_1 = False
        optimal_cost = self._find_max_of_cbf_constriant(
            cbf_index=self.choose_cbf_index,
            x_val=x_val,
            ud=u_d,
            mu=mu
        )
        if optimal_cost <= 0 and self.choose_cbf_index < self.cbfs.shape[0] - 1:
            # the first condition is met, check the second and third conditions
            swtiching_condition_1 = True
        
        # check the second and third condition:
        if swtiching_condition_1:
            candidate_cbf_index = self.choose_cbf_index + 1
            optimal_cost = self._find_max_of_cbf_constriant(
                cbf_index=candidate_cbf_index,
                x_val=x_val,
                ud=u_d,
                mu=mu
            )
            if optimal_cost >= eta and self.cbfs[candidate_cbf_index].Evaluate(
                {self.x[i]: x_val[i] for i in range(x_val.size)}
            ) >= 0:
                self.choose_cbf_index = candidate_cbf_index

    def switching_policy_2(
        self,
        x_val: np.ndarray,
        u_d: np.ndarray,
    ):
        """
        This function defines the second switching policy discussed in the paper.
        The switching signal will switch from i to j if the following conditions
        are met:
        1. h_i(x) = 0,
        meaning that we already reach the boundary of the current CBF;
        2. h_j(x) >= \eta' > 0,
        meaning that the candidate CBF is valid at the current state.
        3. max_{u in U(x)} L_{g}h_j(x) u + L_f h_j(x) + alpha * h_j(x) > 0
        meaning that there exists an admissible control that can make the CBF
        strictly hold.
        """
        mu = 1e-3
        eta_prime = 1e-3
        switching_condition_1 = False

        # check the first switching condition
        if self.cbfs[self.choose_cbf_index].Evaluate(
            {self.x[i]: x_val[i] for i in range(x_val.size)}
        ) <= 1e-4 and self.choose_cbf_index < self.cbfs.shape[0] - 1:
            switching_condition_1 = True
        
        # check the second and third condition:
        if switching_condition_1:
            candidate_cbf_index = self.choose_cbf_index + 1
            if self.cbfs[candidate_cbf_index].Evaluate(
                {self.x[i]: x_val[i] for i in range(x_val.size)}
            ) >= eta_prime:
                optimal_cost = self._find_max_of_cbf_constriant(
                    cbf_index=candidate_cbf_index,
                    x_val=x_val,
                    ud=u_d,
                    mu=mu
                )
                if optimal_cost > 0:
                    self.choose_cbf_index = candidate_cbf_index
        
    def calc_action(
        self,
        context: drake_sys_frame.Context,
        output
    ):
        x_val = self.get_input_port(0).Eval(context)
        u_d = self.nominal_action(x_val)
        print("x_val:", x_val)
        print("u_d:", u_d)
        if self.switching_policy_id == 1:
            self.switching_policy_1(
                x_val=x_val,
                u_d=u_d
            )
        elif self.switching_policy_id == 2:
            self.switching_policy_2(
                x_val=x_val,
                u_d=u_d
            )
        else:
            raise ValueError("switching_policy_id is invalid")
        print("active cbf index:", self.choose_cbf_index)
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        cost = 0.5 * (u - u_d).dot(u - u_d)
        prog.AddQuadraticCost(e=cost, is_convex=True)
        self.cbf_constraints[self.choose_cbf_index].add_to_prog(
            prog=prog,
            x_val=x_val,
            u=u
        )
        (A, c) = self.control_limits
        prog.AddLinearConstraint(
            A=A, lb=np.full_like(c, -np.inf), ub=c, vars=u
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
        cbf_index: int,
        x_val: np.ndarray,
        ud: np.ndarray,
        mu: float,
    ) -> float:
        """
        This function finds the following value:
        max_{u in U(x)} L_{g}h_i(x) u + L_f h_i(x) + alpha * h_i(x)
        where U(x) = {u| Au <= c and u_d^T u >= mu}
        1. cbf_index: the index of the cbf h_i
        2. x_val: the current state value
        3. ud: the nominal action u_d
        4. mu: the lower bound of the projection of u on u_d
        5. return the optimal value of the above maximization problem
           (Not the optimal u, but the optimal value of the objective function)
        """
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        ud = self.nominal_action(x_val)
        self.cbf_constraints[cbf_index].add_as_cost(
            prog=prog,
            x_val=x_val,
            u=u
        )
        (A, c) = self.control_limits
        prog.AddLinearConstraint(
            A=A, lb=np.full_like(c, -np.inf), ub=c, vars=u
        )
        prog.AddLinearConstraint(
            a=ud,
            lb=(mu if np.linalg.norm(ud) > 1e-6 else 0.0),
            ub=np.inf,
            vars=u
        )
        result = solve_with_id(
            prog=prog,
            solver_id=self.solver_id,
            solver_options=self.solver_options
        )
        assert result.is_success()
        optimal_cost = result.get_optimal_cost()
        return optimal_cost



