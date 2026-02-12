from typing import Optional, Tuple, Union

import numpy as np

import pydrake.solvers as solvers
import pydrake.symbolic as sym
import pydrake.systems.framework as drake_sys_frame

from union_cbf_base.utils import(
    solve_with_id
)


class CbfConstraint:
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
        self.rhs = alpha * h + dhdx_times_f
        self.lhs_coeff = -dhdx_times_g
        self.x = x

    def add_to_prog(
        self,
        prog: solvers.MathematicalProgram,
        x_val: np.ndarray,
        u: np.ndarray
    ) -> solvers.Binding[solvers.LinearConstraint]:
        """
        Add the linear constraint 
        -dhdx * g(x)*u <= dhdx * f(x) + alpha * h(x) on u.
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

    def add_as_cost(
        self,
        prog:solvers.MathematicalProgram,
        x_val: np.ndarray,
        u: np.ndarray
    ):
        """
        This function adds the following cost on u:
        cost = - dhdx * g(x)*u - dhdx * f(x) - alpha * h(x)
        This part will be used for checing the swtiching conditions
        for swtiching policy 1. In switching policy 1, we would like
        to see the minimum value of the cost over the admissible
        control set U(x) = {u| Au <= c}. The switching policy will
        ready to switch to some other CBF if the minimum value above
        is greater than zero, because in such case, the current CBF 
        is infeasible and we need to switch to another CBF to ensure
        safety.
        """
        env = {self.x[i]: x_val[i] for i in range(x_val.size)}
        lhs_coeff = np.array([p.Evaluate(env) for p in self.lhs_coeff])
        rhs = self.rhs.Evaluate(env)
        cost = lhs_coeff.dot(u) - rhs
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
        switching_threshold: Union[float, Tuple[float, float]],
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
        if switching_policy_id == 1:
            assert isinstance(switching_threshold, float)
            self.switching_threshold = switching_threshold
        elif switching_policy_id == 2:
            assert isinstance(switching_threshold, Tuple)
            self.switching_threshold=switching_threshold
        self.nu = g.shape[1]
        self.nx = x.shape[0]
        self.choose_cbf_index = 0  # default to the first cbf
        
        # define input ports:
        # 0: state vector x input
        # 1: nominal action u_d input
        self.DeclareVectorInputPort("state", x.shape[0])
        self.DeclareVectorInputPort("nominal_action", self.nu)

        # define the output port:
        self.action_ouput_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()
        self.cbf_constraints = [
            CbfConstraint(h=cbf, f=f, g=g, x=x, alpha=alpha)
            for cbf in cbfs
        ]

    def action_output_port(self):
        return self.get_output_port(self.action_ouput_index)
    
    def switching_policy_1(
        self,
        x_val: np.ndarray
    ):
        """
        This function defines the first switching policy discussed in the paper.
        Before we state the switching conditions, we first define the an admissible
        control set U(x) as follows:
        U(x) = {u| Au<= c}
        this means that we hope every control action u is within the control limits
        and also has a positive projection on the nominal action u_d.
        Now, we give the switching conditions. The switching signal will switch from
        i to j if the following conditions are satisfied:

         1. max_{u in U(x)} L_{g}h_i(x) u + L_f h_i(x) + alpha * h_i(x) <= 0
        In other words,
            min_{u in U(x)} - (L_{g}h_i(x) u + L_f h_i(x) + alpha * h_i(x)) > 0
        meaning that there is no admissible control that can make the CBF condition
        of the current active CBF hold;

         2. max_{u in U(x)} L_{g}h_j(x) u + L_f h_j(x) + alpha * h_j(x) >= η > 0
        meaning that there exists an admissible control that can make the CBF condition
        of the candidate CBF (with index j) hold.

         3. h_j(x) >= 0
        meaning that the candidate CBF is valid at the current state.
        """
        assert self.switching_threshold is not None \
            and isinstance(self.switching_threshold, float)
        eta = self.switching_threshold
        # check the first switching condition
        swtiching_condition_1 = False
        optimal_cost = self._find_max_of_cbf_constriant(
            cbf_index=self.choose_cbf_index,
            x_val=x_val
        )
        if optimal_cost > 0:
            swtiching_condition_1 = True
        
        # check the second and third condition:
        if swtiching_condition_1:
            for i in range(self.cbfs.shape[0]):
                if i == self.choose_switching_cbf_index:
                    continue
                cbf_i_val = self.cbfs[i].Evaluate(
                    {self.x[i]: x_val[i] for i in range(x_val.size)}
                )
                optimal_cost = self._find_max_of_cbf_constriant(
                    switching_cbf_index=i,
                    x_val=x_val
                )
                if optimal_cost <= -eta and cbf_i_val >= 0:
                    self.choose_cbf_index = i

    def switching_policy_2(
        self,
        x_val: np.ndarray,
    ):
        """
        This function defines the second switching policy discussed in the paper.
        The switching signal will switch from i to j if the following conditions
        are met:
        1. h_i(x) <= \eta_low,
        meaning that we already reach the boundary of the current CBF;
        2. h_j(x) >= \eta_high,
        and eta_high - eta_low = constant eps > 0
        meaning that the candidate CBF is valid at the current state.
        3. max_{u in U(x)} L_{g}h_j(x) u + L_f h_j(x) + alpha * h_j(x) > 0
        meaning that there exists an admissible control that can make the CBF
        strictly hold.
        """
        assert self.switching_threshold is not None \
            and isinstance(self.switching_threshold, Tuple)
        assert self.switching_threshold[0] < self.switching_threshold[1]
        (eta_low, eta_high) = self.switching_threshold
        
        switching_condition_1 = False
        # check the first switching condition
        current_cbf_val = self.cbfs[self.choose_cbf_index].Evaluate(
            {self.x[i]: x_val[i] for i in range(x_val.size)}
        )
        if current_cbf_val <= eta_low:
            switching_condition_1 = True
        
        # check the second and third condition:
        if switching_condition_1:
            for i in range(self.cbfs.shape[0]):
                if i == self.choose_cbf_index:
                    continue
                cbf_i_val = self.cbfs[i].Evaluate(
                    {self.x[i]: x_val[i] for i in range(x_val.size)}
                )
                optimal_cost = self._find_max_of_cbf_constriant(
                    cbf_index=i,
                    x_val=x_val,
                )
                if optimal_cost < 0 and cbf_i_val >= eta_high:
                    self.choose_cbf_index = i
        
    def calc_action(
        self,
        context: drake_sys_frame.Context,
        output
    ):
        x_val = self.get_input_port(0).Eval(context)
        u_d = self.get_input_port(1).Eval(context)


        if self.switching_policy_id == 1:
            self.switching_policy_1(
                x_val=x_val,
            )
        elif self.switching_policy_id == 2:
            self.switching_policy_2(
                x_val=x_val,
            )
        else:
            raise ValueError("switching_policy_id is invalid")
        

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
            A=A, lb=np.array([-np.inf]*c.shape[0]), ub=c, vars=u
        )
        result = solve_with_id(
            prog=prog,
            solver_id=self.solver_id,
            solver_options=self.solver_options
        )
        assert result.is_success()
        safe_u = result.GetSolution(u)
        output.set_value(safe_u)
    
    def _find_max_of_cbf_constriant(
        self,
        cbf_index: int,
        x_val: np.ndarray,
    ) -> float:
        """
        This function finds the following value:
        min_{u in U(x)} -L_{g}h_i(x) u - L_f h_i(x) - alpha * h_i(x)
        where U(x) = {u| Au <= c}
        1. cbf_index: the index of the cbf h_i
        2. x_val: the current state value
        3. return the optimal value of the above maximization problem
           (Not the optimal u, but the optimal value of the objective function)
        """
        prog = solvers.MathematicalProgram()
        u = prog.NewContinuousVariables(self.nu, "u")
        self.cbf_constraints[cbf_index].add_as_cost(
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
        optimal_cost = result.get_optimal_cost()
        return optimal_cost



