# """
# This script defines the controllers to be compared in the
# setup of single integrator 2D. The controllers are:
# 1. BNCBF-QP,
# 2. Composite CBF-QP,
# """

import numpy as np
from typing import Tuple

import pydrake.symbolic as sym
import pydrake.solvers as solvers
import pydrake.systems.framework as drake_sys_frame

from union_cbf_base.controller import CbfConstraint
from union_cbf_base.utils import solve_with_id

# this class defines the BNCBF-QP controller:
class BncbfController(drake_sys_frame.LeafSystem):
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        cbfs: np.ndarray,
        alpha: float,
        control_limits: Tuple[np.ndarray, np.ndarray]
    ):
        super().__init__()
        self.x=x
        self.f=f
        self.g=g
        self.cbfs=cbfs
        self.alpha=alpha
        self.control_limits=control_limits
        assert x.shape[0] == 2
        self.nx = 2
        self.nu = 2
        self.num_cbfs = cbfs.shape[0]

        # prepare all the cbf constraints:
        self.cbf_constraints = [
            CbfConstraint(h=cbf, f=f, g=g, x=x, alpha=alpha)
            for cbf in cbfs
        ]
        # the sequence of cbf constraints is the same as the 
        # sequence of cbfs.

        # declare controller's ports:
        # input port 0: state feedback
        # input port 1: desired control input u_d
        self.DeclareVectorInputPort("state", self.nx)
        self.DeclareVectorInputPort("u_d", self.nu)
        # output port 0: safe control input
        self.action_output_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()
        
    def action_output_port(self):
        return self.get_output_port(self.action_output_index)
    
    def calc_action(self, context, output):
        # get the current state:
        x_val = self.get_input_port(0).Eval(context)
        u_d = self.get_input_port(1).Eval(context)
        eval_env = {self.x[i]: x_val[i] for i in range(self.nx)}

        # based on the current state, choose the cbf using the
        # formulation of BNCBF. the chosen cbf h(x) should be
        # max over all the cbfs h_i(x) at the current state x.
        cbf_values = np.array([
            self.cbfs[i].Evaluate(eval_env)
            for i in range(self.num_cbfs)
        ])
        # the cbf_values also follows the same sequence as the
        # cbfs and the cbf_constraints.
        max_cbf_val = np.max(cbf_values)
        max_indices = np.where(cbf_values == max_cbf_val)[0]
        chosen_cbf_constraints = [
            self.cbf_constraints[i]
            for i in max_indices
        ]

        # compute the optimal control by solving the QP with the
        # chosen cbf constraint:
        prog = solvers.MathematicalProgram()
        control_var = prog.NewContinuousVariables(self.nu, "u")
        cost = 0.5 * (control_var - u_d).dot(control_var - u_d)
        prog.AddQuadraticCost(e=cost, is_convex=True)
        # add the cbf_constraint:
        for each_constraint in chosen_cbf_constraints:
            each_constraint.add_to_prog(
                prog=prog,
                x_val=x_val,
                u=control_var
            )
        (A, c) = self.control_limits
        prog.AddLinearConstraint(
            A=A, lb=np.array([-np.inf]*c.shape[0]), ub=c, vars=control_var
        )
        result = solve_with_id(prog=prog)
        assert result.is_success()
        safe_u = result.GetSolution(control_var)
        output.set_value(safe_u)

# this class defines the Composite CBF-QP controller:
class CompositeCbfController(drake_sys_frame.LeafSystem):
    def __init__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        cbfs: np.ndarray,
        alpha: float,
        control_limits: Tuple[np.ndarray, np.ndarray],
        composite_cbf_param: Tuple[float, float],
    ):
        """
        The CBF in this class composites the union cbfs into a single cbf
        h(x) such that the region {x | h(x) >= 0} is an approximation of 
        the union region defined by the given cbfs. 
        i.e., \cup_{i} {x | h_i(x) >= 0}.

        See the paper for more details:
        Molnar, Tamas G., and Aaron D. Ames. 
        "Composing control barrier functions for complex safety 
        specifications." 
        IEEE Control Systems Letters 7 (2023): 3615-3620.

        we use the composition method equation (22) in the paper, which 
        is defined as follows:

        h(x) = (1/k) * ln(sum_{i} exp(k * h_i(x))) - b/k

        where k is a smoothing parameter and should be strictly positive.
        and b can be ln(num_cbfs)
        """
        super().__init__()
        self.x=x
        self.f=f
        self.g=g
        self.cbfs=cbfs
        self.alpha=alpha
        self.control_limits=control_limits
        assert x.shape[0] == 2
        self.nx = 2
        self.nu = 2
        self.num_cbfs = cbfs.shape[0]

        # create the composite cbf from these given CBFs.
        composite_cbf = self._create_composite_cbf(composite_cbf_param)

        # prepare composite cbf constraints:
        self.cbf_constraint = CbfConstraint(
            h=composite_cbf,
            f=f,
            g=g,
            x=x,
            alpha=alpha
            )
        # the sequence of cbf constraints is the same as the 
        # sequence of cbfs.

        # declare controller's ports:
        # input port 0: state feedback
        # input port 1: desired control input u_d
        self.DeclareVectorInputPort("state", self.nx)
        self.DeclareVectorInputPort("u_d", self.nu)
        # output port 0: safe control input
        self.action_output_index = self.DeclareVectorOutputPort(
            "action", self.nu, self.calc_action
        ).get_index()
        
    def action_output_port(self):
        return self.get_output_port(self.action_output_index)
    
    def calc_action(self, context, output):
        # get the current state:
        x_val = self.get_input_port(0).Eval(context)
        u_d = self.get_input_port(1).Eval(context)

        # compute the optimal control by solving the QP with all the
        # cbf constraints:
        prog = solvers.MathematicalProgram()
        control_var = prog.NewContinuousVariables(self.nu, "u")
        cost = 0.5 * (control_var - u_d).dot(control_var - u_d)
        prog.AddQuadraticCost(e=cost, is_convex=True)
        # add the cbf_constraint:
        self.cbf_constraint.add_to_prog(
            prog=prog,
            x_val=x_val,
            u=control_var
        )
        (A, c) = self.control_limits
        prog.AddLinearConstraint(
            A=A, lb=np.array([-np.inf]*c.shape[0]), ub=c, vars=control_var
        )
        result = solve_with_id(prog=prog)
        assert result.is_success()
        safe_u = result.GetSolution(control_var)
        output.set_value(safe_u)

    def _create_composite_cbf(
            self,
            composite_cbf_param: Tuple[float, float]
        ) -> sym.Expression:
        assert len(composite_cbf_param) == 2
        k, b = composite_cbf_param
        composite_cbf = (1 / k) * sym.log(sum(
            sym.exp(k * self.cbfs[i].ToExpression())
            for i in range(self.num_cbfs)
            )
        ) - b / k
        return composite_cbf









