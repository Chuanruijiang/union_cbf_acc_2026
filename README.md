# Union-CBF Verification Framework

## Overview

This code base verifies safety conditions for keeping a control affine system safe inside a **union** of safe regions.

---

## System Model

We consider a **control-affine polynomial system**:

$$\dot{x}(t) = f(x(t)) + G(x(t))u(t)$$

There are $N$ Control Barrier Functions (CBFs) $h_1(x)$,..., $h_{N_h}(x)$ and the goal is to keep the system forward invariant within the union of sets {x | $h_p(x) \geq 0$} for all $p\in$ P = {1,..., $N_h$}.

---

## Switching Safe Controller

The union safety is acheived by a **switching CBF-QP controller** selects among $N_h$ quadratic programs, one per CBF:

$$
\begin{aligned}
\forall p \in P, k_p(x) = \arg\min_u \quad &\tfrac{1}{2} u^T H(x) u + c^T(x) u\\
\text{s.t. }\quad &L_f h_p(x) + L_G h_p(x) u \geq -\kappa h_p(x)\\
& Au \leq c
\end{aligned}
$$ 

where $Au \leq c$ is the control input limits.

A switching signal $\sigma(t) \in P$ selects the active controller at each moment, yielding $u(t) = k_{\sigma(t)}(x(t))$.

---

## Switching Strategies and Verification
We consider the following two types of swithcing strategies for $\sigma(t)$:

**Type-I:** $\sigma(t)$ switches when the current selected CBF-QP is nearly infeasible at current system's state.

**Type-II:** $\sigma(t)$ switches when the current system's state is close to the boundary of the current selected CBF.

We design different verifiers, namely **Verif-I** and **Verif-II**, to ensure that the swtiching CBF-QP always has a feasible control under Type-I and II strategies during the runtime.

---
## Reference
The paper of this project is published in American Control Conference (ACC), 2026

**Paper:** *Verification Framework for the Union of Control Barrier Functions*  
**Authors:** Chuanrui Jiang and Andrew Clark  
**Affiliation:** Electrical and Systems Engineering Dept., McKelvey School of Engineering, Washington University in St. Louis

**Citation:**
```
@inproceedings{jiang2026unioncbf,
  title     = {Verification Framework for the Union of Control Barrier Functions},
  author    = {Chuanrui Jiang and Andrew Clark},
  booktitle = {American Control Conference (ACC)},
  year      = {2026}
}
```
---

## Repository Structure

``` 
union_cbf_base/          # core libraries
    - controller:        # defines the switching CBF-QP controller.
    - union_cbf_I:       # Main script for Verif-I.
    - union_cbf_II:      # Main script for Vereif-II


examples/                             # experiments
    -nonlinear toy:                   # The experiment in Section VI-B of the paper, the CBF layouts
    -single_integrator_2D:            # The experiments in Section VI-A and the verif-I/II time comparison in Section VI-B
    -single_integrator_2D_const_vel:  # The experiment at the end of Section VI-B.
  

tests/                   # Unit tests for core modules
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Dependencies

See [requirements.txt](requirements.txt). Core dependencies include `numpy`, `scipy`, `cvxpy`, and a compatible SOS solver (MOSEK).

