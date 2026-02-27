# Union-CBF Verification Framework

**Paper:** *Verification Framework for the Union of Control Barrier Functions*  
**Authors:** Chuanrui Jiang and Andrew Clark  
**Affiliation:** Electrical and Systems Engineering Dept., McKelvey School of Engineering, Washington University in St. Louis

---

## Overview

This repository accompanies the paper that proposes a **union-CBF framework** for safe control of autonomous systems with **non-convex safe regions**. The key idea is to represent a complex safe region as a **union of simpler sets**, each governed by its own Control Barrier Function (CBF), and switch between CBF-QP controllers as the system evolves.

The framework:
- Avoids the over-constraining issue of requiring multiple CBF constraints simultaneously at some states.
- Guarantees forward invariance of the union of simple safe regions.
- Provides Sum-of-Squares (SOS)-based algorithms to verify the safety conditions.

---

## System Model

We consider a **control-affine polynomial system**:
$$
\begin{equation}
\dot{x}(t) = f(x(t)) + G(x(t))u(t)
\end{equation}
$$
where $x(t) \in \mathcal{X} \subseteq \mathbb{R}^{n_x}$, $u(t) \in \mathcal{U} = \{u \mid Au \leq c\}$, and both $f(x)$ and $G(x)$ are polynomials. Assume there are $N_h$ number of CBFs and our goal is to keep the system above forward invariant within $\mathcal{X}_c = \cup_{p=1}^{N_h}\mathcal{X}_p$. Define $\mathcal{P} = \{1,...,N_h\}$.

A **switching CBF-QP controller** is switching between:
$$
\begin{align*}
k_p(x) = \arg\min_u \quad &\tfrac{1}{2} u^T H(x) u + c^T(x) u \\
\text{s.t. }\quad & L_f h_p(x) + L_G h_p(x) u \geq -\kappa h_p(x) \\ 
& Au \leq c
\end{align*}
$$
For all $p\in\mathcal{P}$. 
The active controller is selected by a switching signal $\sigma(t) \in \mathcal{P} = \{1, \ldots, N_h\}$, yielding $u(t) = k_{\sigma(t)}(x(t))$.

---

## Proposition 1

**Proposition 1** provides the core sufficient conditions for forward invariance of $\mathcal{X}_c$ under a switching CBF-QP controller.

> **Proposition 1.** Consider system (1) with switching CBF-QP controller $u(t) = k_{\sigma(t)}(x(t))$. Suppose:
>
> - **(P1-1)** $\sigma(t)$ is right-continuous, piecewise constant, and has only **finitely many switches** in any finite time horizon $[0, T]$.
> - **(P1-2)** $\forall t \geq 0$, $\sigma(t)$ satisfies $h_{\sigma(t)}(x(t)) \geq 0$ and $\exists u \in \text{Int}(\mathcal{U})$ such that
>   $$L_f h_{\sigma(t)}(x(t)) + L_G h_{\sigma(t)}(x(t)) u + \kappa h_{\sigma(t)}(x(t)) > 0.$$
>
> Then $\forall x(0) \in \mathcal{X}_c$, we have $x(t) \in \mathcal{X}_c$ for all $t \geq 0$.

The proof proceeds by induction over switching intervals, exploiting continuity of the trajectory between switches and the Slater's condition implied by P1-2 to ensure the CBF-QP controller is locally Lipschitz (and hence the closed-loop trajectory is absolutely continuous).

---

## Two Switching Strategies

The two strategies below each guarantee conditions P1-1 and P1-2 of Proposition 1 through conditions on the CBF collection $h_1(x), \ldots, h_{N_h}(x)$ alone.

---

### Strategy I — Feasibility-Based Switching (Theorem 2)

**Idea:** Switch to a new CBF index $p'$ when the current CBF-QP's feasible control region becomes too small.

**Switching rules** (given thresholds $\eta_h > \eta_l > 0$): switch from $p$ to $p'$ at state $x(t)$ when:
- **(S1-1)** $h_p(x(t)) \geq 0$ and $h_{p'}(x(t)) \geq 0$.
- **(S1-2)** $\sup_{u \in \text{Int}(\mathcal{U})} \{L_f h_p(x) + L_G h_p(x)u + \kappa h_p(x)\} \leq \eta_l$ &ensp; (current CBF-QP is nearly infeasible).
- **(S1-3)** $\exists u \in \text{Int}(\mathcal{U})$ with $L_f h_{p'}(x) + L_G h_{p'}(x)u + \kappa h_{p'}(x) \geq \eta_h$ &ensp; (new CBF-QP is sufficiently feasible).

> **Theorem 2.** Let $\epsilon_\text{cbf} \leq \eta_l$. If $\forall x \in \mathcal{X}_c$, $\exists p \in \mathcal{P}$ and $u \in \text{Int}(\mathcal{U})$ such that:
>
> - **(C1-1)** $h_p(x) \geq 0$
> - **(C1-2)** $L_f h_p(x) + L_G h_p(x) u + \kappa h_p(x) \geq \epsilon_\text{cbf}$
>
> Then any switching signal following Strategy I satisfies P1-1 and P1-2.

---

### Strategy II — Hysteresis-Based Switching (Theorem 3)

**Idea:** Switch based on the CBF value itself crossing level-set thresholds, with a minimum dwell time $\tau > 0$ at initialization.

**Switching rules** (given $\eta_h > \eta_l > 0$ and initial dwell time $\tau > 0$):
- For $t \in [0, \tau)$: $\sigma(t) = p_0$ where $h_{p_0}(x(0)) \geq 0$.
- For $t \geq \tau$: switch from $p$ to $p'$ when $h_p(x(t)) \leq \eta_l$ and $h_{p'}(x(t)) \geq \eta_h$.

> **Theorem 3.** Let $\epsilon_\text{cbf} > 0$. If $\forall p \in \mathcal{P}$ and $\forall x \in \mathcal{X}_p$, $\exists u \in \text{Int}(\mathcal{U})$ such that:
>
> - **(C2-1)** $L_f h_p(x) + L_G h_p(x) u + \kappa h_p(x) \geq \epsilon_\text{cbf}$
>
> Then any switching signal following Strategy II satisfies P1-1 and P1-2.

> **Remark:** Condition C2-1 implies C1-2, so Theorem 3's condition is strictly stronger than Theorem 2's. An example may be verifiable by Theorem 2 but not Theorem 3.

---

## Verification Theorems (SOS Programs)

Both verification conditions are reformulated as **Sum-of-Squares (SOS) feasibility problems**, solvable via SOSTOOLS or MOSEK.

### Theorem 4 — SOS Verification for Strategy I

Verifies Theorem 2 (Verification Problem 1) via a 3-step procedure:

1. **Partition** $\mathcal{X}_c = \bigcup_{\mathcal{N} \in S_\mathcal{P}} \mathcal{X}_\mathcal{N}$, where $\mathcal{X}_\mathcal{N} = \{x \mid h_p(x) \geq 0, \forall p \in \mathcal{N};\ h_{p'}(x) < 0, \forall p' \notin \mathcal{N}\}$.
2. **Filter empty subsets** using Lemma 4's SOS program (Program 7), obtaining the non-empty collection $S_\text{ne}$.
3. **For each non-empty $\mathcal{X}_\mathcal{N}$**, verify the existence of a feasible control input via the following SOS program:

> **Theorem 4.** $\mathcal{X}'_\mathcal{N}$ satisfies condition T3 (i.e., $\forall x \in \mathcal{X}'_\mathcal{N}$, $\exists p \in \mathcal{N}$ and $u$ with $\Lambda_p(x) u \leq \xi_p(x)$) if the following SOS program **(Program 8)** is feasible:
>
> Find $s_p(x,y)$, $q_p(x,y)$, $r_p(x,y)$ $\forall p \in \mathcal{N}$; and $s_{p'}(x,y)$ $\forall p' \notin \mathcal{N}$, such that:
> $$-1 - \sum_{p \in \mathcal{N}} s_p h_p(x) + \sum_{p' \notin \mathcal{N}} s_{p'} h_{p'}(x) - \sum_{p \in \mathcal{N}} \left[ q_p^T \Lambda_p^T(x) y_p^2 + r_p \left(\xi_p^T(x) y_p^2 + 1\right) \right] \text{ is SOS}$$
> with $s_p(x,y)$ SOS $\forall p \in \mathcal{N}$ and $s_{p'}(x,y)$ SOS $\forall p' \notin \mathcal{N}$.

Here $\Lambda_p(x)$ and $\xi_p(x)$ encode the CBF constraint and input bounds (see equation (6) in the paper).

---

### Theorem 5 — SOS Verification for Strategy II

Verifies Theorem 3 (Verification Problem 2) independently for each $p \in \mathcal{P}$:

> **Theorem 5.** For all $p \in \mathcal{P}$, the condition $\forall x \in \mathcal{X}_p$, $\exists u$ with $\Lambda_p(x) u \leq \xi_p(x)$ holds if the following SOS program **(Program 9)** is feasible:
>
> Find $s(x,y)$, $q(x,y)$, $r(x,y)$ such that:
> $$-1 - s(x,y) h_p(x) - q(x,y) \Lambda_p^T(x) y^2 - r(x,y)\left(\xi_p^T(x) y^2 + 1\right) \text{ is SOS}$$
> with $s(x,y)$ SOS.

> **Remark:** Program 9 is a special case of Program 8 with $\mathcal{N} = \{p\}$, making it significantly simpler and computationally cheaper.

---

## Repository Structure

``` 
union_cbf_base/          # core libraries
    - controller:        # defines the switching CBF-QP controller.
    - inclusion:         # the code base that verifies whether the union of safe region does not overlap with the unsafe region.
    - non_empty_subset   # used to finish step 2 of Theorem 4.
    - plot:              # defines some basic functions to draw unsafe region, CBF super level set, etc. 
    - union_cbf_I:       # Main script that verifies Theorem 4.
    - union_cbf_II:      # Main script that verifies Theorem 5.
    - utils:             # defines basic functions for all the scripts above. 


examples/                   # experiments
    -nonlinear toy:         # The experiment in Section VI-B of the paper, the CBF layouts
    -single_integrator_2D:  # The experiments in Section VI-A and the verif-I/II time comparison in Section VI-B
    -single_integrator_2D_const_vel:  # The experiment at the end of Section VI-B.
  

tests/                   # Unit tests for core modules
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Dependencies

See [requirements.txt](requirements.txt). Core dependencies include `numpy`, `scipy`, `cvxpy`, and a compatible SOS solver (MOSEK or SOSTOOLS-compatible interface).

---

## Citation

If you use this code, please cite the accompanying paper:

```
@inproceedings{jiang2026unioncbf,
  title     = {Verification Framework for the Union of Control Barrier Functions},
  author    = {Chuanrui Jiang and Andrew Clark},
  booktitle = {American Control Conference (ACC)},
  year      = {2026}
}
```
