An optimization problem consists of two main constituents: a number of decision variables $\boldsymbol{x}$ and an objective function $\boldsymbol{J}$. Furthermore, we might have some parameters $\boldsymbol{\theta}$ (e.g., wind speed, TI, ...) which also determines the optimal decision variables $\boldsymbol{x}^{\star}$. The optimization problem can be written as:

$$\min_{\boldsymbol{x}} \boldsymbol{J}(\boldsymbol{x}, \boldsymbol{\theta}) \qquad \text{s.t.} \qquad \boldsymbol{g}(\boldsymbol{x}, \boldsymbol{\theta}) \leq \boldsymbol{0}, \quad \boldsymbol{h}(\boldsymbol{x}, \boldsymbol{\theta}) = \boldsymbol{0}.$$

*Difficulties:*

- **Multi-objective:** The objective function $\boldsymbol{J}$ is vector-valued, meaning the optimization results in a Pareto front with no clear single optimal solution.
- **Integer decision variables:** The decision variables $\boldsymbol{x}$ are (partly) integer-valued, making the optimization problem significantly more complex (MI...).
- **Discontinuous:** The objective function $J$ is discontinuous, which can make solving the optimization problem infeasible.
- **Non-numeric:** For certain objectives functions/metrix $J$, the value $J(\boldsymbol{x},\boldsymbol{\theta}) \in \\{ \mathrm{good}, \mathrm{bad}, \mathrm{worse} \\}$ are non-numeric values.
- **Absent gradient:** For most objective functions $J$, the gradient $\partial J(\boldsymbol{x}, \boldsymbol{\theta}) / \partial \boldsymbol{x}$, which makes finding solutions difficult (use of gradient-free/global methods, e.g., genetic algorithms, simulated annealing, etc.).
- **Non-convex:** The objective function $J$ is non-convection and/or highly nonlinear, meaning no unique minimum exists.
- **Expensive:** The objective function $J(\boldsymbol{x}, \boldsymbol{\theta})$ is expensive to evaluate, which makes solving the optimization problem slow.
- **Undefined:** For some combinations of decision variables $\boldsymbol{x}$ and objective functions $J$ the values $J(\boldsymbol{x},\boldsymbol{\theta}) = \varnothing$ is undefined.

## 1. Control strategies:

We consider two control strategies:

1. Greedy control, i.e., 'business-as-usual'.
2. Yaw steering, i.e., yaw angle $\gamma_{i}$ for each $i$-th WT.
   - This can also be dependent on the mean wind speed and wind direction (over time, by means of a LUT), i.e. $\gamma_{i}(U_{\infty},\phi_{U})$.
3. Downregulation of downrating each WT, such that $P_{i} < P_{\mathrm{max},i}$ for each $i$-th WT, where $P_{\mathrm{max},i}$ is the maximum available power for each WT (based on the inflow conditions).
   - Note that the 'control' variables of the wind turbine are generator torque and (individual) blade pitch, $\tau_{\mathrm{gen}}$ and $\beta_{j}$, respectively.
   - Instead, the variables we can change are the $C_{T}$ and $C_{P}$ curves of each WT (used by FLORIS/PyWake). We have a mapping $C_{T},C_{P} \mapsto P_{i}$.
   - Note that the former are all at least variables of the wind speed $U_{\infty}$.
4. Shutdown/start-up of individual turbines, where we either do this at fixed times (are these times a decision variable then) or reactive (based on, e.g., data from bird migration or thunderstorms/hurricanes).
  
Control variables (for each $i$-th WT):

- **Available:** Yaw angle $\gamma_{i}$, ... curve $C_{T}$, ... $C_{P}.
- **Inaccesable:** Generator torque $\tau_{\mathrm{gen}}$, blade pitch $\beta_{i}$.
  
## 2. Control co-design:

There are different scenarios for (offline) control co-design:

1. Topology optimization, where the position $(x_{i},y_{i})$ of each WT is a decision variable. This can be combined with greedy control, yaw steering, or downregulation/derating.
2. Repowering, where the topology of the wind farm is fixed but where we can choose a WT model (for each fixed position) based on $n$ different models of WTs available.

## 3. Metrics:

Here, we discuss all the metrics at play:

- **Economical/performance:** Anual energy production (AEP), levelised cost of energy (LCoE).
- **Lifetime:** Fatigue loads (on the blades), generator shaft expected lifetime.
- **Ecological:** Curtailment for bats (shutdown and start-up).
- **Sociological:** Noise curtailment by means of downregulation.
