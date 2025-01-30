## Control strategies:

We consider two control strategies:

1. Greedy control, i.e., 'business-as-usual'.
2. Yaw steering, i.e., yaw angle $\gamma_{i}$ for each $i$-th WT.
   - This can also be dependent on the mean wind speed and wind direction (over time, by means of a LUT), i.e. $\gamma_{i}(U_{\infty},\phi_{U})$.
3. Downregulation of downrating each WT, such that $P_{i} < P_{\mathrm{max},i}$ for each $i$-th WT, where $P_{\mathrm{max},i}$ is the maximum available power for each WT (based on the inflow conditions).
   - Note that the 'control' variables of the wind turbine are generator torque and (individual) blade pitch, $\tau_{\mathrm{gen}}$ and $\beta_{j}$, respectively.
   - Instead, the variables we can change are the $C_{T}$ and $C_{P}$ curves of each WT (used by FLORIS/PyWake). We have a mapping $C_{T},C_{P} \mapsto P_{i}$.
   - Note that the former are all at least variables of the wind speed $U_{\infty}$.
  
Control variables (for each $i$-th WT):

- **Available:** Yaw angle $\gamma_{i}$, ... curve $C_{T}$, ... $C_{P}.
- **Inaccesable:** Generator torque $\tau_{}$
  
## Control co-design:

There are different 
