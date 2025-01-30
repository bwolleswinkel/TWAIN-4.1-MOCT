## Control strategies:

We consider two control strategies:

1. Greedy control, i.e., 'business-as-usual'.
2. Yaw steering, i.e., yaw angle $\gamma_{i}$ for each $i$-th WT.
   - This can also be dependent on the mean wind speed and wind direction (over time, by means of a LUT), i.e. $\gamma_{i}(U_{\infty},\phi_{U})$.
3. Downregulation of downrating each WT, such that $P_{i} < P_{\mathrm{max},i}$ for each $i$-th WT, where $P_{\mathrm{max},i}$ is the maximum available power for each WT (based on the inflow conditions).
