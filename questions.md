---

(2025/05/19) WP4.1 Second Bi-weekly meeting
Moritz: In WP3, the metric evaluation is not as decoupled; WP2 goes to WP3
Nikolay: We don’t consider lifetime as a metric (#Bart why? That’s still a metric, right?)
Moritz: We consider distributions and time-series
Vasilis: Do you have time-series or distribution? What about blow up complexity?
Tuhfe: Partial acces to the WT controller is out of the scope
Irene: Integration is important
Abhinav: You can find the outputs in WP2
Abhinav: Greenhouse Gas, marginal displacement factor (MDF)
Data flows and data usability: we cannot ‘optimize over data’, we use data to tune our models.
Alvarez: Direct metric  which uses both AEP 

---

Is time to start-stop a control variable? From: WP3, DTU lead > T3.4 - Case studies > CaseStudiesDefinition.vsdx 

(2025/05/21) Case Study #1 - Noise
For noise, should we consider noise curtailment modes, instead of power level as a decision variable? So we have a discrete decision variable?
Open-loop or static wind-farm control; no feedback if other turbines are stopped, static curtailment (no awareness of other wind turbines, plans to make this awareness more long-term).
For noise, why are we looking at wake steering? Wouldn’t downregulation be more logical? So the noise model needs to take into effect yaw angle and power setpoint, both.
As Thomas mentioned, wake steering INCREASES noise.
(2025/05/23) Noise case study
Irene/Thomas: impact of ywa steering on produced noise should be taken into account
Andreas, noise model is always complete; contact him! We don’t care about validation, give us the preliminary model.
Surrogate model pwe mode? Why? What about axial-induction, i.e. power regulation? That’s a continuous variable?
I don’t understand noise and power, we need to clear up this discussion; how can power tracking not have an effect?
tufhe: noise model for wake steering?
Andreas: preliminary empirical models, started with implementation on noise.
Thomas: octonoise?? We do not see large differences between turbines. 
🔥 (2024/11/18) These load surrogate models, do thy not already exist? Same as layout optimization?
  https://www.youtube.com/watch?v=mQuvYQmdbtw



“Surrogate Models for Wind Turbine Electrical Power and Fatigue Loads in Wind Farm”, Gasparis et al. (2020)

“Wind Farm Control Optimisation Under Load Constraints Via Surrogate Modelling”, Liew et al. (2024)
(2025/05/24) 

“Comparison of Down-Regulation Strategies for Wind Farm Control and their Effects on Fatigue Loads,” van der Hoek et al. (2018)