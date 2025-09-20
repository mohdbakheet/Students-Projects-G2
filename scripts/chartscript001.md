# Why the current fit fails

1. **Observation model mismatch.**
   You’re fitting **weekly reported cases** to the ODE’s *instantaneous* flow `σ·E(t)` sampled once per week. That ignores within-week dynamics and reporting noise → systematic bias and overconfident curves.

2. **Initial conditions aren’t identifiable.**
   Setting `I0 = first_week_cases` treats **incidence** as **prevalence**. For cholera, infectious duration is short (days), so prevalence is far lower than weekly cases. This alone pushes the system to explode early.

3. **Time scale is off.**
   Cholera incubation 0.5–2 days; infectious 2–5 days. Rates like `σ ≈ 1/10 per week` aren’t biologically consistent. Simulating in **days** and then **aggregating to weeks** is essential.

4. **Population scaling (`N=1e6`) + very small initial infection** makes `S≈N` nearly constant, so `β1` and `β2` become weakly identifiable; curve\_fit compensates by inflating transmission.

5. **Deaths model is unrealistic.**
   Modeling deaths as `μ·I` assumes a death *rate* acting on prevalence. With cholera, deaths are better linked to **incidence with a short delay** and a **case fatality ratio (CFR)**. Your deaths curve drifts because `μ` tries to mimic a lagged, thinned incidence.

6. **Environmental compartment `C` is ad hoc.**
   A logistic `dC/dt = ηI(1−C/κ)+εC−δC` plus a saturating force `β2·S·C/(C+1)` mixes incompatible units and adds extra non-identifiable parameters on sparse data.

7. **Fitting cases only, with least squares.**
   `curve_fit` ignores count noise and the deaths data. For outbreaks you want a **Poisson/NegBin joint likelihood** on (cases, deaths), otherwise the optimizer chases the largest residuals (late weeks) and ruins the early fit.

# A minimal, defensible fix (step-by-step)

1. **Simplify the biology (first pass).**
   Use SEIR + environment with standard forms:

   * Force of infection: $\lambda(t)=\beta_1 \frac{I}{N} + \beta_2 \frac{C}{K+C}$
   * Environment: $\dot C = \xi I - \delta C$ (contamination minus decay).
     Drop logistic growth for C.

2. **Daily clock; aggregate to weeks.**
   Integrate in days; compute **weekly incidence** as
   $\text{inc}_w = \int_{w-1}^{w} \sigma E(t)\,dt$.

3. **Observation model (very important).**

   * Cases: $\text{Cases}_w \sim \text{NegBin}(\rho\cdot \text{inc}_w,\; \theta_c)$ with reporting fraction $\rho\in[0.1,1]$.
   * Deaths: $\text{Deaths}_w \sim \text{NegBin}(\text{CFR}\cdot \rho_d\cdot \text{inc}_{w-\tau},\; \theta_d)$ with delay $\tau\in[5,10]$ days (rounded to weeks) and CFR $\in[0.005,0.05]$ for cholera outbreaks.

4. **Treat initial states as parameters.**
   Estimate $I_0, E_0, C_0$ (small, e.g. 1–50), not derived from week 1 cases. Set $S_0=N-I_0-E_0$ with **N equal to the affected catchment** (city/district), not a generic 1e6.

5. **Use constrained MLE (or Bayesian).**
   Maximize the joint NB log-likelihood for cases+deaths with sensible bounds/priors.

# Biologically justified starting values & bounds (per **day**)

Use these to avoid the optimizer drifting into nonsense:

* Transmission from people: $\beta_1 \in [0.1, 3]\!/ \text{day}$ (per-capita contact × transmission).
* Environmental transmission strength: $\beta_2 \in [0.0, 3]\!/ \text{day}$.
* Incubation → infectious: $\sigma \in [0.5, 2]$ (0.5–2 days to symptoms).
* Recovery: $\gamma \in [0.2, 0.7]$ (infectious 1.5–5 days).
* Disease death via CFR (not μ·I): $\text{CFR} \in [0.003, 0.03]$ (0.3–3%).
* Env. contamination & decay: $\xi \in [0.01, 1]$, $\delta \in [0.1, 1.5]$ (half-life \~0.5–7 days).
* Env half-sat: $K \in [10, 10^5]$ (scales C units).
* Reporting: $\rho \in [0.2, 0.9]$, $\rho_d \in [0.5, 1]$ (deaths often better ascertained).
* Delay: $\tau \in [5, 10]$ days.
* Initial states: $I_0, E_0 \in [1, 50]$, $C_0 \in [0, 100]$.
* Population $N$: set to the local population at risk (e.g., 50k–500k depending on your area); if unknown, add a scaling parameter and fit it with a penalty.

# Drop-in coding pattern (sketch)

* Integrate with `solve_ivp` on `np.arange(0, 7*(n_weeks)+1)` days.
* Sum `sigma*E(t)` over each 7-day block for incidence.
* Build a **joint negative-binomial** log-likelihood for (cases, deaths) with a lagged incidence for deaths.
* Optimize with `scipy.optimize.minimize` (method `"L-BFGS-B"`) under bounds, or use PyMC for priors/posteriors.

If you want, I can hand you a ready-to-run Python script that implements exactly this (daily ODE → weekly aggregation, joint NB likelihood, lagged deaths, bounded MLE), plus diagnostics (incidence vs. observed, posterior predictive checks, parameter identifiability profiles).
