# src/fit_and_project.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import least_squares
from model import Params, ControlGains, simulate

# ==============================
# Weekly data (cases & deaths)
# ==============================
weeks = np.arange(1, 16, dtype=float)
cases_obs = np.array([64, 52, 101, 168, 175,
                      279, 393, 472, 324, 517,
                      460, 722, 1059, 1495, 1140], float)
deaths_obs = np.array([4, 2, 7, 14, 11,
                       12, 8, 7, 8, 13,
                       18, 18, 30, 32, 21], float)

# ========================================
# Initial conditions and parameters (weekly)
# ========================================
N_POP = 1_000_000.0
t_fit = weeks

# We will FIT I0 and C0 (prevalence & environment at start).
# E0 initialized as 0.5*I0; R0=0; A0=D0=0; S0=N - (E0+I0+R0)
def pack_params(theta):
    # theta = [beta1, beta2, sigma, gamma, mu, delta, eta, epsilon, kappa, I0, C0]
    return dict(
        beta1   = theta[0],
        beta2   = theta[1],
        sigma   = theta[2],
        gamma   = theta[3],
        mu      = theta[4],
        delta   = theta[5],
        eta     = theta[6],
        epsilon = theta[7],
        kappa   = theta[8],
        I0      = theta[9],
        C0      = theta[10],
    )

# Biologically grounded initials/bounds (WEEKLY units)
theta0 = np.array([
    0.7,    # beta1
    0.5,    # beta2
    4.0,    # sigma  (≈ 1/1.75 wk)
    1.5,    # gamma  (≈ 1/0.67 wk)
    0.02,   # mu     (1-5% weekly hazard typical; context-dependent)
    2.0,    # delta  (env half-life ~ 2-3 days)
    0.15,   # eta
    0.05,   # epsilon
    200.0,  # kappa
    50.0,   # I0     (initial infectious prevalence)
    5.0     # C0     (initial environmental level)
], float)

lower = np.array([
    0.0, 0.0, 1.4, 1.0, 0.001, 0.5, 0.0, 0.0,   50.0,  1.0,  0.0
], float)
upper = np.array([
    3.0, 3.0, 14.0, 3.0, 0.2,   7.0, 2.0,  0.5, 500.0, 1e4,  5e3
], float)

def simulate_from_theta(theta, u_vac=0.0, u_san=0.0, u_treat=0.0):
    pmap = pack_params(theta)
    pars = Params(
        N=N_POP,
        beta1=pmap["beta1"], beta2=pmap["beta2"], sigma=pmap["sigma"],
        gamma=pmap["gamma"], mu=pmap["mu"], delta=pmap["delta"],
        eta=pmap["eta"], epsilon=pmap["epsilon"], kappa=pmap["kappa"],
        u_vac=u_vac, u_san=u_san, u_treat=u_treat
    )
    I0 = max(1.0, pmap["I0"])
    E0 = 0.5 * I0
    R0 = 0.0
    C0 = max(0.0, pmap["C0"])
    S0 = pars.N - (E0 + I0 + R0)
    A0 = 0.0
    D0 = 0.0
    y0 = (S0, E0, I0, R0, C0, A0, D0)
    return simulate(t_fit, y0, pars)

# Joint residuals (cases & deaths), with mild weighting to deaths
def residuals(theta):
    out = simulate_from_theta(theta)
    # Align to observed weeks
    pred_cases = out["weekly_cases"]
    pred_deaths = out["weekly_deaths"]
    # weights: scale deaths residuals to similar magnitude
    w_cases = 1.0
    w_deaths = 5.0
    return np.concatenate([
        w_cases * (pred_cases - cases_obs),
        w_deaths * (pred_deaths - deaths_obs)
    ])

def main():
    sol = least_squares(
        residuals, theta0, bounds=(lower, upper),
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=20000
    )
    theta_hat = sol.x
    print("Fitted parameters:")
    names = ["beta1","beta2","sigma","gamma","mu","delta","eta","epsilon","kappa","I0","C0"]
    for n, v in zip(names, theta_hat):
        print(f"{n:>8s} = {v:.6g}")

    # 52-week projection (baseline)
    weeks_year = np.arange(t_fit[0], t_fit[0] + 52, dtype=float)
    # re-simulate on projection grid
    from model import Params as P
    pmap = pack_params(theta_hat)
    pars = P(
        N=N_POP, beta1=pmap["beta1"], beta2=pmap["beta2"], sigma=pmap["sigma"],
        gamma=pmap["gamma"], mu=pmap["mu"], delta=pmap["delta"],
        eta=pmap["eta"], epsilon=pmap["epsilon"], kappa=pmap["kappa"]
    )
    I0 = max(1.0, pmap["I0"])
    E0 = 0.5 * I0
    R0 = 0.0
    C0 = max(0.0, pmap["C0"])
    S0 = pars.N - (E0 + I0 + R0)
    y0 = (S0, E0, I0, R0, C0, 0.0, 0.0)

    out_fit = simulate(t_fit, y0, pars)
    # Projection uses same y0 but longer time vector
    from model import simulate as sim_proj
    out_proj = sim_proj(weeks_year, y0, pars)

    # Plots
    import os
    os.makedirs("../plots", exist_ok=True)

    # Fit vs data (cases)
    plt.figure(figsize=(10,5))
    plt.plot(t_fit, cases_obs, "o", label="Observed cases")
    plt.plot(out_fit["t"], out_fit["weekly_cases"], "-", label="Fitted cases")
    plt.xlabel("Week")
    plt.ylabel("Weekly cases")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("../plots/fit_cases.png", dpi=200)

    # Fit vs data (deaths)
    plt.figure(figsize=(10,5))
    plt.plot(t_fit, deaths_obs, "o", label="Observed deaths")
    plt.plot(out_fit["t"], out_fit["weekly_deaths"], "-", label="Fitted deaths")
    plt.xlabel("Week")
    plt.ylabel("Weekly deaths")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("../plots/fit_deaths.png", dpi=200)

    # 1-year projections (cases & deaths)
    plt.figure(figsize=(10,5))
    plt.plot(out_proj["t"], out_proj["weekly_cases"], "-", label="Projected cases (52w)")
    plt.xlabel("Week"); plt.ylabel("Weekly cases")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("../plots/proj_cases_52w.png", dpi=200)

    plt.figure(figsize=(10,5))
    plt.plot(out_proj["t"], out_proj["weekly_deaths"], "-", label="Projected deaths (52w)")
    plt.xlabel("Week"); plt.ylabel("Weekly deaths")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("../plots/proj_deaths_52w.png", dpi=200)

    print("Saved plots to ./plots")

if __name__ == "__main__":
    main()
