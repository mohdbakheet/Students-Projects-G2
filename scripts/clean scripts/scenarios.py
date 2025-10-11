# src/scenarios.py
import numpy as np
import matplotlib.pyplot as plt
from fit_and_project import pack_params, theta0, lower, upper, least_squares, residuals
from model import Params, simulate

# After fitting (reuse least_squares & residuals from fit_and_project)
def fit_theta():
    sol = least_squares(residuals, theta0, bounds=(lower, upper),
                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=20000)
    return sol.x

def run_scenarios(theta_hat):
    p = pack_params(theta_hat)
    base = Params(N=1_000_000.0, beta1=p["beta1"], beta2=p["beta2"], sigma=p["sigma"],
                  gamma=p["gamma"], mu=p["mu"], delta=p["delta"],
                  eta=p["eta"], epsilon=p["epsilon"], kappa=p["kappa"])

    # Initial conditions based on fitted prevalence/environment
    I0 = max(1.0, p["I0"]); E0 = 0.5*I0; R0 = 0.0
    C0 = max(0.0, p["C0"]); S0 = base.N - (E0 + I0 + R0)
    y0 = (S0, E0, I0, R0, C0, 0.0, 0.0)

    weeks_year = np.arange(1.0, 53.0)

    scenarios = {
        "baseline (no control)" : dict(u_vac=0.0, u_san=0.0, u_treat=0.0),
        "vaccination 20%"       : dict(u_vac=0.2, u_san=0.0, u_treat=0.0),
        "sanitation 40%"        : dict(u_vac=0.0, u_san=0.4, u_treat=0.0),
        "treatment 50%"         : dict(u_vac=0.0, u_san=0.0, u_treat=0.5),
        "combo (20/30/30)"      : dict(u_vac=0.2, u_san=0.3, u_treat=0.3),
        "strong combo (40/50/50)":dict(u_vac=0.4, u_san=0.5, u_treat=0.5)
    }

    outs = {}
    for name, ctrl in scenarios.items():
        pars = Params(**{**base.__dict__, **ctrl})
        outs[name] = simulate(weeks_year, y0, pars)

    # Plot comparisons (weekly cases & deaths)
    import os
    os.makedirs("../plots", exist_ok=True)

    plt.figure(figsize=(11,6))
    for name, out in outs.items():
        plt.plot(out["t"], out["weekly_cases"], label=name)
    plt.xlabel("Week"); plt.ylabel("Weekly cases"); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("../plots/scenarios_cases.png", dpi=200)

    plt.figure(figsize=(11,6))
    for name, out in outs.items():
        plt.plot(out["t"], out["weekly_deaths"], label=name)
    plt.xlabel("Week"); plt.ylabel("Weekly deaths"); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("../plots/scenarios_deaths.png", dpi=200)

    print("Scenario plots saved in ./plots")

def main():
    theta_hat = fit_theta()
    run_scenarios(theta_hat)

if __name__ == "__main__":
    main()
