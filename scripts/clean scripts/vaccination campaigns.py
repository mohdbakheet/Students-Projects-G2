import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.integrate import solve_ivp

# =========================
# SEIRC-C Model (with vaccination campaigns)
# =========================
kappa = 5000

@dataclass
class Params:
    N: float
    beta1: float
    beta2: float
    sigma: float
    gamma: float
    mu: float
    delta: float
    eta: float
    epsilon: float
    kappa: float
    u_vac: float = 0.0
    u_san: float = 0.0
    u_treat: float = 0.0

@dataclass
class ControlGains:
    k_s_to_delta: float = 9.0
    k_s_to_beta2: float = 0.98
    k_s_to_eta: float = 0.6
    k_s_to_eps: float = 0.8
    k_v_to_beta: float = 1.0
    k_t_to_gamma: float = 3.0
    k_t_to_mu: float = 0.9

def _effective_params(p: Params, k: ControlGains):
    u_v  = float(np.clip(p.u_vac,  0, 1))
    u_s  = float(np.clip(p.u_san,  0, 1))
    u_t  = float(np.clip(p.u_treat,0, 1))

    susc_scale = (1.0 - k.k_v_to_beta * u_v)
    delta_eff = p.delta * (1.0 + k.k_s_to_delta * u_s)
    beta2_eff = p.beta2 * max(0.0, 1.0 - k.k_s_to_beta2 * u_s)
    eta_eff   = p.eta   * (1.0 - k.k_s_to_eta   * u_s)
    eps_eff   = p.epsilon * (1.0 - k.k_s_to_eps * u_s)
    gamma_eff = p.gamma * (1.0 + k.k_t_to_gamma * u_t)
    mu_eff    = p.mu    * (1.0 - k.k_t_to_mu    * u_t)

    return dict(
        susc_scale=susc_scale, delta_eff=delta_eff, beta2_eff=beta2_eff,
        eta_eff=eta_eff, eps_eff=eps_eff, gamma_eff=gamma_eff, mu_eff=mu_eff
    )

def rhs(t, y, p: Params, gains: ControlGains):
    S, E, I, R, C, A, D = y
    eff = _effective_params(p, gains)

    lambda_dir = p.beta1 * I / p.N
    lambda_env = eff["beta2_eff"] * C / (C + kappa)
    lambda_tot = eff["susc_scale"] * (lambda_dir + lambda_env)

    dS = - lambda_tot * S
    dE =   lambda_tot * S - p.sigma * E
    dI =   p.sigma * E - eff["gamma_eff"] * I - eff["mu_eff"] * I
    dR =   eff["gamma_eff"] * I
    dC =   eff["eta_eff"] * I * (1.0 - C / p.kappa) + eff["eps_eff"] * C - eff["delta_eff"] * C
    dA = p.sigma * E
    dD = eff["mu_eff"] * I

    return np.array([dS, dE, dI, dR, dC, dA, dD], float)

# -------------------------
# Vaccination campaign pulses
# -------------------------
def build_pulse_campaign(t_weeks, interval=26, duration_weeks=4, intensity=0.6):
    u_vac_t = np.zeros_like(t_weeks)
    for i, t in enumerate(t_weeks):
        phase = (t % interval)
        if phase < duration_weeks:
            u_vac_t[i] = intensity
    return u_vac_t

# -------------------------
# Simulation with time-varying vaccination
# -------------------------
def simulate_timevarying(t_weeks, y0, params, gains, u_vac_t):
    y = np.zeros((len(t_weeks), len(y0)))
    y[0, :] = y0
    for i in range(1, len(t_weeks)):
        t_span = (t_weeks[i-1], t_weeks[i])
        p_now = Params(**{**params.__dict__, "u_vac": u_vac_t[i]})
        sol = solve_ivp(lambda t, y: rhs(t, y, p_now, gains),
                        t_span, y[i-1, :], method="LSODA", rtol=1e-6, atol=1e-8)
        y[i, :] = sol.y[:, -1]
    return y

# -------------------------
# Plot comparison
# -------------------------
def plot_comparison(t, results_no_vac, results_with_vac):
    plt.figure(figsize=(10,6))
    plt.plot(t, results_no_vac[:,0], 'b--', label='S بدون تطعيم')
    plt.plot(t, results_with_vac[:,0], 'b-', label='S مع تطعيم')
    plt.plot(t, results_no_vac[:,2], 'r--', label='I بدون تطعيم')
    plt.plot(t, results_with_vac[:,2], 'r-', label='I مع تطعيم')
    plt.plot(t, results_no_vac[:,3], 'g--', label='R بدون تطعيم')
    plt.plot(t, results_with_vac[:,3], 'g-', label='R مع تطعيم')
    plt.title("تأثير حملات التطعيم على S, I, R خلال 3 سنوات")
    plt.xlabel("الأسابيع")
    plt.ylabel("عدد الأفراد")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    t_weeks = np.arange(0, 156)  # 3 سنوات
    params = Params(
        N=1_000_000, beta1=0.7, beta2=0.2,
        sigma=1/2, gamma=1/3, mu=0.001,
        delta=0.05, eta=0.4, epsilon=0.05, kappa=5000
    )
    gains = ControlGains()
    y0 = (999000, 100, 50, 0, 10, 0, 0)

    # بدون تطعيم
    u_vac_none = np.zeros_like(t_weeks)
    res_no_vac = simulate_timevarying(t_weeks, y0, params, gains, u_vac_none)

    # مع حملات تطعيم
    u_vac_campaign = build_pulse_campaign(t_weeks, interval=26, duration_weeks=4, intensity=0.6)
    res_with_vac = simulate_timevarying(t_weeks, y0, params, gains, u_vac_campaign)

    # رسم المقارنة
    plot_comparison(t_weeks, res_no_vac, res_with_vac)
