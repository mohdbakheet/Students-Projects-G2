# src/model.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.integrate import solve_ivp

# =========================
# Weekly SEIRC-C (with C(t))
# =========================
# States: S, E, I, R, C, A, D
# A: cumulative new infections (incidence integral)  dA = sigma * E
# D: cumulative deaths                               dD = mu_eff * I
# Time unit: weeks

@dataclass
class Params:
    N: float
    beta1: float     # direct (short-cycle) transmission rate, per week
    beta2: float     # environmental (long-cycle) transmission rate, per week
    sigma: float     # incubation rate (E->I), per week
    gamma: float     # recovery rate, per week
    mu: float        # disease-induced mortality rate, per week
    delta: float     # environmental loss/clearance rate, per week
    eta: float       # shedding rate into environment, per week
    epsilon: float   # baseline environmental growth/retention, per week
    kappa: float     # environmental carrying capacity (for C)
    # Controls (constant in scenarios; set to 0 for baseline)
    u_vac: float = 0.0     # vaccination / susceptibility reduction [0,1]
    u_san: float = 0.0     # sanitation (environmental clearance) [0,1]
    u_treat: float = 0.0   # treatment (↑gamma, ↓mu) [0,1]

@dataclass
class ControlGains:
    k_v_to_beta: float = 1.0    # scales (1 - u_vac) on force of infection
    k_s_to_delta: float = 4.0   # δ_eff = δ * (1 + k_s*u_san)
    k_s_to_beta2: float = 0.5   # β2_eff = β2 * (1 - k * u_san)
    k_t_to_gamma: float = 3.0   # γ_eff = γ * (1 + k_t*u_treat)
    k_t_to_mu: float = 0.9      # μ_eff = μ * (1 - k_mu*u_treat)

def _effective_params(p: Params, k: ControlGains) -> Dict[str, float]:
    # Susceptibility reduction via vaccination (multiplicative on FOI)
    susc_scale = (1.0 - k.k_v_to_beta * np.clip(p.u_vac, 0, 1))

    # Sanitation: increase δ; optionally slightly reduce β2 (water contact)
    delta_eff = p.delta * (1.0 + k.k_s_to_delta * np.clip(p.u_san, 0, 1))
    beta2_eff = p.beta2 * (1.0 - k.k_s_to_beta2 * np.clip(p.u_san, 0, 1))

    # Treatment: increase γ and reduce μ
    gamma_eff = p.gamma * (1.0 + k.k_t_to_gamma * np.clip(p.u_treat, 0, 1))
    mu_eff    = p.mu    * (1.0 - k.k_t_to_mu   * np.clip(p.u_treat, 0, 1))

    return dict(susc_scale=susc_scale, delta_eff=delta_eff,
                beta2_eff=beta2_eff, gamma_eff=gamma_eff, mu_eff=mu_eff)

def rhs(t: float, y, p: Params, gains: ControlGains) -> np.ndarray:
    S, E, I, R, C, A, D = y
    eff = _effective_params(p, gains)

    # Force of infection (weekly); saturating environmental dose
    lambda_dir = p.beta1 * I / p.N
    lambda_env = eff["beta2_eff"] * C / (C + 1.0)
    lambda_tot = eff["susc_scale"] * (lambda_dir + lambda_env)

    dS = - lambda_tot * S
    dE =   lambda_tot * S - p.sigma * E
    dI =   p.sigma * E - eff["gamma_eff"] * I - eff["mu_eff"] * I
    dR =   eff["gamma_eff"] * I
    # Logistic-like environmental bacteria dynamics
    dC =   p.eta * I * (1.0 - C / p.kappa) + p.epsilon * C - eff["delta_eff"] * C

    # Cumulative flows (weekly incidence & deaths)
    dA = p.sigma * E
    dD = eff["mu_eff"] * I

    return np.array([dS, dE, dI, dR, dC, dA, dD], dtype=float)

def simulate(
    t_weeks: np.ndarray,
    y0: Tuple[float, float, float, float, float, float, float],
    params: Params,
    gains: ControlGains = ControlGains(),
    rtol: float = 1e-6,
    atol: float = 1e-8,
    method: str = "LSODA"
):
    t_span = (float(t_weeks[0]), float(t_weeks[-1]))
    sol = solve_ivp(
        lambda t, y: rhs(t, y, params, gains),
        t_span, np.array(y0, float),
        t_eval=t_weeks, rtol=rtol, atol=atol, method=method
    )
    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)

    S, E, I, R, C, A, D = sol.y
    # Weekly incidence & deaths by differencing cumulative integrals
    # (First entry is week-aligned; prepend 0 to diff over grid)
    A_diff = np.diff(A, prepend=A[0])
    D_diff = np.diff(D, prepend=D[0])

    outputs = dict(
        t=sol.t, S=S, E=E, I=I, R=R, C=C,
        cum_inc=A, cum_deaths=D,
        weekly_cases=A_diff, weekly_deaths=D_diff
    )
    return outputs
