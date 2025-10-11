# src/opt_control_skeleton.py
"""
Minimal skeleton for an optimal control problem (Pontryagin) on the weekly SEIRC-C model.

Goal: Choose time-varying controls u_vac(t), u_san(t), u_treat(t) ∈ [0,1]
to minimize
    J = ∫_0^T [ B_I * I(t) + B_C * C(t) + (A1/2) u_vac(t)^2 + (A2/2) u_san(t)^2 + (A3/2) u_treat(t)^2 ] dt
subject to the SEIRC-C dynamics with control-modulated parameters (see model.py).

This file provides:
  - a parameter block,
  - a grid/time discretization,
  - forward simulation given a *profile* of controls,
  - and a placeholder loop for a forward–backward sweep (FBS) solver.
"""

import numpy as np
from dataclasses import dataclass
from model import Params, ControlGains, simulate

@dataclass
class OCWeights:
    B_I: float = 1.0
    B_C: float = 0.1
    A1: float = 0.5   # cost weight for u_vac^2
    A2: float = 0.5   # cost weight for u_san^2
    A3: float = 0.5   # cost weight for u_treat^2

def forward_sim_with_controls(t_grid, y0, pars_base, u_vac, u_san, u_treat):
    """
    Piecewise-constant controls on the weekly grid.
    For simplicity here: approximate by running simulate once with time-averaged controls.
    Replace with a 'simulate_piecewise' that stitches segments if you need full fidelity.
    """
    pars = Params(**{**pars_base.__dict__,
                     "u_vac": float(np.clip(np.mean(u_vac), 0, 1)),
                     "u_san": float(np.clip(np.mean(u_san), 0, 1)),
                     "u_treat": float(np.clip(np.mean(u_treat), 0, 1))})
    return simulate(t_grid, y0, pars)

def compute_cost(out, u_vac, u_san, u_treat, w: OCWeights, dt=1.0):
    I = out["I"]; C = out["C"]
    # Quadrature on grid (simple Riemann)
    L_state = w.B_I * I + w.B_C * C
    L_ctrl  = 0.5 * (w.A1 * u_vac**2 + w.A2 * u_san**2 + w.A3 * u_treat**2)
    # If controls are constant over intervals, sum them
    J = np.sum(L_state) * dt + np.sum(L_ctrl) * dt
    return J

def forward_backward_sweep_placeholder(pars_base, y0, T=52, steps=52, w=OCWeights()):
    """
    Placeholder FBS:
      1) Initialize controls (zeros)
      2) Forward simulate
      3) Backward integrate co-states (NOT IMPLEMENTED HERE)
      4) Update controls from optimality condition (projection to [0,1])
      5) Iterate until convergence

    Replace this placeholder with a proper implementation when ready.
    """
    t_grid = np.linspace(1.0, 1.0 + T, steps)
    u_v = np.zeros(steps)  # initial guess
    u_s = np.zeros(steps)
    u_t = np.zeros(steps)

    out = forward_sim_with_controls(t_grid, y0, pars_base, u_v, u_s, u_t)
    J0 = compute_cost(out, u_v, u_s, u_t, w)
    print(f"[FBS] Initial cost (all controls 0): J = {J0:.3f}")

    # --- TODO: implement adjoint system and control update rules ---
    print("[FBS] TODO: Implement adjoint equations and gradient-based updates.")
    return dict(J=J0, controls=dict(u_v=u_v, u_s=u_s, u_t=u_t))
