import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# ------------------------------
# بيانات أسبوعية (استبدلها ببياناتك الحقيقية طولها 52)
cases_data = np.array([
    64, 52, 101, 168, 175,
    279, 393, 472, 324, 517,
    460, 722, 1059, 1495, 1140
], dtype=float)

deaths_data = np.array([
    4, 2, 7, 14, 11,
    12, 8, 7, 8, 13,
    18, 18, 30, 32, 21
], dtype=float)
# ------------------------------

weeks_fit = np.arange(1, len(cases_data) + 1)   # أسابيع البيانات
weeks_sim = np.arange(1, 53)                   # محاكاة سنة كاملة

# ثوابت
N = 1_000_000
k_capacity = 5000
season_amp = 0.30
dry_rate_base = 0.02
phi_max = 0.3
theta = 0.2
vacc_rate = 0.0
ve_default = 0.8
vacc_delay_default = 2

# ------------------------------
# نموذج SEICR
# ------------------------------
def seicr_model(t, y, N,
                beta1_base, beta2_base, sigma_base, gamma_base, mu,
                delta, eta,
                phi_max=0.0, theta=0.0, vacc_rate=0.0, ve=0.8, vacc_delay=2,
                dry_rate_base=0.0, season_amp=0.0, k_capacity=5000):
    week = t / 7.0
    season_factor = 1.0 + season_amp * np.sin(2 * np.pi * week / 52.0)
    dry_factor = (1 - dry_rate_base)  week

    beta1 = beta1_base * (1 - theta) * season_factor
    beta2 = beta2_base * (1 - phi_max) * dry_factor * season_factor
    sigma = sigma_base * season_factor
    gamma = gamma_base * (1 + 0.5 * theta) * season_factor

    S, E, I, R, C = y
    lam = (beta1 * I / N) + (beta2 * C / (k_capacity + C))

    dS = -lam * S - vacc_rate * S
    dE = lam * S - sigma * E
    dI = sigma * E - (gamma + mu) * I
    dR = gamma * I + vacc_rate * S * ve
    dC = eta * I * (1 - C / k_capacity) - delta * C

    return [dS, dE, dI, dR, dC]

# ------------------------------
# محاكاة النموذج
# ------------------------------
def simulate_seicr_ode(weeks_array, N, beta1, beta2, sigma, gamma, mu, delta, eta,
                       phi_max=0.0, theta=0.0, vacc_rate=0.0, ve=0.8, vacc_delay=2,
                       dry_rate_base=0.0, season_amp=0.0, k_capacity=5000,
                       I0=None):
    t_eval_days = ((weeks_array - weeks_array[0]) * 7.0)
    t_span = (t_eval_days[0], t_eval_days[-1])

    if I0 is None:
        I0 = max(1.0, cases_data[0])
    S0 = max(N - I0, 0)
    E0 = I0 * 0.5
    R0 = 0.0
    C0 = 1.0
    y0 = [S0, E0, I0, R0, C0]

    sol = solve_ivp(seicr_model, t_span, y0,
                    args=(N, beta1, beta2, sigma, gamma, mu, delta, eta,
                          phi_max, theta, vacc_rate, ve, vacc_delay,
                          dry_rate_base, season_amp, k_capacity),
                    t_eval=t_eval_days, rtol=1e-6, atol=1e-8, method='RK45')

    if not sol.success:
        raise RuntimeError("ODE solver failed")

    S_vals, E_vals, I_vals, R_vals, C_vals = sol.y

    weekly_cases = np.maximum(sigma * E_vals * 7.0, 0.0)
    weekly_deaths = np.maximum((mu * I_vals + eta * C_vals) * 7.0, 0.0)

    weeks_points = (t_eval_days / 7.0)
    season_factor_points = 1.0 + season_amp * np.sin(2 * np.pi * weeks_points / 52.0)
    dry_factor_points = (1 - dry_rate_base)  weeks_points

    beta1_ts = beta1 * (1 - theta) * season_factor_points
    beta2_ts = beta2 * (1 - phi_max) * dry_factor_points * season_factor_points
    sigma_ts = sigma * season_factor_points
    gamma_ts = gamma * (1 + 0.5 * theta) * season_factor_points

    return (weekly_cases, weekly_deaths, I_vals, R_vals, C_vals, weeks_points,
            beta1_ts, beta2_ts, sigma_ts, gamma_ts)

# ------------------------------
# دالة الملاءمة
# ------------------------------
def fit_parameters(weeks_obs, cases_obs, N,
                   p0=None, bounds=None,
                   phi_max=0.0, theta=0.0, vacc_rate=0.0,
                   dry_rate_base=0.0, season_amp=0.0, k_capacity=5000):
    if p0 is None:
        p0 = [0.5, 0.3, 1/5, 0.4, 0.02, 0.08, 0.1]
    if bounds is None:
        lower = [0, 0, 1/12, 0.05, 0.0001, 0.01, 0]
        upper = [3, 3, 1/2, 1.0, 0.2, 1.0, 2.0]
        bounds = (lower, upper)
      def fit_func_short(weeks_input, beta1, beta2, sigma, gamma, mu, delta, eta):
        try:
            weekly_cases, _, _, _, _, _, _, _, _, _ = simulate_seicr_ode(
                weeks_input, N, beta1, beta2, sigma, gamma, mu, delta, eta,
                phi_max=phi_max, theta=theta, vacc_rate=vacc_rate, ve=ve_default, vacc_delay=vacc_delay_default,
                dry_rate_base=dry_rate_base, season_amp=season_amp, k_capacity=k_capacity,
                I0=cases_data[0]
            )
        except RuntimeError:
            return np.full_like(weeks_input, np.nan, dtype=float)
        return weekly_cases.astype(float)

    params_opt, cov = curve_fit(fit_func_short, weeks_obs, cases_obs, p0=p0, bounds=bounds, maxfev=30000)
    return params_opt

# ------------------------------
# تقدير المعاملات
# ------------------------------
p0 = [0.7, 0.5, 1/5, 0.35, 0.03, 0.08, 0.15]
bounds = ([0,0,1/12,0.05,0.0001,0.01,0],[3,3,1/2,1.0,0.2,1.0,2.0])

params_opt = fit_parameters(weeks_fit, cases_data, N, p0=p0, bounds=bounds,
                            phi_max=phi_max, theta=theta, vacc_rate=vacc_rate,
                            dry_rate_base=dry_rate_base, season_amp=season_amp, k_capacity=k_capacity)

beta1_fit, beta2_fit, sigma_fit, gamma_fit, mu_fit, delta_fit, eta_fit = params_opt

print("\nFitted parameters:")
print(f"beta1 = {beta1_fit:.6f}")
print(f"beta2 = {beta2_fit:.6f}")
print(f"sigma = {sigma_fit:.6f}")
print(f"gamma = {gamma_fit:.6f}")
print(f"mu    = {mu_fit:.6f}")
print(f"delta = {delta_fit:.6f}")
print(f"eta   = {eta_fit:.6f}")

# ------------------------------
# محاكاة سنة كاملة
# ------------------------------
(weekly_cases_sim, weekly_deaths_sim, I_sim, R_sim, C_sim,
 weeks_points_sim, beta1_ts, beta2_ts, sigma_ts, gamma_ts) = simulate_seicr_ode(
    weeks_sim, N,
    beta1_fit, beta2_fit, sigma_fit, gamma_fit, mu_fit, delta_fit, eta_fit,
    phi_max=phi_max, theta=theta, vacc_rate=vacc_rate, ve=ve_default, vacc_delay=vacc_delay_default,
    dry_rate_base=dry_rate_base, season_amp=season_amp, k_capacity=k_capacity,
    I0=cases_data[0]
)

# ------------------------------
# رسومات المعاملات الزمنية
# ------------------------------
plt.figure(figsize=(10,6))
plt.plot(weeks_sim, beta1_ts, '-r', label='β1(t)')
plt.plot(weeks_sim, beta2_ts, '-b', label='β2(t)')
plt.plot(weeks_sim, sigma_ts, '-g', label='σ(t)')
plt.plot(weeks_sim, gamma_ts, '-m', label='γ(t)')
plt.xlabel("Week")
plt.ylabel("Parameter value")
plt.title("Time-varying parameters over 52 weeks")
plt.legend()
plt.grid(True)
plt.show()
