# Weekly Cholera SEIRC-C (SEIR + Environment) — Fitting, Projection, and Controls

يعيش السودان أوضاعاً صحية كارثية نتيجة لتفشّي وباء الكوليرا، مع دخول الحرب عامها الثالث ويواجه السودانيون واقعاً مزرياً؛ إذ بجانب الكوليرا، تنتشر الملاريا وحُمّى الضنك، في ظل نظام صحيّ مُتداع.
العديد من الأفراد ممن عادوا إلى منازلهم بعد سيطرة الجيش على العاصمة الخرطوم، يرغبون في المغادرة مجدداً خاصة بعد تفشي مرض الكوليرا في العاصمة وعدد من الولايات، إذ أعلنت وزارة الصحة السودانية، انتشار المرض في كل من (شمال كردفان، وسنار، والجزيرة، والنيل الأبيض ونهر النيل).
 
لذلك استخدام النماذج الرياضية في التحكم والحد من انتشار المرض يساعد كثيرا في التنبؤ بمسار انتشار المرض  ومعرفة كيف سيتطور عدد الإصابات بمرور الوقت وتقدير سرعة الانتشار حساب معدل الانتقال بين الأفراد و تقييم فعالية التدخلات( مثل الحجر الصحي، حملات التوعية، التطعيم أو إغلاق المدارس) وتوزيع الموارد الصحية(مستشفيات، أجهزة تنفس، أدوية) بطريقة أفضل بناءً على التوقعات واخيرا وضع خطط استباقية استعدادا لموجات جديدة من المرض وتحديد الفئات الأكثر عرضة لمعرفة أين يجب أن تركز الجهود الوقائية.



## Outlines
A clean, teaching-oriented implementation of a **weekly** SEIRC-C model for cholera that:
1) fits parameters to **weekly cases & deaths**,
2) produces **52-week projections**, and
3) explores **constant control scenarios** (vaccination, sanitation, treatment),
with a **skeleton** for future **optimal control**.

## Model
States: S (susceptible), E (exposed), I (infectious), R (recovered),  
C (environmental *V. cholerae*), A (cum. infections), D (cum. deaths).

Time unit = **weeks**.

Force of infection:
$$
\lambda(t) = (1-u_\text{vac})\left[\beta_1 \frac{I}{N} + \beta_2 \frac{C}{C+1}\right]
$$

Dynamics:
$$
\begin{aligned}
\dot S &= -\lambda S, \\
\dot E &= \lambda S - \sigma E, \\
\dot I &= \sigma E - (\gamma_\text{eff} + \mu_\text{eff}) I, \\
\dot R &= \gamma_\text{eff} I, \\
\dot C &= \eta I \left(1 - \frac{C}{\kappa}\right) + \epsilon C - \delta_\text{eff} C, \\
\dot A &= \sigma E, \qquad
\dot D = \mu_\text{eff} I.
\end{aligned}
$$

**Controls (constant in `scenarios.py`):**
- Vaccination / susceptibility reduction \(u_\text{vac}\in[0,1]\)
- Environmental sanitation \(u_\text{san}\in[0,1]\)
- Treatment \(u_\text{treat}\in[0,1]\)

**Control effects:**
- \( \text{susceptibility scale} = (1-u_\text{vac}) \)
- \( \delta_\text{eff} = \delta(1 + k_s u_\text{san}) \), and \( \beta_{2,\text{eff}} = \beta_2(1 - \tilde{k}_s u_\text{san}) \)
- \( \gamma_\text{eff} = \gamma(1 + k_t u_\text{treat}), \quad \mu_\text{eff} = \mu(1 - \tilde{k}_t u_\text{treat}) \)

Defaults: \(k_s=4,\ \tilde{k}_s=0.5,\ k_t=3,\ \tilde{k}_t=0.9\).

**Weekly incidence/deaths:** computed as differences of cumulative flows \(A(t)\) and \(D(t)\) on the weekly grid (avoids the common “\(\sigma E\) sampled at points” pitfall).

## Data
Prototype data are embedded in `fit_and_project.py` (15 weeks of cases & deaths). You can replace them or load from `data/weekly_timeseries.csv`.

## Usage
```bash
# 1) create and activate env (optional)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Fit and project (saves plots in ./plots)
python src/fit_and_project.py

# 3) Run constant-control scenarios (baseline vs interventions)
python src/scenarios.py

# 4) Inspect optimal-control skeleton (to be completed)
python src/opt_control_skeleton.py
