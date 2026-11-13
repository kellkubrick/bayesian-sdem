import numpy as np
from sdem import SDEM, Phase
from mcmc import BayesFit



def show(title, model, th):
    p = model.predict(th)
    print(title)
    print('  Ms=%.4g  Mrs=%.4g А·м²/кг  Mrs/Ms=%.3f  Hcr=%.1f мТл  fd=%.3f  d_m=%.1f нм' %
          (p['Ms'], p['Mrs'], p['Mrs']/p['Ms'], p['Hcr_mT'], p['fd'], p['d_m']))
    print('  взаимодействие: sigma=%.2f мТл  x0=%.1f  k4=%.1f  zeta=%.3f  ряд применим: %s' %
          (p['sigma_mT'], p['x0'], p['k4'], p['zeta'], p['series_ok']))
    return p

B = (0, 10, 18, 50, 120, np.inf)                         # условные границы состояний (ASSUMPTION)
    # Пример 1: состав как в статье о глинах (магнетит + титаномагнетит, MD-часть второй фазы -> гематит), сильно разбавлен
clay = SDEM([Phase('1', Is=(400e3, 450e3), Ku=(4e3, 8e3), gamma=(0.02, 0.5), bounds_nm=B),
                 Phase('2', Is=120e3, Ku=(10e3, 12e3), gamma=0.3, bounds_nm=B, md_as=(2e3, 11e3, 0.5))], rho=1700.)
print('параметры:', clay.param_names)
truth = np.array([14.0, 0.45, -3.1, 0.80, 425e3, 6e3, 0.10, 11e3])
p = show('Пример 1 (разбавленный, два минерала):', clay, truth)

# Пример 2: одна фаза, концентрация 5 %: тот же код, блок взаимодействия уже заметен
dense = SDEM([Phase('m', Is=480e3, Ku=(3e3, 15e3), gamma=(0.2, 0.5), bounds_nm=(0, 12, 25, 60, 150, np.inf))], rho=2500.)
for lg in (-3.0, -1.3, -0.8):
    show('Пример 2 (одна фаза), cf=%.3g:' % 10**lg, dense, np.array([20., 0.35, lg, 5e3, 0.45]))

# Обратная задача на синтетических данных примера 1, погрешности 5 % (ASSUMPTION)
rng = np.random.default_rng(0)
data = {k: (p[k]*(1+0.05*rng.normal()), 0.05*p[k]) for k in ('Ms', 'Mrs', 'Hcr_mT', 'fd', 'd_m')}
fit = BayesFit(clay, data); fit.run(); fit.summary(truth)
