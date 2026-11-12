import numpy as np
from sdem import SDEM
from mcmc import BayesFit



model = SDEM(Ms=0.20, rho=1700.)
truth = np.array([14.0, 0.45, 0.80, 6e3, 11e3, 425e3])    # синтетическая "истина"
p = model.predict(truth); obs = np.array([p[k] for k in model.observables])
obs_noisy = obs*(1+0.05*np.random.default_rng(0).normal(size=3))
fit = BayesFit(model, obs_noisy, 0.05*obs)                 # погрешность 5% (ASSUMPTION)
fit.run()
fit.summary(truth)
fit.corner(truth=truth)