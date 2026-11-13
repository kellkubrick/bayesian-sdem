import numpy as np



class BayesFit:
    """MCMC для любой модели с param_names, low, high, predict(theta).
    data = {'имя величины': (значение, погрешность)}: в сравнение входят только те измерения, которые есть."""
    def __init__(self, model, data):
        self.m, self.keys = model, list(data)
        self.obs = np.array([data[k][0] for k in self.keys]); self.err = np.array([data[k][1] for k in self.keys])

    def log_prob(self, th):
        if not np.all((th >= self.m.low) & (th <= self.m.high)):
            return -np.inf
        try:
            p = self.m.predict(th)
        except Exception:
            return -np.inf
        pred = np.array([p[k] for k in self.keys])
        return -0.5*np.sum(((pred-self.obs)/self.err)**2)

    def run(self, nwalk=48, nstep=6000, burn=2000, seed=1):
        import emcee
        rng = np.random.default_rng(seed); lo, hi = self.m.low, self.m.high
        p0 = lo+(hi-lo)*rng.random((nwalk, len(lo)))
        sm = emcee.EnsembleSampler(nwalk, len(lo), self.log_prob)
        sm.run_mcmc(p0, nstep, progress=False)
        self.chain = sm.get_chain(discard=burn, flat=True)
        return self.chain

    def summary(self, truth=None):
        print('%-16s %10s %10s %10s %10s  %s' % ('параметр', 'истина', '16%', 'медиана', '84%', 'ширина/априор'))
        for j, nm in enumerate(self.m.param_names):
            a, b, c = np.percentile(self.chain[:, j], [16, 50, 84])
            t = '' if truth is None else '%10.4g' % truth[j]
            print('%-16s %10s %10.4g %10.4g %10.4g  %.2f' % (nm, t, a, b, c, (c-a)/(self.m.high[j]-self.m.low[j])))


def show(title, model, th):
    p = model.predict(th)
    print(title)
    print('  Ms=%.4g  Mrs=%.4g А·м²/кг  Mrs/Ms=%.3f  Hcr=%.1f мТл  fd=%.3f  d_m=%.1f нм' %
          (p['Ms'], p['Mrs'], p['Mrs']/p['Ms'], p['Hcr_mT'], p['fd'], p['d_m']))
    print('  взаимодействие: sigma=%.2f мТл  x0=%.1f  k4=%.1f  zeta=%.3f  ряд применим: %s' %
          (p['sigma_mT'], p['x0'], p['k4'], p['zeta'], p['series_ok']))
    return p
