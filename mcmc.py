class BayesFit:
    """MCMC поверх любой модели с интерфейсом: param_names, low, high, observables, predict(theta)."""
    def __init__(self, model, obs, err):
        self.m = model
        self.obs, self.err = np.asarray(obs, float), np.asarray(err, float)

    def log_prior(self, th):
        return 0.0 if np.all((th >= self.m.low) & (th <= self.m.high)) else -np.inf

    def log_like(self, th):
        try:
            p = self.m.predict(th)
        except Exception:
            return -np.inf
        pred = np.array([p[k] for k in self.m.observables])
        return -0.5*np.sum(((pred-self.obs)/self.err)**2)

    def log_prob(self, th):
        lp = self.log_prior(th)
        return -np.inf if not np.isfinite(lp) else lp + self.log_like(th)

    def run(self, nwalk=32, nstep=4000, burn=1500, seed=1):
        rng = np.random.default_rng(seed)
        lo, hi = self.m.low, self.m.high
        p0 = lo + (hi-lo)*rng.random((nwalk, len(lo)))
        self.sampler = emcee.EnsembleSampler(nwalk, len(lo), self.log_prob)
        self.sampler.run_mcmc(p0, nstep, progress=False)
        self.chain = self.sampler.get_chain(discard=burn, flat=True)
        return self.chain

    def summary(self, truth=None):
        print('%-10s %9s %9s %9s %9s   %s' % ('параметр', 'истина', '16%', 'медиана', '84%', 'ширина/ширина_априора'))
        for j, nm in enumerate(self.m.param_names):
            lo, md, hi = np.percentile(self.chain[:, j], [16, 50, 84])
            sc = 1e-3 if nm.startswith(('Ku', 'Is')) else 1
            t = '' if truth is None else '%9.3g' % (truth[j]*sc)
            print('%-10s %9s %9.3g %9.3g %9.3g   %.2f' % (nm, t, lo*sc, md*sc, hi*sc, (hi-lo)/(self.m.high[j]-self.m.low[j])))

    def corner(self, fname='sdem_mcmc_corner.png', truth=None):
        import corner, matplotlib; matplotlib.use('Agg')
        corner.corner(self.chain, labels=self.m.param_names, truths=truth, quantiles=[.16, .5, .84]).savefig(fname, dpi=110)
