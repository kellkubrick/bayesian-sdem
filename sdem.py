import numpy as np
from math import erf, sqrt

initial_parameters = {
    "Ms": 1,
    "Mrs":2,
    "Hc":3,
    "Hcr":4,
    "fd":5,
    "m":6,
    "V":7
}


class SDEM:
    """Прямая модель: параметры -> предсказанные наблюдаемые. Только физика, никакого MCMC."""
    param_names = ['d_med_nm', 'sigma', 'Delta1', 'Ku1', 'Ku2', 'Is1']
    low  = np.array([5.0, 0.10, 0.30, 4e3, 10e3, 400e3])      # априорные диапазоны (равномерные)
    high = np.array([30., 0.80, 0.95, 8e3, 12e3, 450e3])

    def __init__(self, Ms, rho, bounds_nm=(0, 10, 18, 50, 120, np.inf), Is2=120e3, Is3=2e3, Ku3=11e3):
        self.Ms, self.rho = Ms, rho                           # Ms, А·м²/кг;  rho, кг/м³
        self.bounds = np.array(bounds_nm, float)              # SP | bSP | SD | PSD | MD, диаметры в нм
        self.Is2, self.Is3, self.Ku3 = Is2, Is3, Ku3

    @staticmethod
    def _Phi(z):
        return 0.5*(1+erf(z/sqrt(2)))

    def states(self, d_med_nm, sigma):
        """Логнормальный ln d ~ N(ln d_med, sigma): числовые доли n_k, объёмные c_k, средние диаметры d_k, d_m  ((2)-(6))."""
        mu, s, B = np.log(d_med_nm), sigma, self.bounds
        lb = np.log(np.where(np.isinf(B), 1e9, np.maximum(B, 1e-9)))
        n = np.zeros(5); v = np.zeros(5)
        for k in range(5):
            lo_ok = B[k] > 0
            n[k] = self._Phi((lb[k+1]-mu)/s) - (self._Phi((lb[k]-mu)/s) if lo_ok else 0.0)
            v[k] = self._Phi((lb[k+1]-mu-3*s**2)/s) - (self._Phi((lb[k]-mu-3*s**2)/s) if lo_ok else 0.0)
        v /= v.sum()
        m3 = np.exp(3*mu+4.5*s**2)                            # <d^3>
        d_k = np.where(n > 1e-12, (v*m3/np.maximum(n, 1e-12))**(1/3), 0.0)   # (4)
        return n, v, d_k, np.exp(mu+1.5*s**2)                 # d_m (5)

    def _H0_fraction(self, Ku, Isi, ci, d):
        H3 = 2*Ku/Isi                                         # mu0*H, Тл
        H2 = H3*(1-(d[1]/max(d[2], 1e-9))**1.5)               # (16)
        H4 = H3*d[2]/max(d[3], 1e-9)
        w = ci[1]+ci[2]+ci[3]
        return (H2*ci[1]+H3*ci[2]+H4*ci[3])/max(w, 1e-12), w  # (17)

    def predict(self, theta):
        d_med, sigma, D1, Ku1, Ku2, Is1 = theta
        D2 = 1-D1
        n, c, d, d_m = self.states(d_med, sigma)
        c1, c2 = D1*c, D2*c                                   # c_ik = Delta_i * c_k   (6)
        c25 = c2[4]                                           # MD часть 2-й фракции -> гематит
        Is = Is1*D1 + self.Is2*(D2-c25) + self.Is3*c25        # (7)
        cf = self.Ms*self.rho/Is                              # (8)
        H01, w1 = self._H0_fraction(Ku1, Is1, c1, d)
        H02, w2 = self._H0_fraction(Ku2, self.Is2, c2, d)
        H03 = 2*self.Ku3/self.Is3                             # H03 = H33
        r = np.array([w1*Is1, w2*self.Is2, c25*self.Is3]); delta = r/r.sum()   # ASSUMPTION вместо (14)-(15)
        H0 = delta[0]*H01 + delta[1]*H02 + delta[2]*H03       # (18)
        return dict(fd=c1[0]+c2[0], d_m=d_m, Hcr_mT=1e3*H0, cf=cf, delta=delta)

    observables = ('fd', 'd_m', 'Hcr_mT')
