import numpy as np
from math import erf, sqrt, exp, pi

MU0 = 4e-7*pi
STATES = ('SP', 'bSP', 'SD', 'PSD', 'MD')


class Phase:
    """Одна магнитная фаза. Is, Ku, gamma: число (фиксировано) или (min, max) (подбирается).
    gamma = Irs/Is для частиц, несущих остаточную намагниченность (bSP, SD, PSD), 0 < gamma <= 0.5.
    bounds_nm: границы диаметров SP|bSP|SD|PSD|MD.  md_as=(Is, Ku, gamma): MD-часть фазы заменяется
    другим минералом (в статье о глинах: MD-часть 2-й фракции считается гематитом)."""
    def __init__(self, name, Is, Ku, gamma, bounds_nm, md_as=None):
        self.name, self.Is, self.Ku, self.gamma = name, Is, Ku, gamma
        self.bounds, self.md_as = np.array(bounds_nm, float), md_as


class SDEM:
    """Прямая модель. Параметры: d_med_nm, sigma, lg_cf, Delta_i (кроме последней фазы) и все Is/Ku/gamma, заданные диапазоном."""
    def __init__(self, phases, rho, d_med=(5., 40.), sigma=(0.1, 0.9), lg_cf=(-4.5, -0.5), g_shape=0.0):
        self.ph, self.rho, self.g = phases, rho, g_shape     # g_shape = 1 - 1.5 cos(theta_max), 0 для сферы
        self.param_names, lo, hi = ['d_med_nm', 'sigma', 'lg_cf'], [d_med[0], sigma[0], lg_cf[0]], [d_med[1], sigma[1], lg_cf[1]]
        for p in phases[:-1]:
            self.param_names.append('Delta_'+p.name); lo.append(0.02); hi.append(0.98)
        self._slots = []                                     # (фаза, атрибут, индекс в theta или None)
        for i, p in enumerate(phases):
            for a in ('Is', 'Ku', 'gamma'):
                v = getattr(p, a)
                if np.ndim(v) == 1:
                    self._slots.append((i, a, len(lo))); self.param_names.append(a+'_'+p.name); lo.append(v[0]); hi.append(v[1])
                else:
                    self._slots.append((i, a, None))
        self.low, self.high = np.array(lo, float), np.array(hi, float)

    # ---------- блок 1: размеры и состояния, формулы (2)-(6) ----------
    @staticmethod
    def _Phi(z):
        return 0.5*(1+erf(z/sqrt(2)))

    def states(self, d_med, s, B):
        """ln d ~ N(ln d_med, s). Возвращает объёмные доли состояний c_k и средние диаметры d_k."""
        mu = np.log(d_med); c = np.zeros(5); n = np.zeros(5)
        for k in range(5):
            up_n = 1.0 if np.isinf(B[k+1]) else self._Phi((np.log(B[k+1])-mu)/s)
            up_c = 1.0 if np.isinf(B[k+1]) else self._Phi((np.log(B[k+1])-mu-3*s*s)/s)
            lo_n = 0.0 if B[k] <= 0 else self._Phi((np.log(B[k])-mu)/s)
            lo_c = 0.0 if B[k] <= 0 else self._Phi((np.log(B[k])-mu-3*s*s)/s)
            n[k], c[k] = up_n-lo_n, up_c-lo_c
        m3 = exp(3*mu+4.5*s*s)
        d = np.where(n > 1e-12, (c*m3/np.maximum(n, 1e-12))**(1/3), 0.0)      # (4)
        return c, d

    # ---------- блок 4: поле взаимодействия, формулы (9)-(11) ----------
    def zeta(self, H0, c, Is):
        """Доля неперевёрнутых моментов в остаточном состоянии. H0 в Тл, c - объёмная концентрация носителей, Is в А/м.
        ASSUMPTION: в моменты входит концентрация и намагниченность носителей остаточной намагниченности."""
        sig = MU0*Is*sqrt(c/90.)                             # sigma, Тл (СИ-вид формулы MPM 2024)
        k4 = 0.0893/c - 3.0                                  # mu4/sigma^4 - 3
        z = 1.0
        for _ in range(60):                                  # <Hi> и k3 зависят от zeta -> итерации
            k3 = 0.1129/sqrt(c)*z
            x0 = (H0 + MU0*c*Is/3*self.g*z)/sig
            if x0 > 8:
                z_new = 1.0
            else:
                ph = exp(-x0*x0/2)/sqrt(2*pi)
                He = [1.0, x0]
                for m in range(2, 6):
                    He.append(x0*He[-1]-(m-1)*He[-2])
                Z0 = 0.5-0.5*erf(x0/sqrt(2))
                S = Z0 - (k3/6)*ph*He[2] + (k4/24)*ph*He[3] + (10*k3*k3/720)*ph*He[5]   # (10),(11)
                z_new = min(1.0, max(0.0, 1-2*S))            # (9)
            if abs(z_new-z) < 1e-6:
                z = z_new; break
            z = 0.5*(z+z_new)
        ok = (x0 > 4) or (abs(k4) < 3)                       # ряд применим, либо поправка заведомо ничтожна
        return z, dict(sigma_mT=1e3*sig, x0=x0, k4=k4, series_ok=ok)

    # ---------- полный прямой расчёт ----------
    def predict(self, theta):
        d_med, s, lg_cf = theta[:3]
        cf = 10.**lg_cf
        nph = len(self.ph)
        D = list(theta[3:3+nph-1]); D.append(1-sum(D))
        if D[-1] < 0:
            raise ValueError('sum Delta > 1')
        val = [dict(Is=p.Is, Ku=p.Ku, gamma=p.gamma) for p in self.ph]
        for i, a, j in self._slots:
            if j is not None:
                val[i][a] = theta[j]
        Is_eff = 0.0; fd = 0.0; rem = []; H0i = []; cvol = []; d_m3 = exp(3*np.log(d_med)+4.5*s*s)
        for p, v, Di in zip(self.ph, val, D):
            c, d = self.states(d_med, s, p.bounds)
            fd += Di*c[0]                                    # объёмная доля SP: сравнивается с fd
            cmd = c[4] if p.md_as else 0.0
            Is_eff += Di*(v['Is']*(1-cmd) + (p.md_as[0]*cmd if p.md_as else 0.0))          # (7)
            w = c[1]+c[2]+c[3]
            H3 = 2*v['Ku']/v['Is']                           # (16), сразу mu0*H в Тл
            H2 = H3*(1-(d[1]/max(d[2], 1e-9))**1.5); H4 = H3*d[2]/max(d[3], 1e-9)
            H0i.append((H2*c[1]+H3*c[2]+H4*c[3])/max(w, 1e-12))                             # (17)
            rem.append(Di*w*v['gamma']*v['Is']); cvol.append(Di*w)                         # вклад в Irs, (14)-(15)
            if p.md_as:
                H0i.append(2*p.md_as[1]/p.md_as[0]); rem.append(Di*cmd*p.md_as[2]*p.md_as[0]); cvol.append(Di*cmd)
        rem = np.array(rem); Irs = rem.sum(); delta = rem/Irs
        H0 = float(np.dot(delta, H0i))                       # (18): сравнивается с Hcr
        c_r = cf*sum(cvol)                                   # (12)
        z, info = self.zeta(H0, c_r, Is_eff)
        out = dict(Ms=cf*Is_eff/self.rho,                    # (8), прочитанная в прямую сторону
                   Mrs=cf*Irs*z/self.rho, Hcr_mT=1e3*H0, fd=fd, d_m=d_m3**(1/3),
                   cf=cf, zeta=z, delta=delta, Is_eff=Is_eff)
        out.update(info)
        return out
