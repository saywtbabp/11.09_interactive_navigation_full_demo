"""Data bounded local mass/resistance density. No pre-contact parameter box."""
from dataclasses import dataclass
import runtime
import numpy as np


@dataclass(frozen=True)
class Action:
    name: str
    fraction: float
    hold: float
    coast: float


ACTIONS = (Action('gentle', .55, .8, 1.2), Action('short', 1., .65, 1.2),
           Action('guarded_coast', 1., 1.4, 1.2), Action('long_coast', 1., 1.4, 2.4),
           Action('ramp_coast', 1., 1.4, 1.2))


def clip(poly, normal, limit):
    result = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = a@normal-limit, b@normal-limit
        if da <= 1e-10:
            result.append(a)
        if (da < 0) != (db < 0):
            result.append(a+(b-a)*da/(da-db))
    return np.asarray(result).reshape(-1, 2)


class Belief:
    def __init__(self):
        self.windows = []
        self.precision = np.zeros((2, 2))
        self.linear = np.zeros(2)
        self.polygon = np.empty((0, 2))
        self.bounds = None
        self.first_moving_window = None
        self.status = 'unbounded_no_informative_acceleration'
        self.conflict = False

    def update(self, rows, damping=0., arm=1.):
        # rows are already local SI: t,s,v,F_parallel,total_contact_force.
        # Nonoverlapping windows; never reuse a prefix as independent evidence.
        for a in range(0, len(rows)-50, 50):
            z = rows[a:a+51]
            t, s, v, f, total = z.T
            dt = t[-1]-t[0]
            if np.min(v) <= .010 or np.max(abs(np.diff(v))) > .12:
                continue
            dv = v[-1]-v[0]
            impulse = float(np.sum(f[1:]*np.diff(t)))
            # Bounds include velocity error and neglected local viscous impulse.
            ev, ey = .025, .025 + .10*float(abs(s[-1]-s[0]))
            self.windows.append([float(dv), float(dt), impulse, ev, ey])
            x = np.array([dv, dt]); sigma = ey/2
            self.precision += np.outer(x, x)/sigma**2
            self.linear += x*impulse/sigma**2
            if self.first_moving_window is None:
                self.first_moving_window = [float(dv), float(dt), impulse]
        self.rebuild()

    def rebuild(self):
        positive = [z for z in self.windows if z[0]-z[3] > 0 and z[2]+z[4] > 0]
        if not positive:
            return
        # These finite bounds are consequences of measured impulse and dv>0.
        mmax = min((y+ey)/(dv-ev) for dv, t, y, ev, ey in positive)
        rmax = min((y+ey)/t for dv, t, y, ev, ey in positive)
        poly = np.array([[0., 0.], [mmax, 0.], [mmax, rmax], [0., rmax]])
        for dv, dt, y, ev, ey in self.windows:
            # [dv-ev,dv+ev] m + dt R intersects [y-ey,y+ey].
            poly = clip(poly, np.array([dv-ev, dt]), y+ey)
            if len(poly):
                poly = clip(poly, -np.array([dv+ev, dt]), -(y-ey))
            if len(poly) < 3:
                self.conflict = True
                self.status = 'inconsistent_windows_conservative_bypass'
                self.polygon = np.empty((0, 2)); self.bounds = None
                return
        self.polygon = poly
        self.bounds = np.column_stack((poly.min(axis=0), poly.max(axis=0)))
        self.status = 'data_bounded'

    def log_density(self, points):
        return points@self.linear-.5*np.sum((points@self.precision)*points, axis=1)

    def quadrature(self, order=64, cuts=()):
        if self.bounds is None or self.conflict:
            return np.empty((0, 2)), np.array([])
        # Triangle quadrature over the actual polygon, never a filled rectangle.
        nodes, weights = np.polynomial.legendre.leggauss(order)
        u = (nodes+1)/2; w = weights/2
        points, volumes = [], []
        a = self.polygon[0]
        for b, c in zip(self.polygon[1:-1], self.polygon[2:]):
            cross = abs(float(np.linalg.det(np.stack((b-a, c-a)))))
            if cross < 1e-14:
                continue
            p = a+u[:, None, None]*(b-a)+(1-u[:, None, None])*u[None, :, None]*(c-a)
            vol = cross*(1-u[:, None])*w[:, None]*w[None, :]
            points.append(p.reshape(-1, 2)); volumes.append(vol.ravel())
        if not points:
            return np.empty((0, 2)), np.array([])
        p, volume = np.vstack(points), np.concatenate(volumes)
        log = self.log_density(p); weight = np.exp(log-log.max())*volume
        return p, weight/weight.sum()

    def probability(self, limit):
        if self.bounds is None or self.conflict:
            return None, False
        # Integrate numerator after clipping exactly at the horizontal decision line.
        p, w = self.quadrature(20)
        peak = self.log_density(p).max()
        def integral(poly, n):
            if len(poly) < 3:
                return 0.
            old = self.polygon; self.polygon = poly
            nodes, weights = np.polynomial.legendre.leggauss(n)
            u=(nodes+1)/2; weights=weights/2; value=0.
            a=poly[0]
            for b,c in zip(poly[1:-1],poly[2:]):
                area=abs(float(np.linalg.det(np.stack((b-a,c-a)))))
                pts=a+u[:,None,None]*(b-a)+(1-u[:,None,None])*u[None,:,None]*(c-a)
                vol=area*(1-u[:,None])*weights[:,None]*weights[None,:]
                value+=float(np.sum(np.exp(self.log_density(pts.reshape(-1,2))-peak)*vol.ravel()))
            self.polygon=old
            return value
        good=clip(self.polygon,np.array([0.,1.]),limit)
        prev=None
        for n in (12,24,48,96):
            denom=integral(self.polygon,n)
            prob=integral(good,n)/denom if denom>0 else None
            if prob is not None and prev is not None and abs(prob-prev)<.002:
                return float(np.clip(prob,0,1)), True
            prev=prob
        return prob, False

    def snapshot(self):
        return dict(bounds=None if self.bounds is None else self.bounds.tolist(),
                    constraints=[list(w) for w in self.windows],
                    polygon=self.polygon.tolist(), precision=self.precision.tolist(),
                    linear=self.linear.tolist(), windows=len(self.windows), status=self.status,
                    first_moving_window=self.first_moving_window, stall=0.)
