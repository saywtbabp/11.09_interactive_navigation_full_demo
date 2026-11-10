"""NumPy-only door particles. No MuJoCo, vehicle, route or truth dependency."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ProbeTrace:
    # torque[k] is the zero-order-held input over [time[k], time[k+1]).
    # The last torque is repeated for plotting; it has no integration interval.
    time: np.ndarray
    torque: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray

    def validate(self):
        arrays = [np.asarray(getattr(self, k), dtype=float) for k in
                  ('time', 'torque', 'theta', 'theta_dot')]
        if arrays[0].ndim != 1 or len(arrays[0]) < 2:
            raise ValueError('ProbeTrace requires at least two samples')
        if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError('invalid/nonfinite ProbeTrace')
        if np.any(np.diff(arrays[0]) <= 0):
            raise ValueError('Probe times must strictly increase')
        return arrays


def rollout(I, tau_f, time, tau_input, theta0=0., omega0=0., *,
            omega_stick_eps=1e-5, max_step=.002, damping=0., angle_limits=None):
    """Vectorized semi-implicit Coulomb integration with a proximal stop step.

    Soft threshold of the free velocity implements static friction and stops
    exactly at zero instead of oscillating across zero under dry friction.
    Substeps bound integration error for irregular/downsampled input traces.
    Optional *known* viscous damping and public joint stops match the scene;
    neither introduces an unknown particle dimension nor reads true parameters.
    """
    I, f = np.asarray(I, float), np.asarray(tau_f, float)
    t, u = np.asarray(time, float), np.asarray(tau_input, float)
    if (I.ndim != 1 or not len(I) or f.shape != I.shape or np.any(I <= 0)
            or np.any(f < 0) or not np.isfinite(I).all() or not np.isfinite(f).all()):
        raise ValueError('I/tau_f must be finite particle arrays, I>0, tau_f>=0')
    if (t.ndim != 1 or len(t) < 2 or u.shape != t.shape or not np.isfinite(t).all()
            or not np.isfinite(u).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('invalid time/input arrays')
    if max_step <= 0 or omega_stick_eps < 0 or damping < 0:
        raise ValueError('invalid integrator settings')
    if not np.isfinite([theta0, omega0, max_step, omega_stick_eps, damping]).all():
        raise ValueError('nonfinite integrator settings')
    if angle_limits is not None and not angle_limits[0] < angle_limits[1]:
        raise ValueError('invalid angle limits')
    q = np.full(len(I), theta0, dtype=float)
    v = np.full(len(I), omega0, dtype=float)
    theta, omega = np.empty((len(I), len(t))), np.empty((len(I), len(t)))
    theta[:, 0], omega[:, 0] = q, v
    for k, interval in enumerate(np.diff(t)):
        count = max(1, int(np.ceil(interval / max_step - 1e-10)))
        dt = interval / count
        for _ in range(count):
            sticking = (np.abs(v) < omega_stick_eps) & (abs(u[k]) <= f)
            v = np.where(sticking, 0., v)
            free = v + u[k] * dt / I
            v = np.sign(free) * np.maximum(0., np.abs(free) - f * dt / I)
            v /= 1. + damping * dt / I
            q = q + v * dt
            if angle_limits is not None:
                lo, hi = angle_limits
                v = np.where(((q < lo) & (v < 0)) | ((q > hi) & (v > 0)), 0., v)
                q = np.clip(q, lo, hi)
        theta[:, k+1], omega[:, k+1] = q, v
    if not np.isfinite(theta).all() or not np.isfinite(omega).all():
        raise ValueError('nonfinite rollout')
    return theta, omega


def update_belief(weights, particles, trace, cfg):
    t, torque, measured_q, measured_v = trace.validate()
    particles, w = np.asarray(particles, float), np.asarray(weights, float)
    if (particles.shape != (len(w), 2) or w.ndim != 1 or not np.isfinite(w).all()
            or np.any(w < 0) or w.sum() <= 0):
        raise ValueError('invalid prior/particles')
    if cfg.SIGMA_THETA <= 0 or cfg.SIGMA_OMEGA <= 0 or cfg.OMEGA_ERROR_WEIGHT < 0:
        raise ValueError('invalid likelihood scales')
    q, v = rollout(particles[:, 0], particles[:, 1], t, torque, measured_q[0], measured_v[0],
                   omega_stick_eps=cfg.OMEGA_STICK_EPS, max_step=cfg.DOOR_INTEGRATION_MAX_STEP,
                   damping=cfg.hinge_damping,
                   angle_limits=(cfg.DOOR_ANGLE_MIN, cfg.DOOR_ANGLE_MAX))
    error = np.mean(((q-measured_q)/cfg.SIGMA_THETA)**2, axis=1)
    if cfg.OMEGA_ERROR_WEIGHT:
        error += cfg.OMEGA_ERROR_WEIGHT * np.mean(((v-measured_v)/cfg.SIGMA_OMEGA)**2, axis=1)
    if not np.isfinite(error).all():
        raise ValueError('nonfinite trajectory error')
    log_weight = np.full(len(w), -np.inf)
    positive = w > 0
    log_weight[positive] = np.log(w[positive]/w.sum()) - .5*error[positive]
    log_weight -= np.max(log_weight)
    posterior = np.exp(log_weight)
    posterior /= posterior.sum()
    return posterior, error, q, v


def belief_summary(particles, weights, top_n=10):
    w = np.asarray(weights, float)
    w = w / w.sum()
    mean = w @ particles
    return dict(mean_I=float(mean[0]), mean_tau_f=float(mean[1]),
                ESS=float(1./np.sum(w*w)),
                top_candidates=[dict(index=int(i), I=float(particles[i, 0]),
                                     tau_f=float(particles[i, 1]), weight=float(w[i]))
                                for i in np.argsort(-w, kind='stable')[:top_n]])
