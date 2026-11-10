"""Contact evidence from own IMU, attitude and motor thrust, with bias/noise.

The estimator never receives exact contacts, environmental geometry or labels.
"""
import numpy as np


class ForceObserver:
    def __init__(self, cfg):
        self.cfg = cfg; self.time = -1.; self.force = np.zeros(3)
        self.bias = np.zeros(3); self.samples = []; self.noise = .03
        self.valid = False; self.trace = []

    def update(self, time, specific_force, rotation, mass, thrust, calibrate=False):
        if time <= self.time or time-self.time < .0099: return self.force.copy()
        dt = .01 if self.time < 0 else time-self.time
        self.time = float(time)
        raw = mass*(rotation@np.asarray(specific_force))-float(thrust)*rotation[:, 2]
        self.valid = bool(np.isfinite(raw).all() and dt < .12)
        if not self.valid: return self.force.copy()
        self.force += dt/(self.cfg.force_filter_time+dt)*(raw-self.force)
        if calibrate:
            self.samples.append(self.force.copy())
            self.samples = self.samples[-150:]
            self.bias = np.mean(self.samples, axis=0)
            self.noise = max(.015, float(np.std(np.asarray(self.samples)[:, :2], axis=0).max()))
        return self.force-self.bias

    def normal_force(self, normal):
        return max(0., -float((self.force-self.bias)[:2]@np.asarray(normal)))

    @property
    def threshold(self):
        return max(self.cfg.verify_force_floor, self.cfg.verify_force_sigma*self.noise)

    def record(self):
        return dict(time_s=self.time, external_force_N=(self.force-self.bias).tolist(), valid=self.valid,
                    bias_N=self.bias.tolist(), noise_N=self.noise, threshold_N=self.threshold,
                    calibration_samples=len(self.samples), source='onboard IMU + AHRS + own motor thrust; low-pass estimate')
