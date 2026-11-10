"""Exact 2-D, unscrambled base-2 Sobol net; no SciPy installation required.

Direction numbers: dimension 1 = 1/2,1/4,...; dimension 2 uses primitive
polynomial x+1 and initial m_1=1. Gray-code recurrence keeps all 2^m points.
No first-point skipping, thinning, random pseudo-Sobol substitution or rounding.
"""
from . import runtime
import numpy as np


def sobol_2d(power=10):
    if not isinstance(power, int) or not 1 <= power <= 30:
        raise ValueError("power must be an integer in [1,30]")
    bits = 32
    directions = np.empty((2, bits), dtype=np.uint64)
    directions[0] = [1 << (bits - 1 - j) for j in range(bits)]
    directions[1, 0] = 1 << (bits - 1)
    for j in range(1, bits):
        last = int(directions[1, j - 1])
        directions[1, j] = last ^ (last >> 1)
    points = np.zeros((1 << power, 2), dtype=float)
    state = np.zeros(2, dtype=np.uint64)
    for i in range(1, len(points)):
        rightmost = (i & -i).bit_length() - 1
        state ^= directions[:, rightmost]
        points[i] = state / float(1 << bits)
    return points


def candidate_worlds(cfg):
    unit = sobol_2d(cfg.sobol_power)
    lower = np.array([cfg.inertia_min, cfg.friction_min])
    upper = np.array([cfg.inertia_max, cfg.friction_max])
    return lower + unit * (upper - lower)


def check_truth(theta, candidates, cfg):
    theta = np.asarray(theta, dtype=float)
    lower = np.array([cfg.inertia_min, cfg.friction_min])
    upper = np.array([cfg.inertia_max, cfg.friction_max])
    if theta.shape != (2,) or not np.all(np.isfinite(theta)):
        raise ValueError("theta* must contain two finite parameters")
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError("theta* is outside the physical feasible domain")
    if np.any(np.all(np.isclose(candidates, theta, atol=1e-12, rtol=0), axis=1)):
        raise ValueError("theta* coincides with a Sobol hypothesis")
    return theta
