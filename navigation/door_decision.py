"""Explicit task utility and confidence state machine; no simulator dependency."""
from dataclasses import dataclass
import numpy as np


@dataclass
class DoorDecision:
    theta_current: float
    theta_clear: float
    remaining_angle: float
    L_short: float
    L_avoid: float
    G: float
    G_norm: float
    B: np.ndarray
    deltaJ: np.ndarray
    feasible: np.ndarray
    labels: np.ndarray
    p_push: float
    D: float
    mu_deltaJ: float
    sigma_deltaJ: float
    mean_B: float
    action: str
    reason: str


def evaluate(particles, weights, theta_current, L_short, L_avoid, cfg, *, post_probe=False,
             theta_clear=None,passage_available=True):
    particles, w = np.asarray(particles, float), np.asarray(weights, float)
    if (w.ndim != 1 or particles.shape != (len(w), 2) or not np.isfinite(particles).all()
            or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0):
        raise ValueError('invalid decision particles/weights')
    if (not np.isfinite([theta_current, L_short, L_avoid]).all()
            or L_avoid <= 0 or L_short < 0):
        raise ValueError('route lengths must be finite; L_avoid>0, L_short>=0')
    if not (0 <= cfg.AVOID_CONFIDENCE < .5 < cfg.PUSH_CONFIDENCE <= 1):
        raise ValueError('invalid confidence thresholds')
    if cfg.b_ref <= 0 or cfg.tau_push_max <= 0 or cfg.DECISION_LAMBDA < 0:
        raise ValueError('invalid effort/feasibility configuration')
    w = w / w.sum()
    target=cfg.THETA_CLEAR if theta_clear is None else float(theta_clear)
    if not np.isfinite(target) or target<0:raise ValueError('invalid clearance target')
    remain = max(0., target - theta_current)
    G = L_avoid - L_short
    gain = G / L_avoid
    B = particles[:, 1] * remain
    delta = gain - cfg.DECISION_LAMBDA * B / cfg.b_ref
    feasible = particles[:, 1] < cfg.tau_push_max
    if not passage_available:feasible[:]=False
    labels = (feasible & (delta > 0)).astype(int)
    p = float(np.clip(w @ labels, 0., 1.))
    D = 4*p*(1-p)
    mu = float(w @ delta)
    sigma = float(np.sqrt(w @ (delta-mu)**2))
    if not passage_available:
        action, reason = 'Avoid', 'own vehicle envelope has no target-door route within hinge limits'
    elif remain == 0:
        action, reason = 'Pass', 'door already clears vehicle; navigate through without Push'
    elif p >= cfg.PUSH_CONFIDENCE:
        action, reason = 'Push', 'weighted physically feasible Push support meets confidence'
    elif p <= cfg.AVOID_CONFIDENCE:
        action, reason = 'Avoid', 'weighted Push support below Avoid confidence'
    elif post_probe:
        action, reason = 'Avoid', 'single Probe used; residual uncertainty -> conservative Avoid'
    elif sigma <= cfg.DELTAJ_NEAR_ZERO_THRESHOLD and abs(mu) <= cfg.DELTAJ_NEAR_ZERO_THRESHOLD:
        action, reason = 'Avoid', 'Push/Avoid utilities concentrated near a tie; Probe has little value'
    elif sigma >= cfg.SIGMA_DELTAJ_PROBE_THRESHOLD and D >= cfg.D_PROBE_THRESHOLD:
        action, reason = 'Probe', 'material utility spread and Push/Avoid disagreement from particles'
    else:
        action, reason = 'Avoid', 'insufficient decision-relevant uncertainty for Probe'
    return DoorDecision(float(theta_current), target, remain, float(L_short), float(L_avoid), float(G), float(gain),
                        B, delta, feasible, labels, p, D, mu, sigma, float(w @ B), action, reason)


def from_geometry(particles, weights, angle, geometry, cfg, *, post_probe=False):
    clearance=geometry.get('clearance')
    target=(clearance['theta_clear'] if clearance['feasible'] else cfg.DOOR_ANGLE_MAX) if clearance else None
    return evaluate(particles, weights, angle, geometry['J_door_geo'], geometry['J_bypass_geo'],
                    cfg, post_probe=post_probe,theta_clear=target,
                    passage_available=clearance['feasible'] if clearance else True)
