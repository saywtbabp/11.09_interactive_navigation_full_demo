"""Weights over physical hypotheses, never an Easy/Hard classifier."""
from . import runtime
import numpy as np


def normalized(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid hypothesis weights")
    return w / w.sum()


def consistency_update(weights, predictions, observation, cfg):
    w = normalized(weights)
    predictions, observation = np.asarray(predictions), np.asarray(observation)
    if predictions.shape != (len(w), 2) or observation.shape != (2,):
        raise ValueError("expected Nx2 predictions and a 2-D observation")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(observation)):
        raise ValueError("nonfinite probe observation")
    error = np.sum(((predictions - observation) / [cfg.force_scale, cfg.angle_scale]) ** 2, axis=1)
    log_score = np.full(len(w), -np.inf)
    positive = w > 0
    log_score[positive] = np.log(w[positive]) - cfg.alpha * error[positive]
    log_score -= np.max(log_score)  # stable even when every raw exp(-alpha*e) underflows
    posterior = normalized(np.exp(log_score))
    return posterior, error


def support(weights, labels):
    w, labels = normalized(weights), np.asarray(labels)
    if labels.shape != w.shape or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary and match the weights")
    p = float(np.clip(np.dot(w, labels),0.,1.))
    return p, float(4 * p * (1 - p))


def decide(geometry_value, p_door, disagreement, cfg, probe_used=False):
    if geometry_value < cfg.geometry_threshold:
        return "Avoid", "low geometric shortcut value"
    if disagreement >= cfg.disagreement_threshold:
        if not probe_used:
            return "Probe", "high geometric value and high action disagreement"
        # One standard Probe is allowed. Never pretend that one observation always resolves D.
        return "Avoid", "residual disagreement after the single Probe; conservative fallback"
    if p_door > 0.5:
        return "Push", "high geometric value and strong Door support"
    return "Avoid", "candidate actions agree on Bypass"
