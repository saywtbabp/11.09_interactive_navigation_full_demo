"""Display only: fixed brightness reference shared by live and saved plots."""
import numpy as np


def weight_strength(weights):
    """Map weight to [0,1) against uniform 1/N, never this frame's maximum.

    strength = w / (w + 1/N): an equal-weight prior is always 0.5;
    0.1x the prior is 0.091, 10x is 0.909. High posterior weights brighten
    and low weights dim on the SAME scale. Does not modify policy weights.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('invalid particle display weights')
    w = w / w.sum()
    return w / (w + 1. / len(w))
