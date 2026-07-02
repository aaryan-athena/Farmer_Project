# inference.py — pure prediction helper (no label data; stays decoupled)
import numpy as np


def restrict_to_crop(probs, allowed):
    """Keep only the chosen crop's classes, then renormalize so the
    confidence shown to the farmer is honest within that crop."""
    masked = np.zeros_like(probs)
    masked[allowed] = probs[allowed]
    total = masked.sum()
    return masked / total if total > 0 else masked
