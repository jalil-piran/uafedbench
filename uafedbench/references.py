"""Scoring functions for the four uncertainty objects of Table XXII.

1. Predictive return distribution or interval  -> predictive_coverage, crps_samples
2. Epistemic interval for Q^pi_phi(s, a)        -> epistemic_coverage
3. Predicted safety probability                 -> brier, spiegelhalter_z
4. Tail-risk estimate (VaR, CVaR)               -> tail_reference, fz0_score

Every function returns the statistic together with the sample size or standard
error the protocol asks to be reported alongside it.
"""
from __future__ import annotations
import numpy as np


# ---------- 1. predictive returns ----------
def predictive_coverage(lo, hi, realized) -> dict:
    """Fraction of realized returns inside [lo, hi], with its binomial standard error.

    Returns from one episode are dependent; pass one return per episode, or
    treat the result as approximate.
    """
    lo, hi, y = map(np.asarray, (lo, hi, realized))
    hit = (y >= lo) & (y <= hi)
    p = hit.mean()
    return {"coverage": float(p), "se": float(np.sqrt(p * (1 - p) / hit.size)), "n": int(hit.size)}


def crps_samples(samples, y) -> float:
    """Continuous ranked probability score of a sample-based predictive distribution."""
    s = np.asarray(samples, float)
    return float(np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :])))


# ---------- 2. epistemic intervals ----------
def epistemic_coverage(intervals, true_values) -> dict:
    """Coverage of the TRUE expected values by intervals, one per independent run.

    Each entry must come from a separate training run on an independently
    generated dataset or environment; realized returns are not a valid reference.
    """
    iv = np.asarray(intervals, float)
    q = np.asarray(true_values, float)
    hit = (q >= iv[:, 0]) & (q <= iv[:, 1])
    p = hit.mean()
    return {"coverage": float(p), "se": float(np.sqrt(p * (1 - p) / hit.size)),
            "n_runs": int(hit.size), "n_unbounded": int(np.sum(~np.isfinite(iv).all(axis=1))),
            "median_width": float(np.median(iv[:, 1] - iv[:, 0]))}


# ---------- 3. safety probabilities ----------
def brier(p_pred, outcome) -> float:
    p, o = np.asarray(p_pred, float), np.asarray(outcome, float)
    return float(np.mean((p - o) ** 2))


def spiegelhalter_z(p_pred, outcome) -> dict:
    """Binning-free calibration test; |z| > 1.96 rejects calibration at the 5% level."""
    p, o = np.asarray(p_pred, float), np.asarray(outcome, float)
    num = np.sum((o - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    return {"z": float(num / den), "n": int(p.size)}


# ---------- 4. tail risk ----------
def tail_reference(returns, alpha=0.1) -> dict:
    """High-precision VaR and CVaR (lower tail of returns) with a bootstrap standard error."""
    g = np.sort(np.asarray(returns, float))
    k = max(1, int(np.floor(alpha * g.size)))
    var, cvar = g[k - 1], g[:k].mean()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(200):
        b = np.sort(rng.choice(g, g.size, replace=True))
        bs.append(b[:k].mean())
    return {"VaR": float(var), "CVaR": float(cvar), "CVaR_se": float(np.std(bs)), "n_tail": k}


def fz0_score(var_pred, es_pred, y, alpha=0.1) -> float:
    """FZ0 joint scoring function for (VaR, ES) of the lower tail (Fissler-Ziegel family).

    Strictly consistent when ES < 0, i.e. for losses expressed as negative
    returns below zero; shift returns so that the predicted ES is negative.
    Lower is better. Averaged over observations.
    """
    v, e, y = map(lambda a: np.asarray(a, float), (var_pred, es_pred, y))
    ind = (y <= v).astype(float)
    return float(np.mean(-ind * (v - y) / (alpha * e) + v / e + np.log(-e) - 1.0))
