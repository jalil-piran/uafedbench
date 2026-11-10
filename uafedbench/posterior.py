"""Bayesian linear-regression posterior over a client's dynamics.

With known noise covariance W = s^2 I, each row of Theta = [A B] has a Gaussian
posterior under a Gaussian prior, so posterior samples of (A, B) are exact and
the epistemic spread of Q^K is computed by evaluating the closed-form value of
each sample. This is the reference estimator against which learned estimators
can be compared, and it lets the coverage of Bayesian credible intervals be
checked against frequentist repetition (Section III-B).
"""
from __future__ import annotations
import numpy as np


class DynamicsPosterior:
    def __init__(self, n: int, m: int, noise_var: float, prior_var: float = 10.0):
        self.n, self.m, self.s2 = n, m, noise_var
        d = n + m
        self.prec = np.eye(d) / prior_var          # shared across rows (isotropic noise)
        self.b = np.zeros((d, n))

    def update(self, X: np.ndarray, U: np.ndarray, Xn: np.ndarray) -> None:
        Z = np.hstack([X, U])
        self.prec = self.prec + Z.T @ Z / self.s2
        self.b = self.b + Z.T @ Xn / self.s2

    def mean(self) -> tuple[np.ndarray, np.ndarray]:
        Th = np.linalg.solve(self.prec, self.b).T     # n x (n+m)
        return Th[:, :self.n], Th[:, self.n:]

    def sample(self, rng: np.random.Generator, size: int):
        cov = np.linalg.inv(self.prec)
        L = np.linalg.cholesky(cov)
        mu = np.linalg.solve(self.prec, self.b)       # (n+m) x n
        out = []
        for _ in range(size):
            Th = (mu + L @ rng.standard_normal(mu.shape)).T
            out.append((Th[:, :self.n], Th[:, self.n:]))
        return out


def epistemic_q_interval(client, post: DynamicsPosterior, K, x, u, level=0.9,
                         n_samples=400, rng=None):
    """Central credible interval for Q^K(x,u) from posterior samples of (A, B).

    Samples whose closed loop is not stable under discounting are dropped and
    counted; a large dropped fraction is itself a sign of high epistemic
    uncertainty.
    """
    from .env import LinearGaussianClient
    rng = np.random.default_rng() if rng is None else rng
    qs, dropped = [], 0
    for A, B in post.sample(rng, n_samples):
        c = LinearGaussianClient(A, B, client.W, client.Qc, client.Rc, client.gamma)
        try:
            qs.append(c.q_value(K, x, u))
        except ValueError:
            dropped += 1
    qs = np.array(qs)
    if qs.size == 0:
        # every posterior sample is unstable: the honest interval is unbounded
        return -np.inf, np.inf, 1.0
    lo, hi = np.quantile(qs, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi), dropped / n_samples
