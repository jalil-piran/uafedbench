"""Linear-Gaussian multi-client MDP with exact action values.

Client k has dynamics  x' = A_k x + B_k u + w,  w ~ N(0, W_k),
and reward          r(x, u) = -(x^T Qc x + u^T Rc u).
For a linear policy u = -K x (with optional Gaussian exploration noise of
covariance S_u) the action value is a closed-form quadratic, so an epistemic
interval on Q^pi_phi(x, u) can be scored against the true value itself, as the
reference of Table XXII requires.

Client heterogeneity is controlled by a single distance delta: every client's
dynamics are A_k = A_0 + delta * D_k with fixed random directions D_k.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


@dataclass
class LinearGaussianClient:
    A: np.ndarray
    B: np.ndarray
    W: np.ndarray
    Qc: np.ndarray
    Rc: np.ndarray
    gamma: float = 0.95
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    # ---------- exact values ----------
    def value_matrix(self, K: np.ndarray) -> tuple[np.ndarray, float]:
        """P and c with V^K(x) = -(x^T P x + c) for u = -K x (no exploration noise)."""
        Acl = self.A - self.B @ K
        if np.max(np.abs(np.linalg.eigvals(np.sqrt(self.gamma) * Acl))) >= 1.0:
            raise ValueError("policy is not stabilizing under discounting; value is infinite")
        M = self.Qc + K.T @ self.Rc @ K
        # P = M + gamma Acl^T P Acl  <=>  P = (sqrt(g) Acl)^T P (sqrt(g) Acl) + M
        P = solve_discrete_lyapunov(np.sqrt(self.gamma) * Acl.T, M)
        c = self.gamma * np.trace(P @ self.W) / (1.0 - self.gamma)
        return P, float(c)

    def q_value(self, K: np.ndarray, x: np.ndarray, u: np.ndarray) -> float:
        """Exact Q^K(x, u): take u now, follow u = -K x afterwards."""
        P, c = self.value_matrix(K)
        nxt = self.A @ x + self.B @ u
        return float(-(x @ self.Qc @ x + u @ self.Rc @ u) - self.gamma * (nxt @ P @ nxt) - c)

    # ---------- simulation ----------
    def step(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, float]:
        r = -(x @ self.Qc @ x + u @ self.Rc @ u)
        w = self.rng.multivariate_normal(np.zeros(self.n), self.W)
        return self.A @ x + self.B @ u + w, float(r)

    def rollout_return(self, K: np.ndarray, x: np.ndarray, u: np.ndarray, horizon: int) -> float:
        """One realized discounted return from (x, u), then u = -K x, truncated at horizon."""
        G, disc = 0.0, 1.0
        x, r = self.step(x, u)
        G += r
        for _ in range(horizon - 1):
            disc *= self.gamma
            u_t = -K @ x
            x, r = self.step(x, u_t)
            G += disc * r
        return G

    def collect(self, K: np.ndarray, T: int, sigma_u: float = 0.5, x0=None):
        """Transitions under u = -K x + exploration noise; returns X, U, Xn."""
        x = np.zeros(self.n) if x0 is None else np.array(x0, float)
        X, U, Xn = [], [], []
        for _ in range(T):
            u = -K @ x + sigma_u * self.rng.standard_normal(self.m)
            xn, _ = self.step(x, u)
            X.append(x); U.append(u); Xn.append(xn)
            x = xn
        return np.array(X), np.array(U), np.array(Xn)


def make_federation(n_clients: int = 5, delta: float = 0.1, n: int = 2, m: int = 1,
                    noise: float = 0.1, gamma: float = 0.95, seed: int = 0):
    """Clients with A_k = A_0 + delta * D_k; returns (clients, A_0, B)."""
    rng = np.random.default_rng(seed)
    A0 = np.array([[1.0, 0.1], [0.0, 1.0]]) if n == 2 else np.eye(n) + 0.1 * rng.standard_normal((n, n))
    B = np.array([[0.0], [0.1]]) if (n, m) == (2, 1) else 0.1 * rng.standard_normal((n, m))
    clients = []
    for k in range(n_clients):
        D = rng.standard_normal((n, n)); D /= np.linalg.norm(D, 2)
        clients.append(LinearGaussianClient(A0 + delta * D, B.copy(), noise ** 2 * np.eye(n),
                                            np.eye(n), 0.1 * np.eye(m), gamma,
                                            rng=np.random.default_rng(seed * 1000 + k)))
    return clients, A0, B


def stabilizing_gain(A: np.ndarray, B: np.ndarray, Qc: np.ndarray, Rc: np.ndarray,
                     gamma: float = 0.95, iters: int = 500) -> np.ndarray:
    """Discounted LQR gain by Riccati iteration (a convenient evaluation policy)."""
    P = Qc.copy()
    for _ in range(iters):
        S = Rc + gamma * B.T @ P @ B
        K = gamma * np.linalg.solve(S, B.T @ P @ A)
        P = Qc + K.T @ Rc @ K + gamma * (A - B @ K).T @ P @ (A - B @ K)
    return K
