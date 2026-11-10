"""Self-checks; run with `python -m pytest tests` or `python tests/test_core.py`."""
import numpy as np
from uafedbench import (make_federation, stabilizing_gain, GilbertElliott,
                        epistemic_coverage, predictive_coverage, spiegelhalter_z,
                        tail_reference, fz0_score)


def test_exact_q_matches_monte_carlo():
    clients, A0, B = make_federation(n_clients=1, delta=0.0, seed=1)
    c = clients[0]
    K = stabilizing_gain(c.A, c.B, c.Qc, c.Rc, c.gamma)
    x, u = np.array([1.0, -0.5]), np.array([0.3])
    q = c.q_value(K, x, u)
    G = [c.rollout_return(K, x, u, horizon=300) for _ in range(3000)]
    se = np.std(G) / np.sqrt(len(G))
    assert abs(np.mean(G) - q) < 4 * se + 1e-3, (np.mean(G), q, se)


def test_gilbert_elliott_stationary_loss():
    ch = GilbertElliott(p_gb=0.05, p_bg=0.3, e_g=0.02, e_b=0.6, seed=3)
    delivered = np.mean([ch.transmit() for _ in range(200000)])
    assert abs((1 - delivered) - ch.stationary_loss()) < 0.01


def test_coverage_of_exact_intervals():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=4000)
    est = truth + rng.normal(size=4000)
    iv = np.c_[est - 1.645, est + 1.645]
    assert abs(epistemic_coverage(iv, truth)["coverage"] - 0.9) < 0.02
    assert abs(predictive_coverage(iv[:, 0], iv[:, 1], truth)["coverage"] - 0.9) < 0.02


def test_spiegelhalter_accepts_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    o = rng.random(20000) < p
    assert abs(spiegelhalter_z(p, o)["z"]) < 3


def test_fz0_prefers_true_tail():
    rng = np.random.default_rng(2)
    y = rng.normal(-5, 1, 50000)
    ref = tail_reference(y, alpha=0.1)
    good = fz0_score(ref["VaR"], ref["CVaR"], y, 0.1)
    bad = fz0_score(ref["VaR"] + 0.8, ref["CVaR"] + 0.8, y, 0.1)
    assert good < bad


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
