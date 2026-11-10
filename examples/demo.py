"""End-to-end demo of the protocol items on the reference environment.

1. Epistemic coverage (item 2, object 2): for each of R independent runs, a
   client collects T transitions, forms the exact Bayesian posterior over its
   dynamics, and reports a 90% credible interval for Q^K(x, u). Coverage is the
   fraction of runs whose interval contains the true Q, not realized returns.
2. Heterogeneity (item 5): the same check when the client pools data from
   peers whose dynamics sit at distance delta, which is what naive federation
   does; coverage falls as delta grows.
3. Channel (item 6): a Gilbert-Elliott link between client and server, with
   and without coupling to the sender's state, and the resulting age profile.
"""
import numpy as np
from uafedbench import (make_federation, stabilizing_gain, DynamicsPosterior,
                        epistemic_q_interval, epistemic_coverage, GilbertElliott, CachedCopy)

x, u = np.array([1.0, -0.5]), np.array([0.3])


def coverage_run(delta, R=100, T=60, pool=True, seed=0):
    ivs, truths = [], []
    for r in range(R):
        clients, A0, B = make_federation(n_clients=4, delta=delta, seed=seed * 10000 + r)
        target = clients[0]
        K = stabilizing_gain(target.A, target.B, target.Qc, target.Rc, target.gamma)
        post = DynamicsPosterior(target.n, target.m, noise_var=target.W[0, 0])
        sources = clients if pool else clients[:1]
        for c in sources:
            post.update(*c.collect(K, T))
        lo, hi, _ = epistemic_q_interval(target, post, K, x, u, level=0.9, n_samples=300,
                                         rng=np.random.default_rng(r))
        ivs.append((lo, hi)); truths.append(target.q_value(K, x, u))
    return epistemic_coverage(ivs, truths)


if __name__ == "__main__":
    print("Epistemic coverage of nominal 90% intervals, scored against the true Q:")
    print("  own data only          ", coverage_run(0.0, pool=False))
    for d in [0.0, 0.02, 0.05]:
        print(f"  pooled, delta = {d:<5}  ", coverage_run(d, pool=True))

    print("\nGilbert-Elliott link, 20000 slots:")
    for coupling in [0.0, 0.3]:
        ch = GilbertElliott(coupling=coupling, seed=1); cache = CachedCopy(0.0)
        rng = np.random.default_rng(2); ages = []
        for t in range(20000):
            stress = float(rng.random() < 0.2)          # sender is 'stressed' 20% of slots
            _, a = cache.tick(ch.transmit(stress), t); ages.append(a)
        print(f"  coupling {coupling}: loss {1 - np.mean(np.array(ages) == 0):.3f}, "
              f"mean age {np.mean(ages):.2f}, 99th pct age {np.percentile(ages, 99):.0f}")
