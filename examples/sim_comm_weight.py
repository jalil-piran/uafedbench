"""Numerical check of the fusion weight (7) on a toy linear-Gaussian network.

Model. A common latent value theta_t (a value at a probe state) drifts as a
Gaussian random walk with per-round variance kappa. Each of N senders j holds an
unbiased estimate y_{j,s} = theta_s + e_{j,s}, e ~ N(0, vc_j), refreshed every
round; vc_j = sigma_j^2 / n_j, so data volume n_j and precision 1/vc_j disagree.
Sender j's message reaches the receiver with probability p_j (Bernoulli, i.i.d.),
and the receiver keeps the freshest copy with age a_j. The receiver's own
estimate (j = 0) is always fresh.

Fusion rules for theta_t from the cached copies:
  eq7        w ∝ 1 / (vc_j + kappa_hat * a_j)        (content + age variance)
  eq7_k/4    same with kappa_hat = kappa / 4
  eq7_kx4    same with kappa_hat = 4 kappa
  content    w ∝ 1 / vc_j                            (age ignored)
  agedecay   w ∝ n_j * exp(-a_j / tau), tau tuned post hoc on the test runs
  volume     w ∝ n_j                                  (FedAvg-style)
  equal      w = 1 / (N + 1)
  oracle     Kalman filter on all received messages (the MMSE benchmark)

Metrics: MSE of the fused value relative to the oracle, and for eq7 the
empirical coverage of the nominal 90% interval  x_hat ± 1.645 sqrt(V), with
V = sum_j w_j^2 (vc_j + kappa_hat a_j)  (the variance eq7 implies when copies
are treated as independent).

Scenario A: losses independent of everything.
Scenario B: "cell edge" -- a sender's loss probability rises sharply in rounds
where the latent drift is large, so copies go stale exactly when theta moves.
"""
import numpy as np

def run(scenario, R=400, T=300, burn=50, N=10, kappa=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(50, 2000, N + 1).astype(float)       # data volume
    sigma2 = rng.uniform(0.5, 25.0, N + 1)                 # per-sample noise, unrelated to n
    vc = sigma2 / n
    p = rng.uniform(0.3, 0.95, N + 1); p[0] = 1.0
    rules = ['eq7', 'eq7_k/4', 'eq7_kx4', 'content', 'volume', 'equal', 'oracle']
    taus = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    se = {r: [] for r in rules}; se_ad = {tau: [] for tau in taus}
    cover = []
    for _ in range(R):
        theta = 0.0
        cache_val = np.zeros(N + 1); cache_age = np.full(N + 1, 0)
        # oracle Kalman: state theta, measurements y_j (delayed copies are dropped
        # from the oracle; it uses only messages that arrive this round, which is
        # optimal for a random walk since older copies carry no new information
        # once their round has been processed)
        m, P = 0.0, 1.0
        for t in range(T):
            w = rng.normal(0, np.sqrt(kappa)); theta += w
            y = theta + rng.normal(0, np.sqrt(vc))
            if scenario == 'A':
                arrive = rng.random(N + 1) < p
            else:  # B: loss correlated with drift magnitude
                big = abs(w) > 1.0 * np.sqrt(kappa)
                pp = p * (0.25 if big else 1.0)
                arrive = rng.random(N + 1) < pp
            arrive[0] = True
            cache_age += 1
            cache_val[arrive] = y[arrive]; cache_age[arrive] = 0
            # oracle update
            P = P + kappa
            for j in np.where(arrive)[0]:
                K = P / (P + vc[j]); m = m + K * (y[j] - m); P = (1 - K) * P
            if t < burn:
                continue
            a = cache_age.astype(float)
            def fuse(wt):
                wt = wt / wt.sum(); return wt @ cache_val, wt
            for name, kh in [('eq7', kappa), ('eq7_k/4', kappa / 4), ('eq7_kx4', kappa * 4)]:
                xh, wt = fuse(1.0 / (vc + kh * a))
                se[name].append((xh - theta) ** 2)
                if name == 'eq7':
                    V = np.sum(wt ** 2 * (vc + kh * a))
                    cover.append(abs(xh - theta) <= 1.645 * np.sqrt(V))
            se['content'].append((fuse(1.0 / vc)[0] - theta) ** 2)
            se['volume'].append((fuse(n.copy())[0] - theta) ** 2)
            se['equal'].append((fuse(np.ones(N + 1))[0] - theta) ** 2)
            se['oracle'].append((m - theta) ** 2)
            for tau in taus:
                se_ad[tau].append((fuse(n * np.exp(-a / tau))[0] - theta) ** 2)
    mse = {r: float(np.mean(v)) for r, v in se.items()}
    best_tau = min(taus, key=lambda t: np.mean(se_ad[t]))
    mse['agedecay'] = float(np.mean(se_ad[best_tau]))
    rel = {r: mse[r] / mse['oracle'] for r in mse}
    return rel, float(np.mean(cover)), best_tau

if __name__ == '__main__':
    import json
    order = ['oracle', 'eq7', 'eq7_k/4', 'eq7_kx4', 'content', 'agedecay', 'volume', 'equal']
    out = {}
    for sc in ['A', 'B']:
        rels, covs, taus_ = [], [], []
        for seed in range(5):
            rel, cov, tau = run(sc, R=200, seed=seed)
            rels.append(rel); covs.append(cov); taus_.append(tau)
        out[sc] = {r: [x[r] for x in rels] for r in order}
        out[sc]['coverage'] = covs; out[sc]['tau'] = taus_
        print(f'Scenario {sc}: eq7 coverage mean {np.mean(covs):.3f} [{min(covs):.3f}, {max(covs):.3f}]; best taus {taus_}')
        for r in order:
            v = out[sc][r]
            print(f'  {r:9s} MSE/oracle mean {np.mean(v):.2f}  range [{min(v):.2f}, {max(v):.2f}]')
    json.dump(out, open('sim_results.json', 'w'), indent=1)
