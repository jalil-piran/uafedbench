# UA-FedBench

A minimal reference environment for uncertainty-aware federated and networked reinforcement learning, released with the survey
"A Survey on Uncertainty-Aware Deep Reinforcement Learning for Federated and Networked Autonomous Systems" (submitted to IEEE Communications Surveys & Tutorials).

It is an instrument for checking uncertainty claims against ground truth, not a benchmark suite.

## What it provides

| Module | Content | Survey reference |
|---|---|---|
| `env.py` | Linear-Gaussian multi-client MDP. Client k has dynamics A_k = A_0 + delta * D_k, so client distance is one parameter. Action values of any stabilizing linear policy are closed-form quadratics, so epistemic intervals can be scored against the true Q. | Sec. III-C, protocol items 1, 5 |
| `posterior.py` | Exact Bayesian posterior over each client's dynamics (known noise), and credible intervals for Q obtained by evaluating the closed-form value of posterior samples. | Sec. III-B |
| `channel.py` | Gilbert-Elliott link, optionally coupled to a sender-side "stress" signal so that losses concentrate where the agent knows least; cached copy with age. | Sec. VI, protocol item 6 |
| `references.py` | Scoring for the four uncertainty objects of Table XXII: predictive coverage and CRPS; epistemic coverage against true values over independent runs; Brier score and Spiegelhalter's binning-free z-test; tail reference (VaR, CVaR) and the FZ0 joint (VaR, ES) score. | Table XXII, protocol items 2, 3, 8 |
| `examples/demo.py` | End-to-end check: epistemic coverage with own data versus pooled data at increasing client distance; age profile of a coupled and an uncoupled link. | |
| `examples/sim_comm_weight.py` | The numerical check of the fusion weight in Sec. VI-A (Table of Sec. VI-A). | Sec. VI-A |

## Quick start

    pip install numpy scipy
    PYTHONPATH=. python tests/test_core.py
    PYTHONPATH=. python examples/demo.py

Expected demo output (100 independent runs, nominal 90%):

    own data only            coverage 0.94
    pooled, delta = 0.0      coverage 0.96   (narrower intervals)
    pooled, delta = 0.02     coverage 0.89
    pooled, delta = 0.05     coverage 0.63

Wilson 95% intervals over the 100 runs: own data [0.875, 0.972]; pooled delta 0 [0.902, 0.984];
delta 0.02 [0.814, 0.937]; delta 0.05 [0.532, 0.718]. Run r uses federation seed r and posterior-sample
generator seed r. Credible intervals are the 5th-95th percentiles of closed-form Q over 300 posterior samples.

The fusion-weight check (examples/sim_comm_weight.py) uses kappa = 0.05, N = 10 senders,
n_j ~ U[50, 2000], sigma_j^2 ~ U[0.5, 25], p_j ~ U[0.3, 0.95], 300 rounds with 50 burn-in,
200 runs per network, and network seeds 0-4; the age-decay tau is chosen post hoc from
{0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32}. Reported brackets are the min and max of the five network means.

## Limits

Linear dynamics and quadratic rewards cannot reproduce function-approximation error, bootstrapped targets, or the proxy U-hat of Sec. III-A. The environment is meant to check whether an estimator, a fusion rule, or a protocol item behaves correctly where the truth is known, before it is trusted on deep agents.

## License

MIT (see LICENSE). Replace if your institution requires otherwise.
