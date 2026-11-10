"""UA-FedBench: a minimal reference environment for uncertainty-aware federated and networked RL.

Components
- env:        linear-Gaussian multi-client MDP with closed-form action values
- posterior:  exact Bayesian posterior over each client's dynamics
- channel:    Gilbert-Elliott link, optionally coupled to the sender's state
- references: scoring functions for the four objects of Table XXII
"""
from .env import LinearGaussianClient, make_federation, stabilizing_gain
from .posterior import DynamicsPosterior, epistemic_q_interval
from .channel import GilbertElliott, CachedCopy
from .references import (predictive_coverage, crps_samples, epistemic_coverage,
                         brier, spiegelhalter_z, tail_reference, fz0_score)

__version__ = "1.0.0"
