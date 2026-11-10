"""Gilbert-Elliott link and the age of the receiver's cached copy.

A two-state Markov chain alternates between a good state G and a bad state B
with transition probabilities p_gb (G->B) and p_bg (B->G); a message is lost
with probability e_g in G and e_b in B. Optionally, the probability of
entering B is raised by a coupling term that depends on the sender's state,
which reproduces the cell-edge case in which links fail exactly where the
agent knows least (Section VI-A, scenario B).
"""
from __future__ import annotations
import numpy as np


class GilbertElliott:
    def __init__(self, p_gb=0.05, p_bg=0.3, e_g=0.02, e_b=0.6, coupling=0.0, seed=0):
        self.p_gb, self.p_bg, self.e_g, self.e_b, self.coupling = p_gb, p_bg, e_g, e_b, coupling
        self.rng = np.random.default_rng(seed)
        self.bad = False

    def stationary_loss(self) -> float:
        pi_b = self.p_gb / (self.p_gb + self.p_bg)
        return (1 - pi_b) * self.e_g + pi_b * self.e_b

    def transmit(self, stress: float = 0.0) -> bool:
        """Advance the chain one slot and return True if the message is delivered.

        stress in [0, 1] is any sender-side quantity (e.g. a normalized epistemic
        estimate or a drift magnitude); with coupling > 0 it raises the
        probability of entering the bad state.
        """
        p_gb = min(1.0, self.p_gb + self.coupling * stress)
        if self.bad:
            self.bad = self.rng.random() >= self.p_bg
        else:
            self.bad = self.rng.random() < p_gb
        loss = self.e_b if self.bad else self.e_g
        return self.rng.random() >= loss


class CachedCopy:
    """Receiver-side cache: freshest delivered value and its age (eq. comm_general)."""

    def __init__(self, value=None):
        self.value, self.age = value, 0

    def tick(self, delivered: bool, new_value=None):
        if delivered:
            self.value, self.age = new_value, 0
        else:
            self.age += 1
        return self.value, self.age
