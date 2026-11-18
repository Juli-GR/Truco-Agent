"""What a policy is, and the policy given by a network.

A policy receives a list of states and returns a matrix (states x actions) with the probability of
each action, so a network evaluates many games in a single forward pass. The fixed opponents
(`truco_rivales`), the networks and anything else that plays implement this interface; the arena
and the self-play environment only rely on it.
"""

import copy
from typing import Protocol

import numpy as np
import torch
from torch import nn

from truco.games.truco import TrucoState


class Policy(Protocol):
    def action_probabilities(self, states: list[TrucoState]) -> np.ndarray:
        """Matrix (states x actions) with the probability of each action."""
        ...


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One action per row of `probs`, sampled according to that row."""
    cumulative = np.cumsum(probs, axis=1)
    u = rng.random(len(probs)) * cumulative[:, -1]
    actions = (cumulative <= u[:, None]).sum(axis=1)
    return np.minimum(actions, probs.shape[1] - 1)


class NetworkPolicy:
    """Policy given by a network that maps information state tensors to one score per action.

    The input is whatever the game's `observacion` parameter encodes. With `greedy=False` the
    scores are logits and the policy is their softmax over the legal actions (PPO's actor). With
    `greedy=True` it plays the legal action with the highest score (DQN's Q-network).
    """

    def __init__(self, network: nn.Module, greedy: bool = False, device: str = "cpu"):
        self.network = network
        self.greedy = greedy
        self.device = device

    @classmethod
    def snapshot(cls, network: nn.Module, greedy: bool = False) -> "NetworkPolicy":
        """Frozen copy of `network`: later training does not change it."""
        frozen = copy.deepcopy(network).cpu().eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        return cls(frozen, greedy)

    def action_probabilities(self, states: list[TrucoState]) -> np.ndarray:
        obs = np.array(
            [s.information_state_tensor(s.current_player()) for s in states], dtype=np.float32
        )
        legal = np.zeros((len(states), states[0].num_distinct_actions()), dtype=bool)
        for i, state in enumerate(states):
            legal[i, state.legal_actions()] = True

        with torch.no_grad():
            scores = self.network(torch.as_tensor(obs, device=self.device)).cpu().numpy()
        scores = np.where(legal, scores.astype(np.float64), -np.inf)
        if self.greedy:
            probs = np.zeros_like(scores)
            probs[np.arange(len(states)), scores.argmax(axis=1)] = 1.0
            return probs
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)
