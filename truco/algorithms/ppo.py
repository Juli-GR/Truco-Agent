"""PPO's network for OpenSpiel's PPO, and loading its checkpoints as policies."""

import numpy as np
import pyspiel
import torch
from open_spiel.python.pytorch.ppo import INVALID_ACTION_PENALTY, CategoricalMasked, layer_init
from torch import nn

from truco.agents.policy import NetworkPolicy


class PPONetwork(nn.Module):
    """Actor and critic for OpenSpiel's PPO (`agent_fn`): two separate MLPs with ReLU.

    Same interface as OpenSpiel's `PPOAgent`, which is fixed at two hidden layers of 64 with tanh.
    """

    def __init__(self, num_actions, observation_shape, device, hidden: int = 256):
        super().__init__()
        size = int(np.prod(observation_shape))

        def mlp(out: int, std: float) -> nn.Sequential:
            return nn.Sequential(
                layer_init(nn.Linear(size, hidden)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden, hidden)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden, out), std=std),
            )

        self.actor = mlp(num_actions, std=0.01)
        self.critic = mlp(1, std=1.0)
        self.device = device
        self.num_actions = num_actions
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()
        probs = CategoricalMasked(
            logits=self.actor(x), masks=legal_actions_mask, mask_value=self.mask_value
        )
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), probs.probs


def load_ppo_policy(path, game: pyspiel.Game, hidden: int) -> NetworkPolicy:
    """Frozen policy of a `PPONetwork` checkpoint (a state_dict saved with torch.save).

    `game` must have the `observacion` the network was trained with.
    """
    network = PPONetwork(
        game.num_distinct_actions(), (game.information_state_tensor_size(),), "cpu", hidden
    )
    network.load_state_dict(torch.load(path, map_location="cpu"))
    return NetworkPolicy.snapshot(network.actor)
