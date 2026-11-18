import numpy as np
import pyspiel
import torch

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.policy import NetworkPolicy
from truco.algorithms.ppo import PPONetwork, load_ppo_policy


def test_load_ppo_policy_plays_like_the_saved_network(tmp_path, make_states):
    game = pyspiel.load_game("truco", {"observacion": "compact"})
    size = game.information_state_tensor_size()
    network = PPONetwork(game.num_distinct_actions(), (size,), "cpu", hidden=32)
    torch.save(network.state_dict(), tmp_path / "checkpoint.pt")
    loaded = load_ppo_policy(tmp_path / "checkpoint.pt", game, hidden=32)
    states = make_states(game, 8)
    expected = NetworkPolicy(network.actor).action_probabilities(states)
    np.testing.assert_allclose(loaded.action_probabilities(states), expected)
