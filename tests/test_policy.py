import numpy as np
import pyspiel
import pytest
import torch

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.policy import NetworkPolicy, SeatPolicy, sample_actions


@pytest.fixture(scope="module")
def game():
    return pyspiel.load_game("truco")


def test_sample_actions_follows_probabilities():
    rng = np.random.default_rng(0)
    probs = np.tile([0.0, 0.25, 0.0, 0.75], (20000, 1))
    counts = np.bincount(sample_actions(probs, rng), minlength=4)
    assert counts[0] == counts[2] == 0
    assert counts[3] / counts.sum() == pytest.approx(0.75, abs=0.02)


def linear_network(game):
    return torch.nn.Linear(game.information_state_tensor_size(), game.num_distinct_actions())


@pytest.mark.parametrize("greedy", [False, True])
def test_network_policy_is_a_distribution_over_legal_actions(game, greedy, make_states):
    torch.manual_seed(0)
    network = linear_network(game)
    states = make_states(game, 32)
    probs = NetworkPolicy(network, greedy).action_probabilities(states)
    assert np.allclose(probs.sum(axis=1), 1)
    for row, state in zip(probs, states, strict=True):
        illegal = np.setdiff1d(np.arange(len(row)), state.legal_actions())
        assert np.all(row[illegal] == 0)
        if greedy:
            assert np.count_nonzero(row) == 1


def test_snapshot_does_not_change_when_the_network_trains(game, make_states):
    network = linear_network(game)
    frozen = NetworkPolicy.snapshot(network)
    states = make_states(game, 8)
    before = frozen.action_probabilities(states)
    with torch.no_grad():
        network.weight.add_(1.0)
    assert np.array_equal(before, frozen.action_probabilities(states))
    assert not np.array_equal(before, NetworkPolicy(network).action_probabilities(states))


class ConstantPolicy:
    """Gives every state the same row: probability 1 on `action`."""

    def __init__(self, action):
        self.action = action

    def action_probabilities(self, states):
        probs = np.zeros((len(states), states[0].num_distinct_actions()))
        probs[:, self.action] = 1.0
        return probs


def test_seat_policy_plays_each_state_with_its_player_policy(game, make_states):
    states = make_states(game, 32)
    players = np.array([s.current_player() for s in states])
    assert set(players) == {0, 1}
    probs = SeatPolicy([ConstantPolicy(0), ConstantPolicy(1)]).action_probabilities(states)
    np.testing.assert_array_equal(probs.argmax(axis=1), players)
