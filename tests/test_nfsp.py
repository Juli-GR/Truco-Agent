import numpy as np
import pyspiel
import pytest
from open_spiel.python import rl_environment

import truco.games.truco  # noqa: F401  (registers the game)
from truco.algorithms.nfsp import (
    average_policy,
    load_nfsp_policy,
    make_agents,
    save_average_policy,
    train_episode,
)

HIDDEN = [16]


@pytest.fixture(scope="module")
def game():
    return pyspiel.load_game("truco", {"observacion": "compact"})


def small_agents(game):
    return make_agents(
        game,
        HIDDEN,
        reservoir_buffer_capacity=1000,
        anticipatory_param=0.5,
        batch_size=8,
        min_buffer_size_to_learn=8,
        learn_every=4,
        replay_buffer_capacity=1000,
    )


def test_training_episodes_are_zero_sum_and_both_seats_learn(game):
    env = rl_environment.Environment(game, seed=0)
    agents = small_agents(game)
    for _ in range(30):
        returns = train_episode(env, agents)
        assert returns[0] == -returns[1]
    for agent in agents:
        sl_loss, rl_loss = agent.loss
        assert sl_loss is not None and rl_loss is not None


def test_frozen_average_policy_does_not_follow_training(game, make_states):
    env = rl_environment.Environment(game, seed=0)
    agents = small_agents(game)
    states = make_states(game, 16)
    frozen = average_policy(agents)
    before = frozen.action_probabilities(states)
    for _ in range(30):
        train_episode(env, agents)
    np.testing.assert_array_equal(frozen.action_probabilities(states), before)
    live = average_policy(agents, frozen=False).action_probabilities(states)
    assert not np.array_equal(live, before)


def test_loaded_policy_plays_like_the_saved_agents(game, tmp_path, make_states):
    agents = small_agents(game)
    save_average_policy(agents, tmp_path / "checkpoint.pt")
    loaded = load_nfsp_policy(tmp_path / "checkpoint.pt", game, HIDDEN)
    states = make_states(game, 16)
    expected = average_policy(agents).action_probabilities(states)
    np.testing.assert_allclose(loaded.action_probabilities(states), expected)
