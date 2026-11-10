import numpy as np
import pyspiel
import pytest
import torch
from open_spiel.python.pytorch.ppo import PPO, PPOAgent
from open_spiel.python.rl_agent import StepOutput

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.truco_rivales import CONSERVADOR, RIVALES, RandomPolicy
from truco.algorithms.selfplay import AGENT_ID, NetworkPolicy, OpponentPool, SelfPlayEnv
from truco.games.truco import Action

NUM_ENVS = 16


class MazoPolicy:
    """Goes to the mazo as soon as it can, otherwise plays like RandomPolicy."""

    def action_probabilities(self, states):
        probs = RandomPolicy().action_probabilities(states)
        for i, state in enumerate(states):
            if Action.MAZO in state.legal_actions():
                probs[i] = 0
                probs[i, Action.MAZO] = 1
        return probs


@pytest.fixture(scope="module")
def game():
    return pyspiel.load_game("truco", {"niveles": 14})


def random_step_outputs(time_steps, rng):
    return [
        StepOutput(action=rng.choice(ts.observations["legal_actions"][0]), probs=None)
        for ts in time_steps
    ]


def assert_agent_to_play(env, time_steps):
    for i, ts in enumerate(time_steps):
        state = env.states[i]
        assert not state.is_terminal()
        assert state.current_player() == env.seats[i]
        assert ts.current_player() == AGENT_ID
        assert ts.observations["legal_actions"][0] == state.legal_actions()
        assert ts.observations["info_state"][0] == state.information_state_tensor(env.seats[i])


def test_reset_leaves_every_game_at_the_agents_turn(game):
    env = SelfPlayEnv(game, OpponentPool([RIVALES["aleatorio"]]), NUM_ENVS)
    time_steps = env.reset()
    assert all(ts.first() for ts in time_steps)
    assert_agent_to_play(env, time_steps)


def test_opponent_only_moves_between_agent_turns(game):
    rng = np.random.default_rng(0)
    env = SelfPlayEnv(game, OpponentPool([RIVALES["aleatorio"]]), NUM_ENVS, seed=1)
    time_steps = env.reset()
    for _ in range(50):
        time_steps, _, dones, unreset = env.step(random_step_outputs(time_steps, rng))
        assert_agent_to_play(env, time_steps)
        for ts, done in zip(unreset, dones, strict=True):
            assert ts.last() == done


def test_agent_plays_both_seats(game):
    env = SelfPlayEnv(game, OpponentPool([RIVALES["aleatorio"]]), num_envs=2000, seed=2)
    env.reset()
    assert env.seats.mean() == pytest.approx(0.5, abs=0.05)


def test_reward_is_the_agents_return_and_only_at_the_end(game):
    rng = np.random.default_rng(3)
    env = SelfPlayEnv(game, OpponentPool([CONSERVADOR]), NUM_ENVS, seed=3)
    time_steps = env.reset()
    finished = 0
    for _ in range(50):
        seats = env.seats.copy()
        states = list(env.states)
        time_steps, rewards, dones, unreset = env.step(random_step_outputs(time_steps, rng))
        for i, done in enumerate(dones):
            if done:
                finished += 1
                assert rewards[i] == [states[i].returns()[seats[i]]]
                assert rewards[i][0] in (-4, -3, -2, -1, 1, 2, 3, 4)
                assert unreset[i].discounts == [0.0]
                assert time_steps[i].first()
            else:
                assert rewards[i] == [0.0]
    assert finished > NUM_ENVS


def test_without_reset_finished_games_stay_over(game):
    env = SelfPlayEnv(game, OpponentPool([MazoPolicy()]), NUM_ENVS)
    env.reset()
    paso = [StepOutput(action=Action.PASO, probs=None)] * NUM_ENVS
    time_steps, rewards, dones, _ = env.step(paso, reset_if_done=False)
    assert all(dones) and all(ts.last() for ts in time_steps)
    with pytest.raises(ValueError, match="over"):
        env.step(paso)


def test_games_the_opponent_ends_before_the_agent_plays_are_dealt_again(game):
    env = SelfPlayEnv(game, OpponentPool([MazoPolicy()]), NUM_ENVS, seed=4)
    time_steps = env.reset()
    assert_agent_to_play(env, time_steps)
    # Every state is the agent's first decision: the opponent never got to the mazo first.
    assert all(not s.history_actions for s in env.states)
    # If the agent passes, the opponent goes to the mazo and the agent wins 1 point.
    paso = [StepOutput(action=Action.PASO, probs=None)] * NUM_ENVS
    _, rewards, dones, _ = env.step(paso)
    assert all(dones)
    assert rewards == [[1.0]] * NUM_ENVS


def test_illegal_action_raises(game):
    env = SelfPlayEnv(game, OpponentPool([RIVALES["aleatorio"]]), 1)
    (time_step,) = env.reset()
    legal = time_step.observations["legal_actions"][0]
    illegal = next(a for a in range(game.num_distinct_actions()) if a not in legal)
    with pytest.raises(ValueError, match="Illegal"):
        env.step([StepOutput(action=illegal, probs=None)])


def test_pool_decay_zero_always_samples_the_last_added():
    a, b, c = RandomPolicy(), RandomPolicy(), RandomPolicy()
    rng = np.random.default_rng(0)
    pool = OpponentPool([a, b], decay=0.0)
    pool.add(c)
    assert pool.probabilities().tolist() == [0, 0, 1]
    assert {id(pool.sample(rng)) for _ in range(20)} == {id(c)}


def test_pool_decay_one_is_uniform():
    pool = OpponentPool([RandomPolicy() for _ in range(4)], decay=1.0)
    assert np.allclose(pool.probabilities(), 0.25)


def test_pool_weight_decays_geometrically_with_age():
    pool = OpponentPool([RandomPolicy() for _ in range(3)], decay=0.5)
    assert np.allclose(pool.probabilities(), np.array([0.25, 0.5, 1]) / 1.75)
    rng = np.random.default_rng(0)
    counts = np.bincount(
        [pool.opponents.index(pool.sample(rng)) for _ in range(20000)], minlength=3
    )
    assert np.allclose(counts / counts.sum(), pool.probabilities(), atol=0.02)


def test_pool_rejects_empty_sampling_and_invalid_decay():
    with pytest.raises(ValueError, match="empty"):
        OpponentPool().sample(np.random.default_rng(0))
    for decay in (-0.1, 1.1):
        with pytest.raises(ValueError, match="decay"):
            OpponentPool(decay=decay)


def make_states(game, n, seed=0):
    """`n` states at a decision of some player, reached by random play."""
    rng = np.random.default_rng(seed)
    env = SelfPlayEnv(game, OpponentPool([RIVALES["aleatorio"]]), n, seed=seed)
    time_steps = env.reset()
    for _ in range(2):
        time_steps, *_ = env.step(random_step_outputs(time_steps, rng))
    return env.states


def test_network_policy_is_a_distribution_over_legal_actions(game):
    torch.manual_seed(0)
    network = torch.nn.Linear(game.information_state_tensor_size(), game.num_distinct_actions())
    states = make_states(game, 32)
    for greedy in (False, True):
        probs = NetworkPolicy(network, greedy).action_probabilities(states)
        assert np.allclose(probs.sum(axis=1), 1)
        for row, state in zip(probs, states, strict=True):
            illegal = np.setdiff1d(np.arange(len(row)), state.legal_actions())
            assert np.all(row[illegal] == 0)
            if greedy:
                assert np.count_nonzero(row) == 1


def test_snapshot_does_not_change_when_the_network_trains(game):
    network = torch.nn.Linear(game.information_state_tensor_size(), game.num_distinct_actions())
    frozen = NetworkPolicy.snapshot(network)
    states = make_states(game, 8)
    before = frozen.action_probabilities(states)
    with torch.no_grad():
        network.weight.add_(1.0)
    assert np.array_equal(before, frozen.action_probabilities(states))
    assert not np.array_equal(before, NetworkPolicy(network).action_probabilities(states))


def test_openspiel_ppo_trains_in_self_play(game):
    env = SelfPlayEnv(game, OpponentPool(decay=1.0), NUM_ENVS)
    agent = PPO(
        env.info_state_shape,
        env.num_actions,
        num_players=1,
        player_id=AGENT_ID,
        num_envs=NUM_ENVS,
        steps_per_batch=8,
        agent_fn=PPOAgent,
    )
    env.pool.add(NetworkPolicy.snapshot(agent.network.actor))
    time_steps = env.reset()
    for _ in range(2):
        for _ in range(agent.steps_per_batch):
            outputs = agent.step(time_steps)
            time_steps, rewards, dones, _ = env.step(outputs)
            agent.post_step(rewards, dones)
        agent.learn(time_steps)
        env.pool.add(NetworkPolicy.snapshot(agent.network.actor))
    assert agent.updates_done == 2
    assert len(env.pool) == 3
