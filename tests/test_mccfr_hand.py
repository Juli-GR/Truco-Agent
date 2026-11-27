import numpy as np
import pyspiel
import pytest

import truco.games.hand  # noqa: F401  (registers the game)
from truco.agents.hand_rivales import HAND_RIVALES
from truco.algorithms import mccfr, mccfr_hand
from truco.evaluation.arena import play_match
from truco.games.truco import sample_deal


def test_training_fills_the_tables_with_strategies():
    tables = mccfr_hand.create_tables(22)
    assert tables.regrets.shape == (1 << 22, mccfr_hand.NUM_ACTIONS)
    ran = mccfr_hand.train(tables, 0, 3000, 0)
    assert ran == 3000
    assert 0 < tables.num_entries and not tables.is_full
    assert (tables.strategy_sum >= 0).all()


def test_parallel_training_runs_every_iteration():
    tables = mccfr_hand.create_tables(22)
    assert mccfr_hand.train(tables, 0, 3000, 0, threads=4) == 3000
    assert tables.num_entries > 0


def test_training_is_reproducible():
    results = []
    for _ in range(2):
        tables = mccfr_hand.create_tables(18)
        mccfr_hand.train(tables, 0, 1000, 7)
        results.append((tables.keys.copy(), tables.regrets.copy()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_policy_plays_legal_actions_and_beats_random(tmp_path):
    game = pyspiel.load_game("truco_con_envido")
    trained = mccfr_hand.create_tables(25)
    assert mccfr_hand.train(trained, 0, 100_000, 0, threads=4) == 100_000
    trained.save(tmp_path / "ckpt")
    policy = mccfr_hand.HandMCCFRPolicy(mccfr.Tables.load(tmp_path / "ckpt", read_only=True))

    rng = np.random.default_rng(0)
    states = []
    for _ in range(200):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        while not state.is_terminal():
            states.append(state)
            state = state.child(int(rng.choice(state.legal_actions())))
    probs = policy.action_probabilities(states)
    for state, row in zip(states, probs, strict=True):
        assert row.sum() == pytest.approx(1)
        assert set(np.flatnonzero(row)) <= set(state.legal_actions())

    result = play_match(game, policy, HAND_RIVALES["aleatorio"], 500)
    assert result.mean > 5 * result.stderr
