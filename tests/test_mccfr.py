import numpy as np
import pyspiel
import pytest

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.truco_rivales import RIVALES
from truco.algorithms import mccfr
from truco.evaluation.arena import play_match
from truco.games import truco_numba as tn
from truco.games.truco import sample_deal

# Information sets of the 10-card deck (experiments/truco_infosets.py, test_abstraction.py).
INFOSETS = {mccfr.EXACT: 654_580, mccfr.ABSTRACT: 178_708}


def test_hash_table_finds_what_it_stores():
    tables = mccfr.Tables.create(12)
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 1 << 62, size=3000)
    slots = [mccfr.find_or_insert(tables.keys, tables.stats, k) for k in keys]
    assert len(set(slots)) == len(keys)
    assert [mccfr.find_or_insert(tables.keys, tables.stats, k) for k in keys] == slots
    assert [mccfr.find(tables.keys, k) for k in keys] == slots
    assert mccfr.find(tables.keys, 12345) == -1
    assert tables.num_entries == len(keys) and not tables.is_full


def test_hash_table_reports_when_full():
    tables = mccfr.Tables.create(4)  # 16 slots, full at 90 %
    for k in range(100):
        mccfr.find_or_insert(tables.keys, tables.stats, k)
    assert tables.is_full
    assert tables.num_entries == 15


@pytest.mark.parametrize("key_mode", [mccfr.EXACT, mccfr.ABSTRACT])
def test_training_visits_only_real_information_sets(key_mode):
    tables = mccfr.Tables.create(21)
    ran = mccfr.train(tables, tn.make_deck(5), 0, 20_000, 0, key_mode)
    assert ran == 20_000
    assert 0 < tables.num_entries <= INFOSETS[key_mode]

    # Every stored average strategy is a probability distribution.
    out = np.empty(tn.MAX_ACTIONS)
    for key in tables.keys[tables.keys != mccfr.EMPTY][:1000]:
        mccfr.average_strategy(tables.keys, tables.strategy_sum, key, 2, out)
        assert out[:2].sum() == pytest.approx(1)
        assert (out[:2] >= 0).all()


def test_parallel_training_runs_every_iteration():
    tables = mccfr.Tables.create(21)
    ran = mccfr.train(tables, tn.make_deck(5), 0, 20_000, 0, mccfr.ABSTRACT, threads=4)
    assert ran == 20_000
    assert 0 < tables.num_entries <= INFOSETS[mccfr.ABSTRACT]
    assert (tables.strategy_sum >= 0).all()


def test_save_and_load(tmp_path):
    tables = mccfr.Tables.create(16)
    mccfr.train(tables, tn.make_deck(5), 0, 500, 3, mccfr.ABSTRACT)
    tables.save(tmp_path / "ckpt")
    loaded = mccfr.Tables.load(tmp_path / "ckpt")
    for name in ("keys", "regrets", "strategy_sum", "stats"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(tables, name))
        assert getattr(loaded, name).dtype == getattr(tables, name).dtype


def test_policy_plays_legal_actions_and_beats_random(tmp_path):
    game = pyspiel.load_game("truco", {"niveles": 5})
    trained = mccfr.Tables.create(21)
    mccfr.train(trained, tn.make_deck(5), 0, 100_000, 0, mccfr.ABSTRACT)
    trained.save(tmp_path / "ckpt")
    policy = mccfr.MCCFRPolicy(mccfr.Tables.load(tmp_path / "ckpt", read_only=True), mccfr.ABSTRACT)

    rng = np.random.default_rng(0)
    states = []
    for _ in range(300):
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

    result = play_match(game, policy, RIVALES["aleatorio"], 1000)
    assert result.mean > 5 * result.stderr


def test_training_is_reproducible():
    results = []
    for _ in range(2):
        tables = mccfr.Tables.create(18)
        mccfr.train(tables, tn.make_deck(5), 0, 2000, 7, mccfr.ABSTRACT)
        results.append((tables.keys.copy(), tables.regrets.copy()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
