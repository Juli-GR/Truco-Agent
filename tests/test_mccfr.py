import numpy as np
import pytest

from truco.algorithms import mccfr
from truco.games import truco_numba as tn

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


def test_training_is_reproducible():
    results = []
    for _ in range(2):
        tables = mccfr.Tables.create(18)
        mccfr.train(tables, tn.make_deck(5), 0, 2000, 7, mccfr.ABSTRACT)
        results.append((tables.keys.copy(), tables.regrets.copy()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
