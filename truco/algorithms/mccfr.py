"""Tabular external-sampling MCCFR on the numba truco simulator (truco/games/truco_numba.py).

Each iteration deals at random and walks the hand for one player, the traverser (alternating
between iterations). At the traverser's nodes it tries every legal action; at the opponent's nodes
it samples a single action from the current strategy. On the way back it adds to each visited
information set of the traverser the regret of every action (its value minus the value of the
node), and to each visited information set of the opponent the current strategy (the average
strategy, the one that approaches a Nash equilibrium; Lanctot et al., 2009).

The current strategy comes from the regrets by regret matching: each action is played in
proportion to its positive cumulative regret, uniformly if none is positive.

Information sets are identified by a key: the abstraction (`ABSTRACT`, see
truco/algorithms/abstraction.py) or the exact information set (`EXACT`, only to validate on small
decks). The keys are found while playing, so the tables are an open-addressing hash table on numpy
arrays (numba's typed dict uses too much memory for ~6.5e8 keys): `Tables`.

Tables are float32 to halve the memory. A float32 stops absorbing small additions once it is
large (around 1.6e7 for additions of 1), and the information sets near the root are visited on
every iteration; so when a value of a row passes RESCALE_AT the whole row is halved. That keeps the
row's strategy (only ratios matter) and gives slightly more weight to later iterations, as in
discounted CFR.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from truco.games import truco_numba as tn

ABSTRACT, EXACT = 0, 1
KEY_MODES = {"abstract": ABSTRACT, "exact": EXACT}
EMPTY = -1
MAX_LOAD = 0.9  # fraction of the slots that may be used before the table counts as full
RESCALE_AT = 1e6

# Counters in Tables.stats.
ENTRIES = 0  # keys stored
FULL = 1  # 1 once a key could not be stored (the run must stop: the results are wrong)
RESCALES = 2  # rows halved


@dataclass
class Tables:
    """Regrets and average strategy of every information set seen, by key."""

    keys: np.ndarray  # int64, one per slot, EMPTY if free
    regrets: np.ndarray  # float32 (slots, MAX_ACTIONS): cumulative regret of each action
    strategy_sum: np.ndarray  # float32 (slots, MAX_ACTIONS): average strategy, not normalized
    stats: np.ndarray  # int64 counters, indices ENTRIES, FULL, RESCALES

    @classmethod
    def create(cls, capacity_log2: int) -> "Tables":
        slots = 1 << capacity_log2
        return cls(
            keys=np.full(slots, EMPTY, dtype=np.int64),
            regrets=np.zeros((slots, tn.MAX_ACTIONS), dtype=np.float32),
            strategy_sum=np.zeros((slots, tn.MAX_ACTIONS), dtype=np.float32),
            stats=np.zeros(3, dtype=np.int64),
        )

    @staticmethod
    def bytes_per_slot() -> int:
        return 8 + 2 * 4 * tn.MAX_ACTIONS

    @property
    def num_entries(self) -> int:
        return int(self.stats[ENTRIES])

    @property
    def load(self) -> float:
        return self.num_entries / len(self.keys)

    @property
    def is_full(self) -> bool:
        return bool(self.stats[FULL])

    def nbytes(self) -> int:
        return self.keys.nbytes + self.regrets.nbytes + self.strategy_sum.nbytes


@njit(cache=True)
def _slot(key, mask):
    """First slot to probe for `key` (splitmix64 finalizer, so similar keys spread out)."""
    x = key
    x ^= x >> 31
    x *= -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
    x ^= x >> 29
    return x & mask


@njit(cache=True)
def find(keys, key):
    """Slot of `key`, or -1 if it is not in the table."""
    mask = len(keys) - 1
    slot = _slot(key, mask)
    while True:
        if keys[slot] == key:
            return slot
        if keys[slot] == EMPTY:
            return -1
        slot = (slot + 1) & mask


@njit(cache=True)
def find_or_insert(keys, stats, key):
    """Slot of `key`, inserting it if new; -1 (and stats[FULL] = 1) if the table is full."""
    mask = len(keys) - 1
    slot = _slot(key, mask)
    while True:
        if keys[slot] == key:
            return slot
        if keys[slot] == EMPTY:
            if stats[ENTRIES] >= MAX_LOAD * len(keys):
                stats[FULL] = 1
                return -1
            keys[slot] = key
            stats[ENTRIES] += 1
            return slot
        slot = (slot + 1) & mask


@njit(cache=True)
def key_of(s, player, key_mode):
    if key_mode == EXACT:
        return tn.exact_key(s, player)
    return tn.info_key(s, player)


@njit(cache=True)
def _regret_matching(regrets, slot, n, out):
    """Current strategy of the information set in `slot` (uniform if it is not stored)."""
    total = 0.0
    if slot >= 0:
        for a in range(n):
            positive = max(regrets[slot, a], 0.0)
            out[a] = positive
            total += positive
    if total > 0:
        for a in range(n):
            out[a] /= total
    else:
        for a in range(n):
            out[a] = 1.0 / n


@njit(cache=True)
def _rescale_if_large(table, slot, n):
    largest = 0.0
    for a in range(n):
        largest = max(largest, abs(table[slot, a]))
    if largest > RESCALE_AT:
        for a in range(n):
            table[slot, a] *= 0.5
        return 1
    return 0


@njit(cache=True)
def _sample(probs, n):
    r = np.random.random()
    cumulative = 0.0
    for a in range(n - 1):
        cumulative += probs[a]
        if r < cumulative:
            return a
    return n - 1


@njit(cache=True)
def traverse(s, traverser, keys, regrets, strategy_sum, stats, key_mode):
    """Value of state `s` for `traverser`, updating the tables along the sampled hand."""
    player = tn.current_player(s)
    if player == tn.TERMINAL:
        return float(tn.returns(s, traverser))

    actions = np.empty(tn.MAX_ACTIONS, dtype=np.int64)
    n = tn.legal_actions(s, actions)
    slot = find_or_insert(keys, stats, key_of(s, player, key_mode))
    strategy = np.empty(tn.MAX_ACTIONS, dtype=np.float64)
    _regret_matching(regrets, slot, n, strategy)

    if player == traverser:
        values = np.empty(tn.MAX_ACTIONS, dtype=np.float64)
        node_value = 0.0
        for a in range(n):
            child = s.copy()
            tn.apply_action(child, actions[a])
            values[a] = traverse(child, traverser, keys, regrets, strategy_sum, stats, key_mode)
            node_value += strategy[a] * values[a]
        if slot >= 0:
            for a in range(n):
                regrets[slot, a] += values[a] - node_value
            stats[RESCALES] += _rescale_if_large(regrets, slot, n)
        return node_value

    if slot >= 0:
        for a in range(n):
            strategy_sum[slot, a] += strategy[a]
        stats[RESCALES] += _rescale_if_large(strategy_sum, slot, n)
    tn.apply_action(s, actions[_sample(strategy, n)])
    return traverse(s, traverser, keys, regrets, strategy_sum, stats, key_mode)


@njit(cache=True)
def run_iterations(deck, first_iteration, num_iterations, seed, tables_arrays, key_mode):
    """Runs iterations first_iteration .. first_iteration + num_iterations - 1.

    numba's random generator cannot be saved, so it is seeded here: a resumed run that repeats
    the same blocks with the same seeds gets the same deals and samples.
    """
    keys, regrets, strategy_sum, stats = tables_arrays
    np.random.seed(seed)
    for i in range(num_iterations):
        traverser = (first_iteration + i) % 2
        traverse(tn.deal(deck), traverser, keys, regrets, strategy_sum, stats, key_mode)
        if stats[FULL]:
            return i + 1
    return num_iterations


def train(
    tables: Tables,
    deck: np.ndarray,
    first_iteration: int,
    num_iterations: int,
    seed: int,
    key_mode: int,
) -> int:
    """Runs a block of iterations; returns how many ran (fewer only if the table got full)."""
    arrays = (tables.keys, tables.regrets, tables.strategy_sum, tables.stats)
    return run_iterations(deck, first_iteration, num_iterations, seed, arrays, key_mode)


@njit(cache=True)
def average_strategy(keys, strategy_sum, key, n, out):
    """Average strategy of the information set with `key` and `n` legal actions, into `out`
    (uniform if it was never visited)."""
    slot = find(keys, key)
    total = 0.0
    if slot >= 0:
        for a in range(n):
            out[a] = strategy_sum[slot, a]
            total += out[a]
    if total > 0:
        for a in range(n):
            out[a] /= total
    else:
        for a in range(n):
            out[a] = 1.0 / n
