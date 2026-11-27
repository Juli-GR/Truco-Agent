"""Tabular external-sampling MCCFR on the full hand with envido (truco/games/hand_numba.py).

The same algorithm and tables as truco.algorithms.mccfr (see its docstring), on the hand
simulator, with the envido abstraction (truco/algorithms/hand_abstraction.py). Only the walk over
the hand is specific: it uses the abstract actions, where cards are played by level, so the
information sets merged by a key always have the same actions. The tables, the regret matching and
the sampling come from mccfr.
"""

import numba
import numpy as np
from numba import njit, prange

from truco.algorithms import mccfr
from truco.games import hand_numba as hn
from truco.games.cards import NUM_TRUCO_LEVELS

NUM_ACTIONS = hn.MAX_ACTIONS


def create_tables(capacity_log2: int) -> mccfr.Tables:
    return mccfr.Tables.create(capacity_log2, NUM_ACTIONS)


@njit(cache=True)
def traverse(s, traverser, keys, regrets, strategy_sum, stats):
    """Value of state `s` for `traverser`, updating the tables along the sampled hand."""
    player = hn.current_player(s)
    if player == hn.TERMINAL:
        return float(hn.returns(s, traverser))

    actions = np.empty(NUM_ACTIONS, dtype=np.int64)
    n = hn.abstract_legal_actions(s, actions)
    slot = mccfr.find_or_insert(keys, stats, hn.info_key(s, player))
    strategy = np.empty(NUM_ACTIONS, dtype=np.float64)
    mccfr.regret_matching(regrets, slot, n, strategy)

    if player == traverser:
        values = np.empty(NUM_ACTIONS, dtype=np.float64)
        node_value = 0.0
        for a in range(n):
            child = s.copy()
            hn.apply_abstract_action(child, actions[a])
            values[a] = traverse(child, traverser, keys, regrets, strategy_sum, stats)
            node_value += strategy[a] * values[a]
        if slot >= 0:
            for a in range(n):
                regrets[slot, a] += values[a] - node_value
            stats[mccfr.RESCALES] += mccfr.rescale_if_large(regrets, slot, n)
        return node_value

    if slot >= 0:
        for a in range(n):
            strategy_sum[slot, a] += strategy[a]
        stats[mccfr.RESCALES] += mccfr.rescale_if_large(strategy_sum, slot, n)
    hn.apply_abstract_action(s, actions[mccfr.sample(strategy, n)])
    return traverse(s, traverser, keys, regrets, strategy_sum, stats)


@njit(cache=True)
def run_iterations(first_iteration, num_iterations, seed, tables_arrays):
    """As mccfr.run_iterations, on the hand."""
    keys, regrets, strategy_sum, stats = tables_arrays
    np.random.seed(seed)
    for i in range(num_iterations):
        traverser = (first_iteration + i) % 2
        traverse(hn.deal(), traverser, keys, regrets, strategy_sum, stats)
        if stats[mccfr.FULL]:
            return i + 1
    return num_iterations


@njit(cache=True, parallel=True)
def run_iterations_parallel(first_iteration, num_iterations, chunk_seeds, tables_arrays):
    """As mccfr.run_iterations_parallel, on the hand."""
    keys, regrets, strategy_sum, stats = tables_arrays
    num_chunks = len(chunk_seeds)
    ran = np.zeros(num_chunks, dtype=np.int64)
    for c in prange(num_chunks):
        np.random.seed(chunk_seeds[c])
        start = c * num_iterations // num_chunks
        end = (c + 1) * num_iterations // num_chunks
        for i in range(start, end):
            if stats[mccfr.FULL]:
                break
            traverser = (first_iteration + i) % 2
            traverse(hn.deal(), traverser, keys, regrets, strategy_sum, stats)
            ran[c] += 1
    return ran.sum()


def train(
    tables: mccfr.Tables,
    first_iteration: int,
    num_iterations: int,
    seed: int,
    threads: int = 1,
) -> int:
    """Runs a block of iterations; returns how many ran (fewer only if the table got full). With
    `threads` > 1 they run in parallel without locks, as in mccfr.train."""
    arrays = (tables.keys, tables.regrets, tables.strategy_sum, tables.stats)
    if threads == 1:
        return run_iterations(first_iteration, num_iterations, seed, arrays)
    numba.set_num_threads(threads)
    chunk_seeds = np.random.SeedSequence(seed).generate_state(threads).astype(np.int64)
    return int(run_iterations_parallel(first_iteration, num_iterations, chunk_seeds, arrays))


class HandMCCFRPolicy:
    """Plays the average strategy of the tables on truco_con_envido (truco.agents.policy.Policy
    interface).

    Each OpenSpiel state is converted to the simulator (replaying its history) to compute its key;
    the probability of a level goes to the card that plays it (hand_numba.card_of_level).
    Information sets never visited are played uniformly.
    """

    def __init__(self, tables: mccfr.Tables):
        self.tables = tables

    def action_probabilities(self, states) -> np.ndarray:
        probs = np.zeros((len(states), states[0].num_distinct_actions()))
        actions = np.empty(NUM_ACTIONS, dtype=np.int64)
        out = np.empty(NUM_ACTIONS)
        for i, state in enumerate(states):
            s = hn.from_openspiel(state)
            player = hn.current_player(s)
            n = hn.abstract_legal_actions(s, actions)
            key = hn.info_key(s, player)
            mccfr.average_strategy(self.tables.keys, self.tables.strategy_sum, key, n, out)
            for a in range(n):
                action = actions[a]
                if action < NUM_TRUCO_LEVELS:
                    action = hn.card_of_level(s, player, action)
                probs[i, action] = out[a]
        return probs
