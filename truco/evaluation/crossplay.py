"""Cross-play matrix between policies, and its analysis.

M[i, j] is the points per hand of policy i against policy j, so M = -M^T. Between checkpoints of
one training run, it shows whether training improved steadily, converged or cycled.
"""

from itertools import combinations

import numpy as np
import pyspiel

from truco.agents.policy import Policy
from truco.evaluation.arena import play_match


def crossplay(
    game: pyspiel.Game,
    policies: list[Policy],
    num_deals: int,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-play matrix: M[i, j] = points per hand of policies[i] against policies[j].

    Returns (means, standard errors). Every pair i < j is played with `num_deals` duplicate deals
    (`arena.play_match`); the rest follows from the game being zero-sum: M[j, i] = -M[i, j]. With
    `verbose`, prints the progress row by row.
    """
    n = len(policies)
    means, stderrs = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            result = play_match(game, policies[i], policies[j], num_deals, seed + i * n + j)
            means[i, j], means[j, i] = result.mean, -result.mean
            stderrs[i, j] = stderrs[j, i] = result.stderr
        if verbose:
            print(f"crossplay: row {i + 1}/{n} done", flush=True)
    return means, stderrs


def select_evenly(items: list, count: int) -> list:
    """`count` items evenly spaced along `items`, always including the first and the last."""
    if count >= len(items):
        return list(items)
    if count < 2:
        raise ValueError("count must be at least 2")
    indices = np.unique(np.round(np.linspace(0, len(items) - 1, count)).astype(int))
    return [items[i] for i in indices]


def beats(means: np.ndarray, stderrs: np.ndarray, z: float = 2.0) -> np.ndarray:
    """B[i, j] = True if i beats j by more than `z` standard errors."""
    return means > z * stderrs


def intransitive_triads(means: np.ndarray, stderrs: np.ndarray, z: float = 2.0) -> int:
    """Number of triples {a, b, c} where a beats b, b beats c and c beats a (rock-paper-scissors),
    each by more than `z` standard errors. It is 0 if the policies can be ranked."""
    b = beats(means, stderrs, z)
    return sum(
        (b[i, j] and b[j, k] and b[k, i]) or (b[j, i] and b[k, j] and b[i, k])
        for i, j, k in combinations(range(len(means)), 3)
    )


def mean_against_others(means: np.ndarray) -> np.ndarray:
    """Points per hand of each policy against all the others (the diagonal is left out)."""
    n = len(means)
    return means.sum(axis=1) / (n - 1)
