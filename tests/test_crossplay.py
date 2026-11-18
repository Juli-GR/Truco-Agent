import numpy as np
import pyspiel
import pytest

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.truco_rivales import CONSERVADOR, RIVALES, HeuristicPolicy
from truco.evaluation.crossplay import (
    beats,
    crossplay,
    intransitive_triads,
    mean_against_others,
    select_evenly,
)

# A heuristic without bluffs: it plays deterministically.
OTHER_DETERMINISTIC = HeuristicPolicy("other", (0.6, 0.7, 0.8), farol=0.0)


def antisymmetric(upper: dict[tuple[int, int], float], n: int) -> np.ndarray:
    m = np.zeros((n, n))
    for (i, j), value in upper.items():
        m[i, j], m[j, i] = value, -value
    return m


def test_select_evenly_keeps_first_and_last():
    items = list(range(83))
    chosen = select_evenly(items, 5)
    assert chosen[0] == 0 and chosen[-1] == 82
    assert len(chosen) == 5 and chosen == sorted(chosen)
    assert select_evenly(items[:3], 10) == [0, 1, 2]
    with pytest.raises(ValueError):
        select_evenly(items, 1)


def test_rock_paper_scissors_is_one_intransitive_triad():
    means = antisymmetric({(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0}, 3)
    stderrs = np.full((3, 3), 0.1)
    assert intransitive_triads(means, stderrs) == 1
    assert intransitive_triads(-means, stderrs) == 1  # the cycle in the other direction


def test_ranked_policies_have_no_intransitive_triads():
    # Policy i beats every j > i: a strict ranking.
    n = 5
    means = antisymmetric({(i, j): 0.5 for i in range(n) for j in range(i + 1, n)}, n)
    assert intransitive_triads(means, np.full((n, n), 0.1)) == 0


def test_cycles_within_the_error_do_not_count():
    means = antisymmetric({(0, 1): 0.1, (1, 2): 0.1, (2, 0): 0.1}, 3)
    stderrs = np.full((3, 3), 0.1)
    assert not beats(means, stderrs).any()
    assert intransitive_triads(means, stderrs) == 0


def test_mean_against_others_leaves_out_the_diagonal():
    means = antisymmetric({(0, 1): 1.0, (0, 2): 0.5, (1, 2): 0.0}, 3)
    np.testing.assert_allclose(mean_against_others(means), [0.75, -0.5, -0.25])


def test_crossplay_is_antisymmetric():
    game = pyspiel.load_game("truco")
    policies = [RIVALES["aleatorio"], CONSERVADOR, OTHER_DETERMINISTIC]
    means, stderrs = crossplay(game, policies, num_deals=50)
    assert means.shape == stderrs.shape == (3, 3)
    assert np.allclose(means, -means.T)
    assert np.allclose(np.diag(means), 0)
