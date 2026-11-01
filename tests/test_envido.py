from math import comb

import numpy as np
import pyspiel
import pytest

from truco.games.cards import RANKS, tantos, tantos_joint_counts
from truco.games.envido import Action

_NUM_TANTOS = 34
E, R, Q, N, P = Action.ENVIDO, Action.REAL_ENVIDO, Action.QUIERO, Action.NO_QUIERO, Action.PASO


def card(rank: int, suit: int) -> int:
    return suit * len(RANKS) + RANKS.index(rank)


@pytest.mark.parametrize(
    ("hand", "expected"),
    [
        ((card(7, 0), card(6, 0), card(1, 1)), 33),
        ((card(1, 0), card(12, 0), card(5, 1)), 21),
        ((card(10, 0), card(11, 0), card(4, 2)), 20),
        ((card(7, 0), card(10, 1), card(1, 2)), 7),
        ((card(10, 0), card(11, 1), card(12, 2)), 0),
        ((card(7, 0), card(6, 0), card(5, 0)), 33),
    ],
)
def test_tantos(hand, expected):
    assert tantos(hand) == expected


def test_tantos_joint_counts():
    counts = tantos_joint_counts()
    assert counts.sum() == comb(40, 3) * comb(37, 3)
    assert np.array_equal(counts, counts.T)
    assert counts[8:20].sum() == 0 and counts[:, 8:20].sum() == 0


@pytest.fixture
def game():
    import truco.games.envido  # noqa: F401  (registra el juego)

    return pyspiel.load_game("truco_envido")


def play(game, mano, t0, t1, actions):
    state = game.new_initial_state()
    state.apply_action(mano)
    state.apply_action(t0 * _NUM_TANTOS + t1)
    for a in actions:
        state.apply_action(a)
    assert state.is_terminal()
    return state.returns()


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        ([P, P], [0, 0]),
        ([E, Q], [2, -2]),
        ([E, N], [1, -1]),
        ([R, Q], [3, -3]),
        ([E, E, Q], [4, -4]),
        ([E, E, N], [-2, 2]),  # canta último el jugador 1 y gana lo acumulado
        ([E, R, Q], [5, -5]),
        ([E, E, R, Q], [7, -7]),
        ([E, E, R, N], [4, -4]),
        ([P, E, N], [-1, 1]),  # el pie canta y el mano no quiere
    ],
)
def test_returns(game, actions, expected):
    # Mano = jugador 0, que tiene más tantos.
    assert play(game, mano=0, t0=30, t1=25, actions=actions) == expected


def test_tie_goes_to_mano(game):
    assert play(game, mano=1, t0=27, t1=27, actions=[E, Q]) == [-2, 2]


def test_legal_actions_after_real_envido(game):
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(30 * _NUM_TANTOS + 25)
    state.apply_action(E)
    state.apply_action(R)
    assert state.legal_actions() == [Q, N]


def test_random_simulations(game):
    pyspiel.random_sim_test(game, num_sims=20, serialize=False, verbose=False)
