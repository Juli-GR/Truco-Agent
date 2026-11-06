import numpy as np
import pyspiel
import pytest

import truco.games.truco  # noqa: F401  (registra el juego)
from truco.agents.truco_rivales import (
    RIVALES,
    View,
    actors,
    choose_card,
    equity,
    hand_percentile,
    playout,
)
from truco.games.truco import Action, hands

P, T, Q = Action.PASO, Action.TRUCO, Action.QUIERO
# Niveles: 0 = 1e, 1 = 1b, 2 = 7e, 3 = 7o, 4 = 3, 5 = 2, 6 = 1 falso, ..., 13 = 4.


def deal(game, mano, h0, h1):
    state = game.new_initial_state()
    state.apply_action(mano)
    state.apply_action(hands(14).index(h0))
    state.apply_action(hands(14).index(h1))
    return state


@pytest.fixture
def game():
    return pyspiel.load_game("truco", {"niveles": 14})


@pytest.mark.parametrize(
    ("remaining", "rival_card", "results", "expected"),
    [
        ((0, 4, 13), None, (), 4),  # abre la primera con la carta media
        ((0, 13), None, (0,), 13),  # ganó la primera: tira la más baja
        ((0, 13), None, (None,), 0),  # después de parda: tira la más alta
        ((0, 4, 13), 5, (), 4),  # mata con la más baja que gana
        ((6, 13), 4, (), 13),  # no puede ganar: tira la más baja
        ((4, 13), 4, (0,), 4),  # pardar después de ganar la primera gana la mano
        ((0, 4, 13), 4, (), 4),  # parda con el 3 y guarda el 1e en vez de gastarlo
        ((1, 7, 13), 7, (), 7),  # parda con el 12 y guarda el 1b
        ((1, 7, 13), 9, (), 7),  # sin parda posible: mata con la más baja que gana
        ((5, 7, 13), 5, (), 13),  # no parda si no le queda una carta buena: tira la más baja
    ],
)
def test_choose_card(remaining, rival_card, results, expected):
    assert choose_card(remaining, rival_card, results, me=0, mano=0) == expected


def test_playout_strongest_hand_wins():
    assert playout((0, 1, 2), (11, 12, 13), None, (), 0, 0, 0) == 0
    assert playout((0, 1, 2), (11, 12, 13), None, (), 1, 0, 1) == 0


def test_hand_percentile_extremes():
    assert hand_percentile(14, (0, 1, 2)) == pytest.approx(1.0)
    assert hand_percentile(14, (13, 13, 13)) < 0.01


def test_actors_match_real_players(game):
    state = deal(game, 1, (0, 4, 13), (1, 5, 12))
    real = []
    for a in [P, T, Q, 1, P, P, 0]:
        real.append(state.current_player())
        state.apply_action(a)
    assert actors(state) == real


def test_equity_and_reading(game):
    state = deal(game, 0, (0, 1, 2), (11, 12, 13))
    assert equity(View.of(state), 1.0) > 0.99
    # Con una mano media, un truco del rival baja la equity si se le cree.
    state = deal(game, 0, (4, 7, 9), (0, 1, 13))
    state.apply_action(P)
    state.apply_action(T)
    view = View.of(state)
    assert view.rival_aggressions == 1
    assert equity(view, 1.0) < equity(view, 0.0)


@pytest.mark.parametrize("nombre", list(RIVALES))
def test_rivals_play_legal_actions(game, nombre):
    policy = RIVALES[nombre]
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes(), strict=True)
                state.apply_action(rng.choice(outcomes, p=probs))
                continue
            probs = policy.action_probabilities([state])[0]
            assert probs.sum() == pytest.approx(1.0)
            assert set(np.flatnonzero(probs)) <= set(state.legal_actions())
            assert Action.MAZO not in np.flatnonzero(probs) or nombre == "aleatorio"
            state.apply_action(rng.choice(len(probs), p=probs))
