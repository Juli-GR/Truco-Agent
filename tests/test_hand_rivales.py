import numpy as np
import pyspiel
import pytest

import truco.games.hand  # noqa: F401  (registers the game)
from truco.agents.hand_rivales import (
    ENVIDO_CONSERVADOR,
    ENVIDO_EQUILIBRADO,
    HAND_RIVALES,
)
from truco.evaluation.arena import play_match
from truco.games import envido
from truco.games.cards import make_card
from truco.games.hand import Action, hands
from truco.games.truco import sample_deal

E, RE, Q, N, P, T = (
    Action.ENVIDO,
    Action.REAL_ENVIDO,
    Action.QUIERO,
    Action.NO_QUIERO,
    Action.PASO,
    Action.TRUCO,
)
TREINTA_Y_TRES = (make_card(7, "espada"), make_card(6, "espada"), make_card(1, "basto"))
SIETE = (make_card(7, "copa"), make_card(4, "oro"), make_card(12, "basto"))  # tantos 7


@pytest.fixture
def game():
    return pyspiel.load_game("truco_con_envido")


def deal(game, mano, h0, h1):
    state = game.new_initial_state()
    for action in [mano, hands().index(tuple(sorted(h0))), hands().index(tuple(sorted(h1)))]:
        state.apply_action(action)
    return state


def test_more_tantos_call_more_often():
    probs = [ENVIDO_EQUILIBRADO.call_probability(t) for t in (0, 7, 20, 24, 26, 28, 33)]
    assert probs == sorted(probs)
    assert ENVIDO_EQUILIBRADO.call_probability(26) == pytest.approx(0.5)
    assert ENVIDO_EQUILIBRADO.call_probability(30) > 0.85
    assert ENVIDO_EQUILIBRADO.call_probability(0) == ENVIDO_EQUILIBRADO.farol
    (call, p), *_ = ENVIDO_CONSERVADOR.start(33).items()
    assert call == RE and p > 0.9
    assert list(ENVIDO_CONSERVADOR.start(29)) == [E]


@pytest.mark.parametrize(
    ("t", "is_mano", "calls", "legal", "expected"),
    [
        (27, False, ["E"], [E, RE, Q, N], Q),
        (26, False, ["E"], [E, RE, Q, N], N),
        (26, True, ["E"], [E, RE, Q, N], Q),  # the mano wins ties: one tanto less
        (28, False, ["E", "E", "R"], [Q, N], N),  # 7 at stake: needs 27 + 5
        (32, False, ["E", "E", "R"], [Q, N], Q),
        (31, False, ["E"], [E, RE, Q, N], RE),  # raises with real envido
        (31, False, ["E", "E"], [RE, Q, N], RE),
    ],
)
def test_conservador_answers(t, is_mano, calls, legal, expected):
    calls = [{"E": envido.Action.ENVIDO, "R": envido.Action.REAL_ENVIDO}[c] for c in calls]
    assert ENVIDO_CONSERVADOR.answer(t, is_mano, calls, legal) == expected


def test_envido_only_in_the_first_decision(game):
    policy = HAND_RIVALES["agresivo"]
    state = deal(game, 0, TREINTA_Y_TRES, SIETE)
    assert policy.probabilities(state)[RE] > 0.9  # the mano, first decision
    state.apply_action(P)
    assert policy.probabilities(state)[E] == pytest.approx(0.15)  # the pie bluffs with 7
    state.apply_action(P)
    state.apply_action(int(min(state.legal_actions())))  # the mano plays a card
    state.apply_action(P)
    assert E in state.legal_actions()
    assert E not in policy.probabilities(state)  # the pie already passed on the envido


@pytest.mark.parametrize("name", list(HAND_RIVALES))
def test_rivals_play_legal_distributions(game, name):
    policy = HAND_RIVALES[name]
    rng = np.random.default_rng(0)
    states = []
    for _ in range(100):
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


def test_heuristic_beats_random(game):
    result = play_match(game, HAND_RIVALES["equilibrado"], HAND_RIVALES["aleatorio"], 200)
    assert result.mean > 5 * result.stderr
