import pyspiel
import pytest

from truco.games.cards import NUM_CARDS, TRUCO_LEVEL_CARDS
from truco.games.truco import (
    Action,
    deal_outcomes,
    hand_winner,
    hands,
    parse_information_state,
)

_NIVELES = 6  # 1e, 1b, 7e, 7o, 3 y 2: mazo de 12 cartas
P, T, R, V = Action.PASO, Action.TRUCO, Action.RETRUCO, Action.VALE_CUATRO
Q, N, M = Action.QUIERO, Action.NO_QUIERO, Action.MAZO
# Niveles: 0 = 1e, 1 = 1b, 2 = 7e, 3 = 7o, 4 = 3, 5 = 2.
FUERTE, DEBIL = (0, 1, 2), (3, 4, 4)


@pytest.fixture
def game():
    import truco.games.truco  # noqa: F401  (registra el juego)

    return pyspiel.load_game("truco", {"niveles": _NIVELES})


def deal(game, mano, h0, h1):
    state = game.new_initial_state()
    state.apply_action(mano)
    state.apply_action(hands(_NIVELES).index(h0))
    state.apply_action(hands(_NIVELES).index(h1))
    return state


def play(game, mano, h0, h1, actions):
    state = deal(game, mano, h0, h1)
    for a in actions:
        state.apply_action(a)
    assert state.is_terminal()
    return state.returns()


# El mano (jugador 0, con FUERTE) gana las dos primeras bazas si nadie canta.
_SIN_CANTOS = [P, P, 0, P, P, 4, P, P, 1, P, P, 4]


def test_hierarchy_covers_deck():
    assert sorted(c for cards in TRUCO_LEVEL_CARDS for c in cards) == list(range(NUM_CARDS))


def test_deal_probabilities():
    outcomes = dict(deal_outcomes(_NIVELES, ()))
    assert sum(outcomes.values()) == pytest.approx(1.0)
    assert outcomes[hands(_NIVELES).index((0, 1, 2))] == pytest.approx(1 / 220)
    assert outcomes[hands(_NIVELES).index((4, 4, 4))] == pytest.approx(4 / 220)


def test_number_of_distinct_deals():
    first = deal_outcomes(_NIVELES, ())
    pairs = sum(len(deal_outcomes(_NIVELES, hands(_NIVELES)[i])) for i, _ in first)
    assert pairs == 518  # contado por fuerza bruta sobre los 18.480 repartos de 3 + 3 cartas


@pytest.mark.parametrize(
    ("results", "mano", "expected"),
    [
        ([0], 0, None),
        ([0, 1], 0, None),
        ([1, 1], 0, 1),
        ([None], 0, None),
        ([None, 1], 0, 1),  # primera parda: define la segunda
        ([0, None], 1, 0),  # segunda parda: gana quien hizo la primera
        ([0, 1, None], 1, 0),  # tercera parda: gana quien hizo la primera
        ([None, None], 0, None),
        ([None, None, 1], 0, 1),  # primera y segunda pardas: define la tercera
        ([None, None, None], 1, 1),  # triple parda: gana el mano
    ],
)
def test_hand_winner(results, mano, expected):
    assert hand_winner(results, mano) == expected


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (_SIN_CANTOS, [1, -1]),
        ([M], [-1, 1]),
        ([T, N], [1, -1]),
        ([T, Q, *_SIN_CANTOS[1:]], [2, -2]),
        ([T, R, N], [-2, 2]),  # el pie responde retruco y el mano no quiere
        ([T, R, V, Q, M], [4, -4]),
    ],
)
def test_returns(game, actions, expected):
    assert play(game, 0, FUERTE, DEBIL, actions) == expected


def test_only_quiero_owner_raises(game):
    state = deal(game, 0, FUERTE, DEBIL)
    for a in [T, Q]:
        state.apply_action(a)
    assert state.current_player() == 1
    assert state.legal_actions() == [P, R, M]
    for a in [P, 0]:
        state.apply_action(a)
    assert state.current_player() == 0
    assert state.legal_actions() == [P, M]


def test_window_between_two_cards_of_rival(game):
    # El pie gana la primera baza y abre la segunda: el mano puede cantar entre sus dos cartas.
    state = deal(game, 0, DEBIL, FUERTE)
    for a in [P, P, 4, P, P, 0, P]:
        state.apply_action(a)
    assert state.current_player() == 0
    assert T in state.legal_actions()
    state.apply_action(P)
    assert state.current_player() == 1
    assert state.legal_actions() == [1, 2]


def test_mano_leads_after_parda(game):
    state = deal(game, 1, (0, 4, 5), (1, 4, 5))
    for a in [P, P, 4, P, P, 4, P, P]:
        state.apply_action(a)
    assert state.results == [None]
    assert state.current_player() == 1


def test_parse_information_state(game):
    state = deal(game, 1, FUERTE, DEBIL)
    for a in [P, T, Q]:
        state.apply_action(a)
    assert parse_information_state(state.information_state_string(0)) == {
        "jugador": 0,
        "es_mano": 0,
        "cartas": "1e,1b,7e",
        "historia": "- T Q",
    }


@pytest.mark.parametrize("niveles", [5, 14])
def test_random_simulations(niveles):
    import truco.games.truco  # noqa: F401  (registra el juego)

    game = pyspiel.load_game("truco", {"niveles": niveles})
    pyspiel.random_sim_test(game, num_sims=50, serialize=False, verbose=False)
