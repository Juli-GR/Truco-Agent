import numpy as np
import pyspiel
import pytest

import truco.games.hand  # noqa: F401  (registers the game)
import truco.games.truco  # noqa: F401  (registers the game)
from truco.games.cards import NUM_CARDS, NUM_TRUCO_LEVELS, make_card, tantos, truco_level
from truco.games.hand import Action, hands, levels
from truco.games.truco import Action as TrucoAction
from truco.games.truco import hands as truco_hands
from truco.games.truco import sample_deal

P, T, R, V = Action.PASO, Action.TRUCO, Action.RETRUCO, Action.VALE_CUATRO
Q, N, M = Action.QUIERO, Action.NO_QUIERO, Action.MAZO
E, RE = Action.ENVIDO, Action.REAL_ENVIDO

SIETE_E, SEIS_E, UNO_B = make_card(7, "espada"), make_card(6, "espada"), make_card(1, "basto")
CUATRO_C, CINCO_C, DOCE_O = make_card(4, "copa"), make_card(5, "copa"), make_card(12, "oro")
TREINTA_Y_TRES = (SIETE_E, SEIS_E, UNO_B)  # tantos 33, strong for the truco too
VEINTINUEVE = (CUATRO_C, CINCO_C, DOCE_O)  # tantos 29, weak


@pytest.fixture
def game():
    return pyspiel.load_game("truco_con_envido")


def deal(game, mano, h0, h1):
    state = game.new_initial_state()
    state.apply_action(mano)
    state.apply_action(hands().index(tuple(sorted(h0))))
    state.apply_action(hands().index(tuple(sorted(h1))))
    return state


def play(game, mano, h0, h1, actions):
    state = deal(game, mano, h0, h1)
    for a in actions:
        assert a in state.legal_actions(), (a, state)
        state.apply_action(a)
    return state


def test_hands_and_tantos():
    assert len(hands()) == 9880
    assert tantos(TREINTA_Y_TRES) == 33 and tantos(VEINTINUEVE) == 29


def test_deal_probabilities(game):
    state = game.new_initial_state()
    state.apply_action(0)
    first = state.chance_outcomes()
    assert len(first) == 9880 and sum(p for _, p in first) == pytest.approx(1)
    state.apply_action(first[0][0])
    second = state.chance_outcomes()
    assert len(second) == 7770 and sum(p for _, p in second) == pytest.approx(1)


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        # Player 0 is mano with 33 against 29: envido points + truco points.
        ([E, Q, M], [2 - 1, -2 + 1]),  # the mano wins the envido, then goes to the mazo
        ([E, N, M], [1 - 1, -1 + 1]),
        ([RE, N, M], [1 - 1, -1 + 1]),
        ([E, E, RE, Q, M], [7 - 1, -7 + 1]),
        ([E, RE, N, M], [-2 - 1, 2 + 1]),  # the pie raises and the mano rejects
        ([P, E, Q, M], [2 + 1, -2 - 1]),  # the pie calls after passing, then goes to the mazo
        ([T, E, Q, Q, M], [2 + 2, -2 - 2]),  # el envido está primero, then the truco is accepted
        ([T, E, N, N], [-1 + 1, 1 - 1]),  # the mano rejects the envido, the pie the truco
    ],
)
def test_returns(game, actions, expected):
    state = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, actions)
    assert state.is_terminal()
    assert state.returns() == expected


def test_child_states_are_independent_copies(game):
    state = deal(game, 0, TREINTA_Y_TRES, VEINTINUEVE)
    for a in [E, N, M]:
        state = state.child(a)
    assert state.is_terminal()
    assert state.returns() == [1 - 1, -1 + 1]

    parent = deal(game, 0, TREINTA_Y_TRES, VEINTINUEVE)
    parent.child(M)
    assert not parent.is_terminal() and parent.legal_actions() == [P, T, M, E, RE]


def test_tantos_are_declared_as_in_the_rules(game):
    state = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, [E, Q])
    assert state.declared == [33, None]  # the pie says "son buenas"
    assert state.information_state_string(1).endswith("tantos:33/buenas")

    state = play(game, 1, TREINTA_Y_TRES, VEINTINUEVE, [E, Q])  # now player 1 (29) is mano
    assert state.declared == [33, 29]
    assert state.information_state_string(0).endswith("tantos:29/33")

    state = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, [E, N])  # rejected: nothing declared
    assert state.declared == [None, None]


def test_envido_does_not_use_the_chance_in_the_window(game):
    state = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, [E, Q])
    assert state.current_player() == 0
    assert state.legal_actions() == [P, T, M]  # the envido is played only once


def test_envido_only_before_the_players_first_card(game):
    state = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, [P, P, UNO_B])
    assert state.current_player() == 0
    assert state.legal_actions() == [P, T, M]  # the mano already played
    state.apply_action(P)
    assert state.current_player() == 1
    assert state.legal_actions() == [P, T, M, E, RE]  # the pie did not
    for a in [P, CUATRO_C]:
        state.apply_action(a)
    assert E not in state.legal_actions()


def test_el_envido_esta_primero(game):
    state = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, [T])
    assert state.legal_actions() == [R, Q, N, E, RE]
    for a in [E, Q]:
        state.apply_action(a)
    assert state.current_player() == 1  # back to answering the truco
    assert state.legal_actions() == [R, Q, N]


def test_no_envido_after_a_truco_is_accepted(game):
    state = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, [T, Q])
    assert state.current_player() == 1
    assert state.legal_actions() == [P, R, M]


def test_without_envido_it_plays_like_truco(game):
    """With no envido calls, the hand ends as the truco game with the same deal by levels."""
    truco_game = pyspiel.load_game("truco")
    rng = np.random.default_rng(0)
    for _ in range(300):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        reference = truco_game.new_initial_state()
        reference.apply_action(state.mano)
        for hand in state.cards:
            reference.apply_action(truco_hands(NUM_TRUCO_LEVELS).index(levels(hand)))
        while not state.is_terminal():
            action = int(rng.choice([a for a in state.legal_actions() if a not in (E, RE)]))
            state.apply_action(action)
            if action < NUM_CARDS:
                reference.apply_action(truco_level(action))
            else:
                reference.apply_action(TrucoAction[Action(action).name])
        assert reference.is_terminal()
        assert state.returns() == reference.returns()


def test_information_state_hides_the_rival_cards(game):
    a = play(game, 0, TREINTA_Y_TRES, VEINTINUEVE, [E, N])
    b = play(game, 0, TREINTA_Y_TRES, (make_card(4, "oro"), CINCO_C, DOCE_O), [E, N])
    assert a.information_state_string(0) == b.information_state_string(0)
    assert a.information_state_string(1) != b.information_state_string(1)
    np.testing.assert_array_equal(a.information_state_tensor(0), b.information_state_tensor(0))


def test_random_simulations(game):
    pyspiel.random_sim_test(game, num_sims=200, serialize=False, verbose=False)
