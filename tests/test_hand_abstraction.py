import numpy as np
import pyspiel
import pytest

import truco.games.hand  # noqa: F401  (registers the game)
from truco.algorithms.abstraction import NONE
from truco.algorithms.hand_abstraction import (
    RESOLVED,
    RIVAL_30,
    abstract_actions,
    abstract_hand_key,
    card_of_level,
    pack_key,
)
from truco.games.cards import NUM_TRUCO_LEVELS, make_card, truco_level
from truco.games.hand import Action, hands
from truco.games.truco import sample_deal

P, Q, N, E = Action.PASO, Action.QUIERO, Action.NO_QUIERO, Action.ENVIDO
TREINTA_Y_TRES = (make_card(7, "espada"), make_card(6, "espada"), make_card(1, "basto"))
VEINTINUEVE = (make_card(4, "copa"), make_card(5, "copa"), make_card(12, "oro"))


@pytest.fixture
def game():
    return pyspiel.load_game("truco_con_envido")


def decision_states(game, num_games, seed=0):
    """Every decision state of random games, favoring the envido calls."""
    rng = np.random.default_rng(seed)
    for _ in range(num_games):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        while not state.is_terminal():
            yield state
            actions = state.legal_actions()
            calls = [a for a in actions if a in (Action.ENVIDO, Action.REAL_ENVIDO)]
            state = state.child(int(rng.choice(calls if calls and rng.random() < 0.5 else actions)))


def play(game, actions, mano=0, h0=TREINTA_Y_TRES, h1=VEINTINUEVE):
    state = game.new_initial_state()
    deal = [mano, hands().index(tuple(sorted(h0))), hands().index(tuple(sorted(h1)))]
    for action in [*deal, *actions]:
        state.apply_action(action)
    return state


def test_merged_information_sets_have_the_same_abstract_actions(game):
    legal = {}
    for state in decision_states(game, 1500):
        key = abstract_hand_key(state, state.current_player())
        actions = abstract_actions(state)
        assert legal.setdefault(key, actions) == actions


def test_card_of_level_plays_a_card_of_that_level(game):
    for state in decision_states(game, 300, seed=2):
        actions = abstract_actions(state)
        if actions[0] < NUM_TRUCO_LEVELS:
            player = state.current_player()
            for level in actions:
                card = card_of_level(state, player, level)
                assert card in state.legal_actions() and truco_level(card) == level


def test_different_keys_pack_differently(game):
    packed = {}
    for state in decision_states(game, 1500, seed=1):
        key = abstract_hand_key(state, state.current_player())
        assert packed.setdefault(pack_key(key), key) == key


def test_rival_declaring_30_or_more_is_remembered(game):
    state = play(game, [E, Q])  # the mano declares 33, the pie says son buenas
    assert abstract_hand_key(state, 1)[-2:] == (RIVAL_30, NONE)
    assert abstract_hand_key(state, 0)[-2:] == (RESOLVED, NONE)


def test_forgets_who_called_and_who_won_a_rejected_envido(game):
    card = min(TREINTA_Y_TRES)
    rival_card = min(VEINTINUEVE)
    mano_calls = play(game, [E, N, P, P, card, P, P, rival_card])  # the mano wins 1
    pie_calls = play(game, [P, E, N, P, card, P, P, rival_card])  # the pie wins 1
    player = mano_calls.current_player()
    assert pie_calls.current_player() == player
    assert mano_calls.information_state_string(player) != pie_calls.information_state_string(player)
    assert abstract_hand_key(mano_calls, player) == abstract_hand_key(pie_calls, player)


def test_tantos_only_while_the_envido_is_open(game):
    state = play(game, [])
    assert abstract_hand_key(state, 0)[-1] != NONE  # the mano may call
    state = play(game, [P, P, min(TREINTA_Y_TRES), P])
    assert abstract_hand_key(state, 1)[-1] != NONE  # the pie has not played a card yet
    state = play(game, [P, P, min(TREINTA_Y_TRES), P, P, min(VEINTINUEVE)])
    assert abstract_hand_key(state, state.current_player())[-1] == NONE
