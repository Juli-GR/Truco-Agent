import numpy as np
import pyspiel
import pytest

import truco.games.truco  # noqa: F401  (registers the game)
from truco.algorithms.abstraction import abstract_key
from truco.games.truco import Action, sample_deal


def decision_states(game, num_games: int, seed: int = 0):
    """Every decision state of `num_games` random games."""
    rng = np.random.default_rng(seed)
    for _ in range(num_games):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        while not state.is_terminal():
            yield state
            state = state.child(int(rng.choice(state.legal_actions())))


@pytest.mark.parametrize("niveles", [5, 14])
def test_key_holds_the_same_information_as_the_full_encoding(niveles):
    game = pyspiel.load_game("truco", {"niveles": niveles, "observacion": "full"})
    key_to_tensor, tensor_to_key = {}, {}
    for state in decision_states(game, 2000):
        player = state.current_player()
        key = abstract_key(state, player)
        tensor = (player, tuple(state.information_state_tensor(player)))
        assert key_to_tensor.setdefault(key, tensor) == tensor
        assert tensor_to_key.setdefault(tensor, key) == key


@pytest.mark.parametrize("niveles", [5, 14])
def test_merged_information_sets_have_the_same_legal_actions(niveles):
    game = pyspiel.load_game("truco", {"niveles": niveles})
    legal = {}
    for state in decision_states(game, 2000):
        key = abstract_key(state, state.current_player())
        assert legal.setdefault(key, state.legal_actions()) == state.legal_actions()


def test_merges_calls_made_in_different_windows_of_the_same_trick():
    game = pyspiel.load_game("truco")
    paso, truco, quiero = int(Action.PASO), int(Action.TRUCO), int(Action.QUIERO)
    deal = sample_deal(game, np.random.default_rng(0))

    def play(actions):
        state = game.new_initial_state()
        for action in deal + actions:
            state.apply_action(action)
        return state

    # Mano calls truco before its first card, or right after it; then the pie has to play.
    card = play([paso, paso]).legal_actions()[0]
    before = play([truco, quiero, paso, card, paso, paso])
    after = play([paso, paso, card, truco, quiero, paso])
    pie = before.current_player()
    assert after.current_player() == pie
    assert before.information_state_string(pie) != after.information_state_string(pie)
    assert abstract_key(before, pie) == abstract_key(after, pie)
