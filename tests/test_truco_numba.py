import numpy as np
import pyspiel
import pytest

import truco.games.truco  # noqa: F401  (registers the game)
from truco.algorithms.abstraction import abstract_key, pack_key
from truco.games import truco_numba as tn
from truco.games.truco import deal_outcomes, hands, sample_deal


def numba_state(state):
    """Simulator state with the same deal as the OpenSpiel `state` (right after the deal)."""
    return tn.new_state(state.mano, np.array(state.hands[0]), np.array(state.hands[1]))


def legal(s):
    out = np.empty(tn.MAX_ACTIONS, dtype=np.int64)
    return out[: tn.legal_actions(s, out)].tolist()


@pytest.mark.parametrize("niveles", [5, 14])
def test_plays_like_the_openspiel_game(niveles):
    game = pyspiel.load_game("truco", {"niveles": niveles})
    rng = np.random.default_rng(0)
    for _ in range(3000):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        s = numba_state(state)
        while not state.is_terminal():
            player = state.current_player()
            assert tn.current_player(s) == player
            assert legal(s) == state.legal_actions()
            assert tn.info_key(s, player) == pack_key(abstract_key(state, player))
            action = int(rng.choice(state.legal_actions()))
            state.apply_action(action)
            tn.apply_action(s, action)
        assert tn.current_player(s) == tn.TERMINAL
        assert [tn.returns(s, p) for p in (0, 1)] == state.returns()
        np.testing.assert_array_equal(tn.from_openspiel(state), s)


@pytest.mark.parametrize("niveles", [5, 14])
def test_exact_key_identifies_the_exact_information_set(niveles):
    game = pyspiel.load_game("truco", {"niveles": niveles})
    rng = np.random.default_rng(2)
    key_to_string, string_to_key = {}, {}
    for _ in range(2000):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        s = numba_state(state)
        while not state.is_terminal():
            player = state.current_player()
            key, string = tn.exact_key(s, player), state.information_state_string(player)
            assert key >= 0
            assert key_to_string.setdefault(key, string) == string
            assert string_to_key.setdefault(string, key) == key
            action = int(rng.choice(state.legal_actions()))
            state.apply_action(action)
            tn.apply_action(s, action)


def test_different_keys_pack_differently():
    game = pyspiel.load_game("truco", {"niveles": 5})
    rng = np.random.default_rng(1)
    packed = {}
    for _ in range(2000):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        while not state.is_terminal():
            key = abstract_key(state, state.current_player())
            assert packed.setdefault(pack_key(key), key) == key
            state.apply_action(int(rng.choice(state.legal_actions())))


def test_deal_has_the_game_probabilities():
    niveles, num_deals = 5, 200_000
    deck = tn.make_deck(niveles)
    tn.seed(0)
    all_hands = hands(niveles)
    counts = np.zeros(len(all_hands))
    manos = 0
    for _ in range(num_deals):
        s = tn.deal(deck)
        counts[all_hands.index(tuple(s[tn.REMAINING : tn.REMAINING + 3]))] += 1
        manos += s[tn.MANO]
    expected = np.zeros(len(all_hands))
    for i, p in deal_outcomes(niveles, ()):
        expected[i] = p
    np.testing.assert_allclose(counts / num_deals, expected, atol=0.005)
    assert abs(manos / num_deals - 0.5) < 0.005
