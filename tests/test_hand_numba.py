from itertools import combinations

import numpy as np
import pyspiel
import pytest

import truco.games.hand  # noqa: F401  (registers the game)
from truco.algorithms.hand_abstraction import (
    abstract_actions,
    abstract_hand_key,
    card_of_level,
    pack_key,
)
from truco.games import hand_numba as hn
from truco.games import truco_numba as tn
from truco.games.cards import NUM_CARDS, NUM_TRUCO_LEVELS, tantos
from truco.games.truco import sample_deal


@pytest.fixture
def game():
    return pyspiel.load_game("truco_con_envido")


def legal(s):
    out = np.empty(hn.MAX_ACTIONS, dtype=np.int64)
    return out[: hn.legal_actions(s, out)].tolist()


def random_games(game, num_games, seed):
    """Yields (OpenSpiel state, simulator state) pairs along random games, at every decision and
    at the end. Envido actions are favored so most games include an envido."""
    rng = np.random.default_rng(seed)
    for _ in range(num_games):
        state = game.new_initial_state()
        for action in sample_deal(game, rng):
            state.apply_action(action)
        s = hn.new_state(state.mano, np.array(state.cards[0]), np.array(state.cards[1]))
        while True:
            yield state, s
            if state.is_terminal():
                break
            actions = state.legal_actions()
            calls = [a for a in actions if a in (hn.ENVIDO, hn.REAL_ENVIDO)]
            action = int(rng.choice(calls if calls and rng.random() < 0.5 else actions))
            state.apply_action(action)
            hn.apply_action(s, action)


def test_tantos_matches_cards():
    for cards in list(combinations(range(NUM_CARDS), 3))[::7]:
        assert hn.tantos(np.array(cards)) == tantos(cards)


def test_plays_like_the_openspiel_game(game):
    for state, s in random_games(game, 1000, seed=0):
        if state.is_terminal():
            assert hn.current_player(s) == hn.TERMINAL
            assert [hn.returns(s, p) for p in (0, 1)] == state.returns()
        else:
            player = state.current_player()
            assert hn.current_player(s) == player
            assert legal(s) == state.legal_actions()
            assert hn.info_key(s, player) == pack_key(abstract_hand_key(state, player))
            out = np.empty(hn.MAX_ACTIONS, dtype=np.int64)
            assert out[: hn.abstract_legal_actions(s, out)].tolist() == abstract_actions(state)
            if out[0] < NUM_TRUCO_LEVELS:
                after = s.copy()
                hn.apply_abstract_action(after, out[0])
                expected = s.copy()
                hn.apply_action(expected, card_of_level(state, player, out[0]))
                np.testing.assert_array_equal(after, expected)
        np.testing.assert_array_equal(hn.from_openspiel(state), s)


def test_exact_key_identifies_the_exact_information_set(game):
    key_to_string, string_to_key = {}, {}
    for state, s in random_games(game, 1000, seed=1):
        if state.is_terminal():
            continue
        player = state.current_player()
        key, string = hn.exact_key(s, player), state.information_state_string(player)
        assert key >= 0
        assert key_to_string.setdefault(key, string) == string
        assert string_to_key.setdefault(string, key) == key


def test_deal_is_uniform():
    tn.seed(0)
    counts = np.zeros(NUM_CARDS)
    manos = 0
    num_deals = 100_000
    for _ in range(num_deals):
        s = hn.deal()
        counts[s[hn.CARDS : hn.CARDS + 6]] += 1
        manos += s[tn.MANO]
        assert len(set(s[hn.CARDS : hn.CARDS + 6])) == 6
    np.testing.assert_allclose(counts / num_deals, 6 / NUM_CARDS, atol=0.005)
    assert abs(manos / num_deals - 0.5) < 0.01
