import numpy as np
import pyspiel
import pytest

import truco.games.truco  # noqa: F401  (registers the game)
from truco.games.cards import card_score
from truco.games.observations import encode
from truco.games.truco import OBSERVACIONES, Action, hands, sample_deal

PASO, TRUCO, QUIERO = int(Action.PASO), int(Action.TRUCO), int(Action.QUIERO)
LEVEL = {"1e": 0, "7o": 3, "3": 4, "2": 5, "10": 9, "4": 13}


@pytest.fixture(scope="module")
def game():
    return pyspiel.load_game("truco")


def deal(game, mano, hand0, hand1):
    state = game.new_initial_state()
    state.apply_action(mano)
    all_hands = hands(game.niveles)
    for hand in (hand0, hand1):
        state.apply_action(all_hands.index(tuple(sorted(LEVEL[c] for c in hand))))
    return state


def example(game, rival_hand=("7o", "2", "10")):
    """Player 0 is mano with 1e, 3, 4. Player 0 passes, player 1 calls truco, player 0 accepts and
    plays the 3, both pass, player 1 plays the 10 (player 0 wins the first trick) and passes."""
    state = deal(game, 0, ("1e", "3", "4"), rival_hand)
    for action in [PASO, TRUCO, QUIERO, LEVEL["3"], PASO, PASO, LEVEL["10"], PASO]:
        state.apply_action(action)
    assert state.current_player() == 0
    return state


def test_full_is_the_default():
    assert pyspiel.load_game("truco").observacion == "full"


def test_unknown_observacion_raises():
    with pytest.raises(ValueError, match="observacion"):
        pyspiel.load_game("truco", {"observacion": "tiny"})


@pytest.mark.parametrize("observacion", ["compact", "full"])
def test_the_game_returns_the_encoding(observacion):
    game = pyspiel.load_game("truco", {"observacion": observacion})
    state = example(game)
    for player in (0, 1):
        expected = encode(state, player, observacion)
        np.testing.assert_array_equal(state.information_state_tensor(player), expected)
        np.testing.assert_array_equal(state.observation_tensor(player), expected)
    assert game.information_state_tensor_size() == len(expected)


def test_strings_are_the_exact_information_set_whatever_the_encoding():
    strings = set()
    for observacion in OBSERVACIONES:
        state = example(pyspiel.load_game("truco", {"observacion": observacion}))
        strings.add((state.information_state_string(0), state.information_state_string(1)))
    assert len(strings) == 1


@pytest.mark.parametrize("observacion", OBSERVACIONES)
def test_tensors_exist_before_the_deal_and_at_the_end(observacion):
    game = pyspiel.load_game("truco", {"observacion": observacion})
    state = game.new_initial_state()
    size = game.information_state_tensor_size()
    assert len(state.information_state_tensor(0)) == size  # chance node, no cards yet
    rng = np.random.default_rng(0)
    for action in sample_deal(game, rng):
        state.apply_action(action)
    while not state.is_terminal():
        state.apply_action(int(rng.choice(state.legal_actions())))
    assert len(state.information_state_tensor(1)) == size


def test_compact_encoding_of_the_example(game):
    x = encode(example(game), 0, "compact")
    expected = np.zeros(56, dtype=np.float32)
    expected[0] = 1  # mano
    expected[1] = 1  # won the first trick
    expected[10 + 1] = 1  # the hand is worth 2
    expected[17 + 1] = 1  # I hold the quiero
    expected[20 + 1] = 1  # truco called by the opponent...
    expected[22 + 0] = 1  # ...in the first trick
    expected[35 + 1] = 1  # phase: call or pass
    cards = [(1, card_score(0)), (1, card_score(13)), (0, 0)]  # remaining: 1e, 4
    cards += [(1, card_score(4)), (1, card_score(9))]  # first trick: my 3, opponent's 10
    for slot, (present, score) in enumerate(cards):
        expected[38 + 2 * slot : 40 + 2 * slot] = present, score
    np.testing.assert_allclose(x, expected)


def test_full_encoding_adds_one_hot_cards(game):
    x = encode(example(game), 0, "full")
    np.testing.assert_array_equal(x[:56], encode(example(game), 0, "compact"))
    one_hot = x[56:].reshape(7, 14)
    assert np.flatnonzero(one_hot[0]).tolist() == [0, 13]  # remaining: 1e, 4
    assert np.flatnonzero(one_hot[1]).tolist() == [4]  # first trick, mine: 3
    assert np.flatnonzero(one_hot[2]).tolist() == [9]  # first trick, opponent's: 10
    assert not one_hot[3:].any()


@pytest.mark.parametrize("observacion", OBSERVACIONES)
def test_does_not_depend_on_the_opponents_hidden_cards(observacion):
    # Same public history, different cards still in the opponent's hand.
    game = pyspiel.load_game("truco", {"observacion": observacion})
    a = example(game, ("7o", "2", "10")).information_state_tensor(0)
    b = example(game, ("4", "4", "10")).information_state_tensor(0)
    np.testing.assert_array_equal(a, b)


def test_each_player_sees_its_own_side(game):
    state = example(game)
    me, rival = encode(state, 0, "compact"), encode(state, 1, "compact")
    assert me[0] == 1 and rival[0] == 0  # only player 0 is mano
    assert me[1] == 1 and rival[2] == 1  # player 0 won the first trick, player 1 lost it
    assert me[21] == 1 and rival[20] == 1  # truco: by the opponent / by me
    assert me[18] == 1 and rival[19] == 1  # quiero: mine / the opponent's


def test_unknown_encoding_raises(game):
    with pytest.raises(ValueError, match="encoding"):
        encode(example(game), 0, "tiny")
