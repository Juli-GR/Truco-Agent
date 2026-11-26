"""Fast simulator of one hand of truco (truco/games/truco.py), compiled with numba.

The OpenSpiel game in Python takes tens of microseconds per node, too slow for tabular MCCFR on
the full deck. This module implements the same rules on a small integer array, with functions that
numba compiles to machine code. truco.py stays the reference: tests/test_truco_numba.py checks that
both give the same legal actions, players, keys and returns along random games.

A state is a 1-D int64 array (`new_state`, layout below). `apply_action` modifies it in place, so
to explore several actions from one state, copy it first (`state.copy()`).

Actions are the same integers as in truco.py: 0-13 play a card of that level, then `Action`.
"""

import numpy as np
from numba import njit

from truco.algorithms.abstraction import FIELD_BITS, FIELD_OFFSET
from truco.games.cards import HAND_SIZE, NUM_TRUCO_LEVELS, TRUCO_LEVEL_CARDS
from truco.games.truco import Action

PASO = int(Action.PASO)
TRUCO = int(Action.TRUCO)
QUIERO = int(Action.QUIERO)
NO_QUIERO = int(Action.NO_QUIERO)
MAZO = int(Action.MAZO)
NUM_CALLS = 3  # TRUCO, RETRUCO, VALE_CUATRO: action TRUCO + call index
MAX_ACTIONS = 3  # at most 3 legal actions in any state
TERMINAL = -4  # pyspiel.PlayerId.TERMINAL
PARDA = 2  # trick result when it is a tie

# State layout (indices into the array). -1 means nothing / nobody.
MANO = 0
POINTS = 1  # value of the hand, 1-4
QUIERO_HOLDER = 2  # who may raise the last accepted call
PENDING = 3  # call index waiting for an answer
CALLER = 4  # who made the pending call
WINDOW_LEN = 5  # players that still have their chance in the call window (0-2)
WINDOW = 6  # 2 slots, in order
CARD_PLAYER = 8  # who plays when the window closes
NUM_RESULTS = 9
RESULTS = 10  # 3 slots: winner of each finished trick, or PARDA
TABLE_LEN = 13  # cards on the table in the current trick (0-1)
TABLE_PLAYER = 14
TABLE_LEVEL = 15
GAME_OVER = 16
WINNER = 17
REMAINING = 18  # 3 slots per player, sorted, played cards removed and -1 at the end
TRICK_CARDS = 24  # [trick][player] = level played
CARDS_PLAYED = 30
CALLS = 31  # [call][who, trick]
HANDS = 37  # 3 slots per player: the dealt hand, sorted
HISTORY_HASH = 43  # hash of the actions played, for exact_key
STATE_SIZE = 44

_HASH_START = 1469598103934665603
_HASH_PRIME = 1099511628211  # FNV-1a prime; int64 multiplication wraps around
_KEY_MASK = (1 << 62) - 1  # keys are non-negative


@njit(cache=True)
def new_state(mano, hand0, hand1):
    """State right after the deal. Hands are arrays of 3 levels."""
    s = np.full(STATE_SIZE, -1, dtype=np.int64)
    s[MANO] = mano
    s[POINTS] = 1
    s[WINDOW_LEN] = 2
    s[WINDOW] = mano
    s[WINDOW + 1] = 1 - mano
    s[CARD_PLAYER] = mano
    s[NUM_RESULTS] = 0
    s[TABLE_LEN] = 0
    s[GAME_OVER] = 0
    s[CARDS_PLAYED] = 0
    s[REMAINING : REMAINING + HAND_SIZE] = np.sort(hand0)
    s[REMAINING + HAND_SIZE : REMAINING + 2 * HAND_SIZE] = np.sort(hand1)
    s[HANDS : HANDS + 2 * HAND_SIZE] = s[REMAINING : REMAINING + 2 * HAND_SIZE]
    s[HISTORY_HASH] = _HASH_START
    return s


@njit(cache=True)
def _mix(h, value):
    return (h ^ (value + 2)) * _HASH_PRIME


def make_deck(niveles: int) -> np.ndarray:
    """One entry per card of the deck with the `niveles` highest levels: the card's level."""
    return np.array(
        [level for level in range(niveles) for _ in TRUCO_LEVEL_CARDS[level]], dtype=np.int64
    )


@njit(cache=True)
def seed(value):
    """Seeds numba's random generator (separate from numpy's)."""
    np.random.seed(value)


@njit(cache=True)
def deal(deck):
    """Random state after the deal: who is mano, and 3 cards for each player from `deck`."""
    cards = deck.copy()
    n = len(cards)
    for i in range(2 * HAND_SIZE):  # partial Fisher-Yates shuffle
        j = np.random.randint(i, n)
        cards[i], cards[j] = cards[j], cards[i]
    return new_state(np.random.randint(0, 2), cards[:HAND_SIZE], cards[HAND_SIZE : 2 * HAND_SIZE])


@njit(cache=True)
def current_player(s):
    if s[GAME_OVER]:
        return TERMINAL
    if s[PENDING] >= 0:
        return 1 - s[CALLER]
    if s[WINDOW_LEN] > 0:
        return s[WINDOW]
    return s[CARD_PLAYER]


@njit(cache=True)
def returns(s, player):
    """Points won by `player` in a finished hand (negative if lost)."""
    points = s[POINTS]
    return points if s[WINNER] == player else -points


@njit(cache=True)
def legal_actions(s, out):
    """Writes the legal actions, sorted, into `out` (length MAX_ACTIONS); returns how many."""
    player = current_player(s)
    if s[PENDING] >= 0:
        n = 0
        if s[PENDING] + 1 < NUM_CALLS:
            out[n] = TRUCO + s[PENDING] + 1
            n += 1
        out[n] = QUIERO
        out[n + 1] = NO_QUIERO
        return n + 2
    if s[WINDOW_LEN] > 0:
        out[0] = PASO
        n = 1
        if s[POINTS] <= NUM_CALLS and (s[QUIERO_HOLDER] < 0 or s[QUIERO_HOLDER] == player):
            out[n] = TRUCO + s[POINTS] - 1
            n += 1
        out[n] = MAZO
        return n + 1
    n = 0
    base = REMAINING + HAND_SIZE * player
    for i in range(HAND_SIZE):
        level = s[base + i]
        if level >= 0 and (n == 0 or out[n - 1] != level):
            out[n] = level
            n += 1
    return n


@njit(cache=True)
def _finish(s, winner, points):
    s[GAME_OVER] = 1
    s[WINNER] = winner
    s[POINTS] = points


@njit(cache=True)
def _hand_winner(s):
    """Winner of the hand from the finished tricks, or -1 if not decided yet."""
    n = s[NUM_RESULTS]
    wins0 = 0
    wins1 = 0
    first_decided = -1
    pardas = 0
    for t in range(n):
        r = s[RESULTS + t]
        if r == PARDA:
            pardas += 1
        else:
            if first_decided < 0:
                first_decided = r
            if r == 0:
                wins0 += 1
            else:
                wins1 += 1
    if wins0 == 2:
        return 0
    if wins1 == 2:
        return 1
    if pardas > 0 and first_decided >= 0:
        return first_decided
    if n == HAND_SIZE:
        return s[MANO]
    return -1


@njit(cache=True)
def _play_card(s, player, level):
    base = REMAINING + HAND_SIZE * player
    for i in range(HAND_SIZE):  # remove the card, keeping the rest sorted
        if s[base + i] == level:
            for j in range(i, HAND_SIZE - 1):
                s[base + j] = s[base + j + 1]
            s[base + HAND_SIZE - 1] = -1
            break
    trick = s[CARDS_PLAYED] // 2
    s[TRICK_CARDS + 2 * trick + player] = level
    s[CARDS_PLAYED] += 1

    if s[TABLE_LEN] == 0:
        s[TABLE_LEN] = 1
        s[TABLE_PLAYER] = player
        s[TABLE_LEVEL] = level
        next_player = 1 - player
    else:
        first = s[TABLE_PLAYER]
        first_level = s[TABLE_LEVEL]
        if first_level == level:
            winner = PARDA
        elif first_level < level:
            winner = first
        else:
            winner = 1 - first
        s[RESULTS + s[NUM_RESULTS]] = winner
        s[NUM_RESULTS] += 1
        s[TABLE_LEN] = 0
        hand = _hand_winner(s)
        if hand >= 0:
            _finish(s, hand, s[POINTS])
            return
        next_player = s[MANO] if winner == PARDA else winner
    s[CARD_PLAYER] = next_player
    s[WINDOW_LEN] = 2  # the player who played the last card acts first
    s[WINDOW] = player
    s[WINDOW + 1] = 1 - player


@njit(cache=True)
def apply_action(s, action):
    """Applies `action` (legal) of the current player, in place."""
    player = current_player(s)
    s[HISTORY_HASH] = _mix(s[HISTORY_HASH], action)
    if action < NUM_TRUCO_LEVELS:
        _play_card(s, player, action)
        return
    if s[PENDING] < 0:  # the player uses its chance in the window
        s[WINDOW] = s[WINDOW + 1]
        s[WINDOW_LEN] -= 1
    if action == PASO:
        return
    if action == MAZO:
        _finish(s, 1 - player, s[POINTS])
    elif action == QUIERO:
        s[POINTS] = s[PENDING] + 2
        s[QUIERO_HOLDER] = player
        s[PENDING] = -1
    elif action == NO_QUIERO:
        _finish(s, s[CALLER], s[PENDING] + 1)
    else:  # a call
        call = action - TRUCO
        s[PENDING] = call
        s[CALLER] = player
        s[CALLS + 2 * call] = player
        s[CALLS + 2 * call + 1] = min(s[CARDS_PLAYED] // 2, HAND_SIZE - 1)


@njit(cache=True)
def _relative(who, player):
    """0 if `who` is `player`, 1 if it is the opponent, -1 if nobody."""
    if who < 0:
        return -1
    return 0 if who == player else 1


@njit(cache=True)
def info_key(s, player):
    """Abstract key of `player`'s information set, packed: equals
    pack_key(abstract_key(state, player)) of truco.algorithms.abstraction."""
    fields = np.empty(len(FIELD_BITS), dtype=np.int64)
    fields[0] = player
    fields[1] = 1 if s[MANO] == player else 0
    i = 2
    for k in range(HAND_SIZE):
        fields[i] = s[REMAINING + HAND_SIZE * player + k]
        i += 1
    for t in range(HAND_SIZE):
        fields[i] = s[TRICK_CARDS + 2 * t + player]
        fields[i + 1] = s[TRICK_CARDS + 2 * t + 1 - player]
        i += 2
    fields[i] = s[POINTS]
    fields[i + 1] = s[PENDING]
    fields[i + 2] = _relative(s[QUIERO_HOLDER], player)
    i += 3
    for call in range(NUM_CALLS):
        fields[i] = _relative(s[CALLS + 2 * call], player)
        fields[i + 1] = s[CALLS + 2 * call + 1]
        i += 2
    if s[PENDING] >= 0:
        fields[i] = 0
    elif s[WINDOW_LEN] > 0:
        fields[i] = 1
    else:
        fields[i] = 2

    packed = 0
    for k in range(len(FIELD_BITS)):
        packed = (packed << FIELD_BITS[k]) | (fields[k] + FIELD_OFFSET[k])
    return packed


@njit(cache=True)
def exact_key(s, player):
    """Key of `player`'s exact information set (no abstraction): a 62-bit hash of the player,
    who is mano, the player's hand and the history. Used to validate MCCFR on small decks, where
    a hash collision is practically impossible."""
    h = _mix(_mix(s[HISTORY_HASH], player), s[MANO])
    for k in range(HAND_SIZE):
        h = _mix(h, s[HANDS + HAND_SIZE * player + k])
    return h & _KEY_MASK
