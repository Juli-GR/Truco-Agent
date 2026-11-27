"""Fast simulator of the full hand with envido (truco/games/hand.py), compiled with numba.

As hand.py builds on truco.py, this module builds on truco_numba.py: a state is the truco
simulator's array (indices 0 to tn.STATE_SIZE - 1, on levels, updated by tn's functions) followed
by the envido fields and the real cards. hand.py stays the reference: tests/test_hand_numba.py
checks that both give the same players, legal actions, keys and returns along random games.

Actions are the same integers as in hand.py: 0-39 play that card, then `Action`.
"""

import numpy as np
from numba import njit

from truco.games import truco_numba as tn
from truco.games.cards import HAND_SIZE, NUM_CARDS, RANKS, truco_level
from truco.games.hand import Action

PASO = int(Action.PASO)
TRUCO = int(Action.TRUCO)
QUIERO = int(Action.QUIERO)
NO_QUIERO = int(Action.NO_QUIERO)
MAZO = int(Action.MAZO)
ENVIDO = int(Action.ENVIDO)
REAL_ENVIDO = int(Action.REAL_ENVIDO)
TRUCO_OFFSET = tn.PASO - PASO  # truco_numba action = hand action + TRUCO_OFFSET (non-card)
MAX_ACTIONS = 5  # e.g. paso, truco, mazo, envido, real envido
MAX_ENVIDO_CALLS = 3  # E, E, R
TERMINAL = tn.TERMINAL

# Layout after the truco part (indices into the array). -1 means nothing / nobody.
CARDS = tn.STATE_SIZE  # 3 slots per player: the dealt cards, sorted
CARDS_LEFT = CARDS + 6  # 3 slots per player: cards still held, sorted, -1 at the end
TANTOS = CARDS_LEFT + 6  # 2 slots: tantos of each player
ENVIDO_N = TANTOS + 2  # envido calls made
ENVIDO_CALLS = ENVIDO_N + 1  # 3 slots: ENVIDO or REAL_ENVIDO
ENVIDO_CALLER = ENVIDO_CALLS + MAX_ENVIDO_CALLS  # who made the last envido call
ENVIDO_WINNER = ENVIDO_CALLER + 1  # set when the envido is resolved
ENVIDO_POINTS = ENVIDO_WINNER + 1
DECLARED = ENVIDO_POINTS + 1  # 2 slots: tantos declared by each player
HAND_HASH = DECLARED + 2  # hash of the hand's actions, for exact_key
STATE_SIZE = HAND_HASH + 1

CARD_LEVEL = np.array([truco_level(c) for c in range(NUM_CARDS)], dtype=np.int64)
CARD_SUIT = np.array([c // len(RANKS) for c in range(NUM_CARDS)], dtype=np.int64)
CARD_ENVIDO = np.array(
    [0 if RANKS[c % len(RANKS)] >= 10 else RANKS[c % len(RANKS)] for c in range(NUM_CARDS)],
    dtype=np.int64,
)
DECK = np.arange(NUM_CARDS, dtype=np.int64)


@njit(cache=True)
def tantos(cards):
    """Tantos of 3 cards (as cards.tantos)."""
    best = 0
    for i in range(HAND_SIZE):
        best = max(best, CARD_ENVIDO[cards[i]])
        for j in range(i + 1, HAND_SIZE):
            if CARD_SUIT[cards[i]] == CARD_SUIT[cards[j]]:
                best = max(best, 20 + CARD_ENVIDO[cards[i]] + CARD_ENVIDO[cards[j]])
    return best


@njit(cache=True)
def new_state(mano, cards0, cards1):
    """State right after the deal. Hands are arrays of 3 cards."""
    s = np.full(STATE_SIZE, -1, dtype=np.int64)
    s[: tn.STATE_SIZE] = tn.new_state(mano, CARD_LEVEL[cards0], CARD_LEVEL[cards1])
    s[CARDS : CARDS + HAND_SIZE] = np.sort(cards0)
    s[CARDS + HAND_SIZE : CARDS + 2 * HAND_SIZE] = np.sort(cards1)
    s[CARDS_LEFT : CARDS_LEFT + 2 * HAND_SIZE] = s[CARDS : CARDS + 2 * HAND_SIZE]
    s[TANTOS] = tantos(cards0)
    s[TANTOS + 1] = tantos(cards1)
    s[ENVIDO_N] = 0
    s[ENVIDO_POINTS] = 0
    s[HAND_HASH] = tn._HASH_START
    return s


def from_openspiel(state) -> np.ndarray:
    """Simulator state equal to `state`, a HandState after the deal (replays its history)."""
    s = new_state(state.mano, np.array(state.cards[0]), np.array(state.cards[1]))
    for action in state.history_actions:
        apply_action(s, action)
    return s


@njit(cache=True)
def deal():
    """Random state after the deal: who is mano, and 3 cards for each player."""
    cards = DECK.copy()
    for i in range(2 * HAND_SIZE):  # partial Fisher-Yates shuffle
        j = np.random.randint(i, NUM_CARDS)
        cards[i], cards[j] = cards[j], cards[i]
    return new_state(np.random.randint(0, 2), cards[:HAND_SIZE], cards[HAND_SIZE : 2 * HAND_SIZE])


@njit(cache=True)
def envido_pending(s):
    return s[ENVIDO_N] > 0 and s[ENVIDO_WINNER] < 0


@njit(cache=True)
def current_player(s):
    if s[tn.GAME_OVER]:
        return TERMINAL
    if envido_pending(s):
        return 1 - s[ENVIDO_CALLER]
    return tn.current_player(s)


@njit(cache=True)
def returns(s, player):
    """Points won by `player` in a finished hand: truco plus envido (negative if lost)."""
    points = tn.returns(s, player)
    if s[ENVIDO_WINNER] >= 0:
        points += s[ENVIDO_POINTS] if s[ENVIDO_WINNER] == player else -s[ENVIDO_POINTS]
    return points


@njit(cache=True)
def can_call_envido(s, player):
    """Whether `player` may start the envido (in a window or answering a truco)."""
    return (
        s[ENVIDO_N] == 0
        and s[CARDS_LEFT + HAND_SIZE * player + HAND_SIZE - 1] >= 0  # has not played a card
        and s[tn.POINTS] == 1  # no truco accepted
    )


@njit(cache=True)
def legal_actions(s, out):
    """Writes the legal actions, sorted, into `out` (length MAX_ACTIONS); returns how many."""
    player = current_player(s)
    if envido_pending(s):
        out[0] = QUIERO
        out[1] = NO_QUIERO
        n = 2
        calls = s[ENVIDO_N]
        if calls == 1 and s[ENVIDO_CALLS] == ENVIDO:  # E: envido or real envido
            out[2] = ENVIDO
            out[3] = REAL_ENVIDO
            n = 4
        elif calls == 2 and s[ENVIDO_CALLS + 1] == ENVIDO:  # E-E: real envido
            out[2] = REAL_ENVIDO
            n = 3
        return n

    truco_actions = np.empty(tn.MAX_ACTIONS, dtype=np.int64)
    m = tn.legal_actions(s, truco_actions)
    n = 0
    if truco_actions[0] < tn.PASO:  # playing a card: any card held (already sorted)
        base = CARDS_LEFT + HAND_SIZE * player
        for i in range(HAND_SIZE):
            if s[base + i] >= 0:
                out[n] = s[base + i]
                n += 1
        return n
    for k in range(m):
        out[n] = truco_actions[k] - TRUCO_OFFSET
        n += 1
    if can_call_envido(s, player):
        out[n] = ENVIDO
        out[n + 1] = REAL_ENVIDO
        n += 2
    return n


@njit(cache=True)
def _call_points(call):
    return 2 if call == ENVIDO else 3


@njit(cache=True)
def _resolve_envido(s, accepted):
    calls = s[ENVIDO_N]
    if not accepted:  # the last caller scores what was accumulated before, at least 1
        before = 0
        for k in range(calls - 1):
            before += _call_points(s[ENVIDO_CALLS + k])
        s[ENVIDO_WINNER] = s[ENVIDO_CALLER]
        s[ENVIDO_POINTS] = max(1, before)
        return
    mano = s[tn.MANO]
    pie = 1 - mano
    s[DECLARED + mano] = s[TANTOS + mano]
    if s[TANTOS + pie] > s[TANTOS + mano]:  # the pie declares only if it wins
        s[DECLARED + pie] = s[TANTOS + pie]
        s[ENVIDO_WINNER] = pie
    else:
        s[ENVIDO_WINNER] = mano
    total = 0
    for k in range(calls):
        total += _call_points(s[ENVIDO_CALLS + k])
    s[ENVIDO_POINTS] = total


@njit(cache=True)
def _remove_card(s, player, card):
    base = CARDS_LEFT + HAND_SIZE * player
    for i in range(HAND_SIZE):  # remove the card, keeping the rest sorted
        if s[base + i] == card:
            for j in range(i, HAND_SIZE - 1):
                s[base + j] = s[base + j + 1]
            s[base + HAND_SIZE - 1] = -1
            return


@njit(cache=True)
def apply_action(s, action):
    """Applies `action` (legal) of the current player, in place."""
    player = current_player(s)
    s[HAND_HASH] = tn._mix(s[HAND_HASH], action)
    if action == ENVIDO or action == REAL_ENVIDO:  # does not use the chance in the window
        s[ENVIDO_CALLS + s[ENVIDO_N]] = action
        s[ENVIDO_N] += 1
        s[ENVIDO_CALLER] = player
    elif envido_pending(s):
        _resolve_envido(s, action == QUIERO)
    elif action < NUM_CARDS:
        _remove_card(s, player, action)
        tn.apply_action(s, CARD_LEVEL[action])
    else:
        tn.apply_action(s, action + TRUCO_OFFSET)


@njit(cache=True)
def exact_key(s, player):
    """Key of `player`'s exact information set: a 62-bit hash of the player, who is mano, the
    player's cards, the history and the declared tantos. Used to validate on small samples."""
    h = tn._mix(tn._mix(s[HAND_HASH], player), s[tn.MANO])
    for k in range(HAND_SIZE):
        h = tn._mix(h, s[CARDS + HAND_SIZE * player + k])
    h = tn._mix(tn._mix(h, s[DECLARED]), s[DECLARED + 1])
    return h & tn._KEY_MASK
