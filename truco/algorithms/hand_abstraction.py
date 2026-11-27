"""Information abstraction of the full hand with envido (truco/games/hand.py) for tabular MCCFR.

The key is the truco key (truco/algorithms/abstraction.py) of the hand's truco part, plus two
envido fields, chosen to add few keys (about x1.3 over the truco alone):

- envido: NOT_CALLED; the pending calls (E, R, E-E, E-R or E-E-R); RESOLVED; or RIVAL_30, resolved
  with the rival declaring 30 or more (it surely holds two good cards of the same suit). Everything
  else about a resolved envido is forgotten: who called, who won, the exact tantos.
- tantos: the player's own tantos in TANTOS_RANGES, while the envido can still be started by the
  player or is pending; NONE otherwise (then they no longer change what it can do).

The key has no suits, so the strategy plays cards by level (`abstract_actions`): merged information
sets may hold different cards of a level (e.g. the 3 of espadas or of oros), but always the same
levels. The card played for a level is the lowest-numbered one held (`card_of_level`); its suit
would only tell something about the tantos, which the key does not track. With that, the actions of
merged information sets are equal: the envido ones depend on the pending calls, on whether the
player has played a card (its remaining cards) and on whether a truco was accepted (the hand's
value), all in the key.

The truco fields keep their meaning, but the key is packed in mixed radix (each field takes exactly
its number of values) instead of bits: the 58 bits of the truco key leave no room for the envido,
while in mixed radix everything takes about 61.3 bits.

This module is the reference implementation, on the OpenSpiel state; the fast simulator
(truco/games/hand_numba.py) must compute the same keys.
"""

import numpy as np

from truco.algorithms.abstraction import NONE, abstract_key
from truco.games import envido
from truco.games.cards import MAX_TANTOS, NUM_CARDS, tantos, truco_level
from truco.games.hand import HandState

E, R = envido.Action.ENVIDO, envido.Action.REAL_ENVIDO
# envido field: NOT_CALLED, then each pending sequence of calls, then RESOLVED and RIVAL_30.
NOT_CALLED = 0
PENDING_CALLS = ((E,), (R,), (E, E), (E, R), (E, E, R))
RESOLVED = 1 + len(PENDING_CALLS)
RIVAL_30 = RESOLVED + 1
NUM_ENVIDO = RIVAL_30 + 1

# Range of each tantos value (0-33): <=7 (no two cards of a suit), 20-22, 23-25, 26-27, 28-29, 30,
# 31, 32-33.
TANTOS_RANGES = np.array(
    [0] * 20 + [1] * 3 + [2] * 3 + [3] * 2 + [4] * 2 + [5, 6, 7, 7], dtype=np.int64
)
assert len(TANTOS_RANGES) == MAX_TANTOS + 1
NUM_RANGES = int(TANTOS_RANGES.max()) + 1

_CARD = (15, 1)  # NONE or a level 0-13
# Number of values and offset (added so it starts at 0) of each field of the key, in order.
_FIELDS = (
    (2, 0),  # player
    (2, 0),  # is mano
    *[_CARD] * 9,  # remaining cards, then the cards of each trick
    (4, -1),  # value of the hand, 1-4
    (4, 1),  # pending truco call
    (3, 1),  # quiero
    *[(3, 1), (4, 1)] * 3,  # who made each truco call, and in which trick
    (3, 0),  # phase
    (NUM_ENVIDO, 0),  # envido
    (NUM_RANGES + 1, 1),  # tantos
)
FIELD_SIZES = np.array([size for size, _ in _FIELDS], dtype=np.int64)
FIELD_OFFSET = np.array([offset for _, offset in _FIELDS], dtype=np.int64)
assert np.prod(FIELD_SIZES.astype(float)) < 2**63


def pack_key(key: tuple[int, ...]) -> int:
    """The key as a single non-negative integer (fits in an int64), in mixed radix."""
    packed = 0
    for value, size, offset in zip(key, FIELD_SIZES, FIELD_OFFSET, strict=True):
        assert 0 <= value + offset < size
        packed = packed * int(size) + value + int(offset)
    return packed


def abstract_actions(state: HandState) -> list[int]:
    """Legal actions with each card replaced by its level, once per level (levels 0-13 do not
    clash with the other actions, which start at NUM_CARDS)."""
    actions = state.legal_actions()
    if actions[0] >= NUM_CARDS:
        return actions
    return sorted({truco_level(card) for card in actions})


def card_of_level(state: HandState, player: int, level: int) -> int:
    """The card that plays `level`: the lowest-numbered one of that level that `player` holds."""
    return min(c for c in state.cards_left[player] if truco_level(c) == level)


def envido_field(state: HandState, player: int) -> int:
    if not state.envido_calls:
        return NOT_CALLED
    if state.envido_pending:
        return 1 + PENDING_CALLS.index(tuple(state.envido_calls))
    rival = state.declared[1 - player]
    return RIVAL_30 if rival is not None and rival >= 30 else RESOLVED


def abstract_hand_key(state: HandState, player: int) -> tuple[int, ...]:
    """Key of the information set of `player` in `state` (a decision node of `player`): the truco
    key, the envido field and the tantos field (see the module docstring)."""
    open_ = state.envido_pending or (not state.envido_calls and state.can_call_envido(player))
    own = int(TANTOS_RANGES[tantos(state.cards[player])]) if open_ else NONE
    return (*abstract_key(state.truco, player), envido_field(state, player), own)
