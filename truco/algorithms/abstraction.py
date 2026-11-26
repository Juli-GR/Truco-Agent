"""Information abstraction of the truco game for tabular MCCFR.

Tabular MCCFR keeps one row per information set, and the exact information sets of the full deck
are about 2e9 (experiments/truco_infosets.py). The abstraction maps every information set to a
smaller key, and MCCFR keeps one row per key, sharing the strategy among the information sets it
merges.

The key holds the same information as the `full` encoding (truco/games/observations.py): what is
lost is the order of the calls within a trick, only the trick in which each call was made is kept.
On the 10-card deck (`niveles=5`) it merges 654,580 information sets into 178,708 keys.

The merged information sets always have the same legal actions, which MCCFR needs: these depend
only on the player's remaining cards, the pending call, the value of the hand and who holds the
quiero, and all of them are in the key.

The abstraction loses perfect recall, so the convergence guarantee of CFR does not apply; the
strategy is evaluated by playing it on the exact game.

This module is the reference implementation, on the OpenSpiel state; the fast simulator must
compute the same keys.
"""

import numpy as np

from truco.games.cards import HAND_SIZE, NUM_TRUCO_LEVELS
from truco.games.truco import _CALL_VALUE, TrucoState, actors

_CALLS = tuple(_CALL_VALUE)  # TRUCO, RETRUCO, VALE_CUATRO
NONE = -1  # empty field (no card, no call, nobody)

# Bits and offset (added to the value so it is >= 0) of each field of the key, in order, to pack it
# into one 64-bit integer (58 bits used).
_CARD = (4, 1)  # NONE or a level 0-13
_FIELDS = (
    (1, 0),  # player
    (1, 0),  # is mano
    *[_CARD] * HAND_SIZE,  # remaining cards
    *[_CARD] * (2 * HAND_SIZE),  # cards of each trick
    (2, -1),  # value of the hand, 1-4
    (2, 1),  # pending call
    (2, 1),  # quiero
    *[(2, 1)] * (2 * len(_CALLS)),  # who made each call, and in which trick
    (2, 0),  # phase
)
FIELD_BITS = np.array([bits for bits, _ in _FIELDS], dtype=np.int64)
FIELD_OFFSET = np.array([offset for _, offset in _FIELDS], dtype=np.int64)
assert FIELD_BITS.sum() <= 63


def pack_key(key: tuple[int, ...]) -> int:
    """The key as a single non-negative integer (fits in an int64)."""
    packed = 0
    for value, bits, offset in zip(key, FIELD_BITS, FIELD_OFFSET, strict=True):
        packed = (packed << int(bits)) | (value + int(offset))
    return packed


def abstract_key(state: TrucoState, player: int) -> tuple[int, ...]:
    """Key of the information set of `player` in `state` (a decision node of `player`).

    Fields, from the point of view of `player`, with NONE where there is nothing:
    - the player and whether it is mano;
    - its remaining cards, sorted (3 slots);
    - each trick's cards, its own and the opponent's (6 slots);
    - the value of the hand, the pending call, who holds the quiero (0 me, 1 the opponent);
    - for each call, who made it (0 me, 1 the opponent) and in which trick;
    - the phase: 0 answer a call, 1 call or pass, 2 play a card.
    """
    tricks = [[NONE, NONE] for _ in range(HAND_SIZE)]  # [my card, opponent's card]
    calls = {call: (NONE, NONE) for call in _CALLS}  # call -> (who, trick)
    cards_played = 0
    for actor, action in zip(actors(state), state.history_actions, strict=True):
        if action < NUM_TRUCO_LEVELS:
            tricks[cards_played // 2][0 if actor == player else 1] = action
            cards_played += 1
        elif action in _CALL_VALUE:
            calls[action] = (int(actor != player), min(cards_played // 2, HAND_SIZE - 1))

    remaining = sorted(state.remaining[player])
    remaining += [NONE] * (HAND_SIZE - len(remaining))
    quiero = NONE if state.quiero is None else int(state.quiero != player)
    pending = NONE if state.pending is None else _CALLS.index(state.pending)
    phase = 0 if state.pending is not None else (1 if state.window else 2)
    return (
        player,
        int(state.mano == player),
        *remaining,
        *(card for trick in tricks for card in trick),
        state.points,
        pending,
        quiero,
        *(field for call in _CALLS for field in calls[call]),
        phase,
    )
