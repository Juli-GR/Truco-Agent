"""Observation tensors of the truco game, the input of the networks that play it.

The game parameter `observacion` picks the encoding that information_state_tensor and
observation_tensor return. Every encoding is computed from the player's information set only: its
own cards and the public history. They differ in how much of the game's structure is spelled out
for the network:

- `raw`: `TrucoObserver` in truco.py. Dealt cards and the list of actions, without saying who
  made each one; the network has to rebuild tricks, calls and turns from the rules. It is the
  only exact encoding of the information set.
- `compact`: the same information, interpreted. Each card is a single number (its strength:
  the fraction of the deck it beats), plus the state of the tricks and the calls.
- `full`: `compact` plus a one-hot vector per card, so the network can also learn effects of a
  specific card that do not follow from its strength.

What `compact` and `full` describe, from the point of view of `player`:

- Whether the player is mano.
- Its remaining cards, best first.
- In each trick, its card and the opponent's (the opponent's card in the current trick is the one
  on the table), and who won the trick.
- The value of the hand, the pending call, who holds the quiero (may raise next), who made each
  call and in which trick.
- What the player has to do now: answer a call, call or pass, or play a card.

The order of calls within a trick (e.g. truco before or after the first card) is reduced to the
trick in which they were made, so `compact` and `full` merge a few information sets that `raw`
tells apart.

Information state strings are always those of `TrucoObserver` (the exact information set), whatever
the encoding: CFR and the EFG export rely on them.
"""

import numpy as np

from truco.games.cards import HAND_SIZE, NUM_TRUCO_LEVELS, card_score
from truco.games.truco import _CALL_VALUE, TrucoObserver, TrucoState, actors

_CALLS = tuple(_CALL_VALUE)  # TRUCO, RETRUCO, VALE_CUATRO
_MAX_POINTS = 4
_SUMMARY_SIZE = (
    1  # mano
    + HAND_SIZE * 3  # result of each trick: won, lost, parda
    + _MAX_POINTS  # value of the hand
    + len(_CALLS)  # pending call
    + 3  # quiero: nobody, me, opponent
    + len(_CALLS) * (2 + HAND_SIZE)  # each call: by me / by the opponent, and in which trick
    + 3  # phase: answer a call, call or pass, play a card
)
_CARD_SLOTS = HAND_SIZE + HAND_SIZE * 2  # remaining cards, and both cards of every trick
_SIZES = {
    "compact": _SUMMARY_SIZE + _CARD_SLOTS * 2,
    "full": _SUMMARY_SIZE + _CARD_SLOTS * 2 + NUM_TRUCO_LEVELS * (1 + HAND_SIZE * 2),
}


class EncodedObserver:
    """Observer (OpenSpiel's PyObserver interface) for the `compact` and `full` encodings.

    Tensors come from `encode`; strings from the `raw` observer, so they stay exact.
    """

    def __init__(self, encoding: str, raw: TrucoObserver):
        if encoding not in _SIZES:
            raise ValueError(f"Unknown encoding: {encoding}")
        self.encoding = encoding
        self.raw = raw
        self.tensor = np.zeros(_SIZES[encoding], np.float32)
        self.dict = {"observacion": self.tensor}

    def set_from(self, state: TrucoState, player: int):
        self.tensor[:] = encode(state, player, self.encoding)

    def string_from(self, state: TrucoState, player: int) -> str:
        return self.raw.string_from(state, player)


def encode(state: TrucoState, player: int, encoding: str) -> np.ndarray:
    """`compact` or `full` observation of `player` in `state` (float32 vector)."""
    if encoding not in _SIZES:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Rebuild who played each card and who made each call, from public information.
    tricks = [[None, None] for _ in range(HAND_SIZE)]  # [my card, opponent's card]
    calls: dict[int, tuple[bool, int]] = {}  # call -> (made by me, trick)
    cards_played = 0
    for actor, action in zip(actors(state), state.history_actions, strict=True):
        if action < NUM_TRUCO_LEVELS:
            tricks[cards_played // 2][0 if actor == player else 1] = action
            cards_played += 1
        elif action in _CALL_VALUE:
            calls[action] = (actor == player, min(cards_played // 2, HAND_SIZE - 1))
    # Before the deal (OpenSpiel asks for tensors at the initial state too) there are no cards.
    remaining = sorted(state.remaining[player]) if len(state.remaining) > player else []
    slots = [*remaining, *[None] * (HAND_SIZE - len(remaining))]
    slots += [card for trick in tricks for card in trick]

    x = np.zeros(_SIZES[encoding], dtype=np.float32)
    i = 0

    def put(index: int | None, size: int):
        """One-hot of `index` (all zeros if None) in the next `size` positions."""
        nonlocal i
        if index is not None:
            x[i + index] = 1
        i += size

    put(0 if state.mano == player else None, 1)
    for t in range(HAND_SIZE):
        if t < len(state.results):
            winner = state.results[t]
            put(2 if winner is None else (0 if winner == player else 1), 3)
        else:
            put(None, 3)
    put(state.points - 1, _MAX_POINTS)
    put(_CALLS.index(state.pending) if state.pending is not None else None, len(_CALLS))
    put(0 if state.quiero is None else (1 if state.quiero == player else 2), 3)
    for call in _CALLS:
        by_me, trick = calls.get(call, (None, None))
        put(None if by_me is None else (0 if by_me else 1), 2)
        put(trick, HAND_SIZE)
    if state.is_chance_node():
        phase = None
    else:
        phase = 0 if state.pending is not None else (1 if state.window else 2)
    put(phase, 3)

    for card in slots:  # each card as (present, strength)
        if card is not None:
            x[i], x[i + 1] = 1, card_score(card)
        i += 2

    if encoding == "full":
        for card in remaining:
            x[i + card] += 1
        i += NUM_TRUCO_LEVELS
        for card in slots[HAND_SIZE:]:
            put(card, NUM_TRUCO_LEVELS)
    assert i == len(x)
    return x
