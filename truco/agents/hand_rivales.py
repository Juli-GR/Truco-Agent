"""Fixed opponents for the full hand with envido (truco/games/hand.py).

The heuristics play the truco exactly as in truco_rivales, on the hand's inner truco state, and
the envido with simple rules by their tantos: the more tantos, the more likely they call, and they
accept or raise above a threshold. How they play is explained in docs/rivales.md.
"""

import math
from dataclasses import dataclass

import numpy as np

from truco.agents.truco_rivales import AGRESIVO, CONSERVADOR, EQUILIBRADO, HeuristicPolicy
from truco.games import envido
from truco.games.cards import NUM_TRUCO_LEVELS, tantos, truco_level
from truco.games.hand import Action, HandState
from truco.games.truco import Action as TrucoAction

_CALLS = (Action.ENVIDO, Action.REAL_ENVIDO)
_SCALE = 1.8  # tantos: the call probability goes from 10 % to 90 % in about 4 tantos each side


class HandRandomPolicy:
    """Chooses uniformly among the legal actions."""

    def action_probabilities(self, states: list[HandState]) -> np.ndarray:
        probs = np.zeros((len(states), states[0].num_distinct_actions()))
        for i, state in enumerate(states):
            legal = state.legal_actions()
            probs[i, legal] = 1 / len(legal)
        return probs


@dataclass(frozen=True)
class EnvidoStyle:
    """How a heuristic plays the envido. Thresholds are in tantos.

    - cantar: tantos with which it starts the envido half of the time (more tantos, more often).
    - real: from these tantos it calls real envido instead of envido.
    - querer: tantos needed to accept an envido worth 2; +1 per extra point at stake, -1 as mano
      (the mano wins ties).
    - subir: from these tantos it raises instead of accepting, if it can.
    - farol: minimum probability of starting the envido, whatever its tantos.
    """

    cantar: int
    real: int
    querer: int
    subir: int
    farol: float

    def call_probability(self, t: int) -> float:
        return max(self.farol, 1 / (1 + math.exp(-(t - self.cantar) / _SCALE)))

    def start(self, t: int) -> dict[Action, float]:
        """Probability of starting the envido with each call."""
        call = Action.REAL_ENVIDO if t >= self.real else Action.ENVIDO
        return {call: self.call_probability(t)}

    def answer(self, t: int, is_mano: bool, calls: list[envido.Action], legal) -> Action:
        """Answer to a pending envido: raise, accept or reject (deterministic)."""
        raises = [a for a in _CALLS if a in legal]
        if raises and t >= self.subir:
            if Action.REAL_ENVIDO in raises and t >= self.real:
                return Action.REAL_ENVIDO
            return raises[0]
        at_stake = envido.accepted_points(calls)
        threshold = self.querer + (at_stake - 2) - int(is_mano)
        return Action.QUIERO if t >= threshold else Action.NO_QUIERO


@dataclass(frozen=True)
class HandHeuristicPolicy:
    """Truco by a truco_rivales heuristic, envido by an EnvidoStyle.

    It considers starting the envido only in its first decision of the hand (as mano, before
    anything; as pie, after the mano's first action), like the isolated envido.
    """

    truco: HeuristicPolicy
    envido: EnvidoStyle

    @property
    def nombre(self) -> str:
        return self.truco.nombre

    def probabilities(self, state: HandState) -> dict[int, float]:
        player = state.current_player()
        legal = state.legal_actions()
        t = tantos(state.cards[player])
        is_mano = state.mano == player
        if state.envido_pending:
            return {self.envido.answer(t, is_mano, state.envido_calls, legal): 1.0}

        probs = self._truco_probabilities(state, player)
        first_decision = len(state.history_actions) == (0 if is_mano else 1)
        if not (first_decision and Action.ENVIDO in legal):
            return probs
        start = self.envido.start(t)
        keep = 1 - sum(start.values())
        return {**{a: keep * p for a, p in probs.items()}, **start}

    def _truco_probabilities(self, state: HandState, player: int) -> dict[int, float]:
        """The truco heuristic's decision on the inner truco state, in the hand's actions (a
        level becomes the lowest-numbered card of that level the player holds)."""
        probs = {}
        for a, p in self.truco.probabilities(state.truco).items():
            if a < NUM_TRUCO_LEVELS:
                probs[min(c for c in state.cards_left[player] if truco_level(c) == a)] = p
            else:
                probs[Action[TrucoAction(a).name]] = p
        return probs

    def action_probabilities(self, states: list[HandState]) -> np.ndarray:
        probs = np.zeros((len(states), states[0].num_distinct_actions()))
        for i, state in enumerate(states):
            for action, p in self.probabilities(state).items():
                probs[i, action] = p
        return probs


ENVIDO_CONSERVADOR = EnvidoStyle(cantar=28, real=31, querer=27, subir=31, farol=0.0)
ENVIDO_EQUILIBRADO = EnvidoStyle(cantar=26, real=30, querer=26, subir=30, farol=0.05)
ENVIDO_AGRESIVO = EnvidoStyle(cantar=24, real=28, querer=24, subir=29, farol=0.15)
HAND_RIVALES = {
    "aleatorio": HandRandomPolicy(),
    **{
        p.nombre: p
        for p in (
            HandHeuristicPolicy(CONSERVADOR, ENVIDO_CONSERVADOR),
            HandHeuristicPolicy(EQUILIBRADO, ENVIDO_EQUILIBRADO),
            HandHeuristicPolicy(AGRESIVO, ENVIDO_AGRESIVO),
        )
    },
}
