"""Rivales fijos para el truco: un piso aleatorio y heurísticas que juegan como una persona.

Con el mazo completo no se puede calcular la exploitabilidad, así que estos rivales son la vara fija
contra la que se miden los agentes que aprenden. Cómo juegan está explicado en docs/rivales.md.

Todas las políticas reciben una lista de estados y devuelven una matriz (estados x acciones) con la
probabilidad de cada acción, para que una red pueda evaluar muchas partidas en un solo paso.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cache, lru_cache

import numpy as np

from truco.games.cards import NUM_TRUCO_LEVELS, TRUCO_LEVEL_CARDS
from truco.games.truco import (
    _CALL_VALUE,
    _NEXT_CALL,
    Action,
    TrucoState,
    deal_outcomes,
    hand_winner,
    hands,
)

NUM_ACTIONS = NUM_TRUCO_LEVELS + len(Action)
_AGGRESSIVE = {Action.TRUCO, Action.RETRUCO, Action.VALE_CUATRO, Action.QUIERO}
_CACHE_SIZE = 1 << 20  # situaciones recordadas; las de una mano se repiten mucho entre partidas


class RandomPolicy:
    """Elige uniformemente entre las acciones legales."""

    def action_probabilities(self, states: list[TrucoState]) -> np.ndarray:
        probs = np.zeros((len(states), NUM_ACTIONS))
        for i, state in enumerate(states):
            legal = state.legal_actions()
            probs[i, legal] = 1 / len(legal)
        return probs


# --- Fuerza de una mano ---------------------------------------------------------------------------


@cache
def card_score(level: int) -> float:
    """Fracción de las otras 39 cartas del mazo completo a las que le gana una carta del nivel."""
    lower = sum(len(TRUCO_LEVEL_CARDS[lv]) for lv in range(level + 1, NUM_TRUCO_LEVELS))
    return lower / 39


def hand_strength(hand: tuple[int, ...]) -> float:
    """Fuerza de una mano en [0, 1]: promedio de card_score de sus cartas."""
    return sum(card_score(level) for level in hand) / len(hand)


@cache
def hand_percentile(niveles: int, hand: tuple[int, ...]) -> float:
    """Probabilidad de que una mano repartida al azar tenga fuerza menor o igual que `hand`."""
    strength = hand_strength(hand)
    return sum(
        p for i, p in deal_outcomes(niveles, ()) if hand_strength(hands(niveles)[i]) <= strength
    )


# --- Juego de cartas ------------------------------------------------------------------------------


def choose_card(
    remaining: tuple[int, ...],
    rival_card: int | None,
    results: tuple[int | None, ...],
    me: int,
    mano: int,
) -> int:
    """Carta a tirar según reglas "de manual". Niveles: menor nivel = carta más alta.

    `rival_card` es la carta del rival en la baza en curso (None si abro yo) y `results` el ganador
    de cada baza terminada (None = parda).
    """
    cards = sorted(remaining)
    if len(cards) == 1:
        return cards[0]
    if rival_card is None:  # abro la baza
        if not results:
            return cards[len(cards) // 2]  # primera: una carta media
        if me in results:
            return cards[-1]  # ya gané una baza: tiro la más baja y guardo la buena
        return cards[0]  # después de una parda la baza define: tiro la más alta

    def baza(card: int) -> int | None:
        if card == rival_card:
            return None
        return me if card < rival_card else 1 - me

    # 1) La carta más baja que gana la mano ya mismo (incluye pardar después de ganar la primera).
    winning_hand = [c for c in cards if hand_winner([*results, baza(c)], mano) == me]
    if winning_hand:
        return winning_hand[-1]
    # 2) Pardar la primera si me queda una carta buena: la segunda define y la juego yo.
    if not results and rival_card in cards:
        others = list(cards)
        others.remove(rival_card)
        if card_score(others[0]) >= 0.6:
            return rival_card
    # 3) La carta más baja que gana la baza.
    winning_baza = [c for c in cards if baza(c) == me]
    if winning_baza:
        return winning_baza[-1]
    # 4) Si no puedo ganar, tiro la más baja.
    return cards[-1]


@lru_cache(maxsize=_CACHE_SIZE)
def playout(
    my_cards: tuple[int, ...],
    rival_cards: tuple[int, ...],
    table: int | None,
    results: tuple[int | None, ...],
    next_player: int,
    me: int,
    mano: int,
) -> int:
    """Ganador de la mano si ambos jugadores tiran según choose_card y nadie canta más.

    `table` es la carta ya tirada en la baza en curso (por el jugador que no es `next_player`).
    """
    remaining = {me: list(my_cards), 1 - me: list(rival_cards)}
    results_ = list(results)
    player = next_player
    first: tuple[int, int] | None = (1 - next_player, table) if table is not None else None
    while True:
        rival_card = first[1] if first else None
        card = choose_card(tuple(remaining[player]), rival_card, tuple(results_), player, mano)
        remaining[player].remove(card)
        if first is None:
            first = (player, card)
            player = 1 - player
            continue
        leader, lead_card = first
        if card == lead_card:
            winner = None
        else:
            winner = leader if lead_card < card else player
        results_.append(winner)
        first = None
        result = hand_winner(results_, mano)
        if result is not None:
            return result
        player = mano if winner is None else winner


# --- Lo que un jugador sabe de la mano ------------------------------------------------------------


class _AnyCards(list):
    """Mano de reemplazo para rehacer el historial sin conocer las cartas del rival."""

    def remove(self, value):
        pass


def actors(state: TrucoState) -> list[int]:
    """Quién hizo cada acción del historial, rehaciéndolo con información pública."""
    replay = TrucoState(state.get_game())
    replay.mano = state.mano
    replay.hands = [(), ()]
    replay.remaining = [_AnyCards(), _AnyCards()]
    replay.window = [state.mano, 1 - state.mano]
    replay.card_player = state.mano
    players = []
    for action in state.history_actions:
        players.append(replay.current_player())
        replay._apply_action(action)
    return players


@dataclass(frozen=True)
class View:
    """Información de la mano visible para el jugador `me`."""

    niveles: int
    me: int
    mano: int
    my_hand: tuple[int, ...]
    my_cards: tuple[int, ...]  # cartas que todavía tengo
    rival_played: tuple[int, ...]  # cartas que ya tiró el rival
    table: int | None  # carta ya tirada en la baza en curso (por quien no es next_player)
    results: tuple[int | None, ...]
    next_player: int  # quien tira la próxima carta
    rival_aggressions: int  # cantos y "quiero" del rival hasta ahora

    @classmethod
    def of(cls, state: TrucoState) -> "View":
        me = state.current_player()
        rival_played, aggressions = [], 0
        for player, action in zip(actors(state), state.history_actions, strict=True):
            if player == me:
                continue
            if action < NUM_TRUCO_LEVELS:
                rival_played.append(action)
            elif action in _AGGRESSIVE:
                aggressions += 1
        return cls(
            niveles=state.niveles,
            me=me,
            mano=state.mano,
            my_hand=state.hands[me],
            my_cards=tuple(sorted(state.remaining[me])),
            rival_played=tuple(sorted(rival_played)),
            table=state.table[0][1] if state.table else None,
            results=tuple(state.results),
            next_player=state.card_player,
            rival_aggressions=aggressions,
        )


@lru_cache(maxsize=_CACHE_SIZE)
def equity(view: View, lectura: float) -> float:
    """Probabilidad de ganar la mano, promediando sobre las manos posibles del rival.

    Cada mano posible pesa según su probabilidad de reparto dada mi mano, multiplicada por
    percentil(mano) ** lectura por cada canto o "quiero" del rival: quien canta suele tener cartas.
    """
    played = Counter(view.rival_played)
    total = wins = 0.0
    for index, p in deal_outcomes(view.niveles, view.my_hand):
        rival_hand = hands(view.niveles)[index]
        counts = Counter(rival_hand)
        if any(counts[level] < n for level, n in played.items()):
            continue
        weight = p * hand_percentile(view.niveles, rival_hand) ** (lectura * view.rival_aggressions)
        rival_cards = tuple(sorted((counts - played).elements()))
        winner = playout(
            view.my_cards,
            rival_cards,
            view.table,
            view.results,
            view.next_player,
            view.me,
            view.mano,
        )
        total += weight
        wins += weight * (winner == view.me)
    return wins / total


# --- Política heurística --------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicPolicy:
    """Canta según su probabilidad de ganar, acepta según las cuentas de puntos y a veces farolea.

    - umbrales: equity mínima para cantar truco, retruco y vale cuatro.
    - farol: probabilidad de cantar sin llegar al umbral (el doble si ya ganó una baza).
    - lectura: cuánto cree que el rival tiene cartas cuando canta o acepta (0 = no le cree nada).
    - margen: equity extra que exige para aceptar, sobre el punto de equilibrio.
    """

    nombre: str = "equilibrado"
    umbrales: tuple[float, float, float] = (0.65, 0.75, 0.85)
    farol: float = 0.05
    lectura: float = 1.0
    margen: float = 0.0

    def call_probability(self, view: View, call: Action, e: float) -> float:
        """Probabilidad de cantar `call` (o subir a `call`) con equity `e`."""
        if e >= self.umbrales[_CALL_VALUE[call] - 2]:
            return 1.0
        won_baza = view.me in view.results
        return min(1.0, self.farol * (2 if won_baza else 1))

    def probabilities(self, state: TrucoState) -> dict[int, float]:
        legal = state.legal_actions()
        if state.pending is None and not state.window:  # toca tirar una carta
            view = View.of(state)
            return {choose_card(view.my_cards, view.table, view.results, view.me, view.mano): 1.0}

        view = View.of(state)
        e = equity(view, self.lectura)
        if state.pending is not None:
            # Rechazar pierde v - 1; aceptar gana o pierde v: conviene si v(2e - 1) > -(v - 1).
            value = _CALL_VALUE[state.pending]
            accept = e > 1 / (2 * value) + self.margen
            raise_ = _NEXT_CALL.get(value)
            p_raise = self.call_probability(view, raise_, e) if raise_ in legal else 0.0
            rest = Action.QUIERO if accept else Action.NO_QUIERO
            return {raise_: p_raise, rest: 1 - p_raise} if p_raise else {rest: 1.0}

        call = _NEXT_CALL.get(state.points)
        p_call = self.call_probability(view, call, e) if call in legal else 0.0
        return {call: p_call, Action.PASO: 1 - p_call} if p_call else {Action.PASO: 1.0}

    def action_probabilities(self, states: list[TrucoState]) -> np.ndarray:
        probs = np.zeros((len(states), NUM_ACTIONS))
        for i, state in enumerate(states):
            for action, p in self.probabilities(state).items():
                probs[i, action] = p
        return probs


CONSERVADOR = HeuristicPolicy("conservador", (0.70, 0.80, 0.90), farol=0.0, margen=0.10)
EQUILIBRADO = HeuristicPolicy()
AGRESIVO = HeuristicPolicy("agresivo", (0.55, 0.65, 0.75), farol=0.15, lectura=0.5, margen=-0.05)
RIVALES = {
    "aleatorio": RandomPolicy(),
    **{p.nombre: p for p in (CONSERVADOR, EQUILIBRADO, AGRESIVO)},
}
