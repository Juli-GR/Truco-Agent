"""Valor esperado del envido cuando ambos jugadores siguen una estrategia dada."""

from collections import defaultdict
from collections.abc import Callable

import pyspiel

from truco.games.envido import Action, EnvidoState

Strategy = Callable[[EnvidoState], dict[Action, float]]
ROLES = ("mano", "pie")


def strategy_from_rows(rows: list[dict]) -> Strategy:
    """Convierte las filas de estrategia.csv en una función: estado -> probabilidad por acción."""
    table = {
        (r["jugador"], r["es_mano"], r["tantos"], r["historia"]): {a: r[a.name] for a in Action}
        for r in rows
    }

    def strategy(state: EnvidoState) -> dict[Action, float]:
        p = state.current_player()
        return table[(p, int(state.mano == p), state.tantos[p], state.history_string())]

    return strategy


def _value(state: EnvidoState, strategy: Strategy) -> list[float]:
    """Retorno esperado de cada jugador desde `state`, recorriendo el árbol exacto."""
    if state.is_terminal():
        return state.returns()
    probs = strategy(state)
    total = [0.0, 0.0]
    for action in state.legal_actions():
        p = probs.get(Action(action), 0.0)
        if p > 0:
            child = _value(state.child(action), strategy)
            total = [t + p * v for t, v in zip(total, child, strict=True)]
    return total


def expected_value_by_tantos(
    game: pyspiel.Game, strategy: Strategy, jugador: int = 0
) -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    """Puntos esperados de `jugador` según sus tantos, como mano y como pie.

    Devuelve (por_tantos, total): por_tantos["mano"][t] es el valor esperado condicionado a tener
    t tantos siendo mano; total["mano"] es el valor esperado siendo mano, promediando los tantos.
    """
    sums = defaultdict(float)
    weights = defaultdict(float)
    root = game.new_initial_state()
    for mano, _ in root.chance_outcomes():
        after_mano = root.child(mano)
        role = "mano" if mano == jugador else "pie"
        for deal, p_deal in after_mano.chance_outcomes():
            state = after_mano.child(deal)
            t = state.tantos[jugador]
            sums[(role, t)] += p_deal * _value(state, strategy)[jugador]
            weights[(role, t)] += p_deal

    by_tantos = {role: {} for role in ROLES}
    total = dict.fromkeys(ROLES, 0.0)
    for (role, t), w in sorted(weights.items()):
        by_tantos[role][t] = sums[(role, t)] / w
        total[role] += sums[(role, t)]  # los pesos de cada rol suman 1
    return by_tantos, total
