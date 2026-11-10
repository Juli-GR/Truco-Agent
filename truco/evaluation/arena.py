"""Arena: enfrenta dos políticas en partidas duplicadas.

Cada reparto (quién es mano y las cartas de cada uno) se juega dos veces con los asientos
invertidos, así la suerte de las cartas se cancela y hacen falta muchas menos manos para separar
a dos políticas. El resultado de un reparto es el promedio de los puntos de A en sus dos partidas.

Las partidas se juegan en lotes: en cada paso, cada política recibe juntos todos los estados en los
que le toca jugar, así una red evalúa cientos de estados en una sola pasada.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pyspiel

from truco.games.truco import TrucoState


class Policy(Protocol):
    def action_probabilities(self, states: list[TrucoState]) -> np.ndarray:
        """Matriz (estados x acciones) con la probabilidad de cada acción."""
        ...


@dataclass(frozen=True)
class MatchResult:
    mean: float  # puntos por mano de A (promedio de los repartos)
    stderr: float  # error estándar de `mean`
    deals: int  # repartos jugados (cada uno, dos partidas)


def sample_deal(game: pyspiel.Game, rng: np.random.Generator) -> list[int]:
    """Acciones de azar de una mano: quién es mano y la mano de cada jugador."""
    state = game.new_initial_state()
    actions = []
    while state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes(), strict=True)
        action = int(rng.choice(outcomes, p=probs))
        state.apply_action(action)
        actions.append(action)
    return actions


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Calcula una acción por fila de `probs`, muestreada según esa fila."""
    cumulative = np.cumsum(probs, axis=1)
    u = rng.random(len(probs)) * cumulative[:, -1]
    actions = (cumulative <= u[:, None]).sum(axis=1)
    return np.minimum(actions, probs.shape[1] - 1)


def play_games(
    game: pyspiel.Game, seats: list[tuple[Policy, Policy]], deals: list[list[int]], rng
) -> np.ndarray:
    """Juega en paralelo una partida por elemento: seats[i] = (jugador 0, jugador 1) y deals[i]
    las acciones de azar. Devuelve los puntos del jugador 0 de cada partida."""
    states = []
    for deal in deals:
        state = game.new_initial_state()
        for action in deal:
            state.apply_action(action)
        states.append(state)

    active = [i for i, s in enumerate(states) if not s.is_terminal()]
    while active:
        # Agrupa por la política que tiene que jugar, para llamarla una sola vez.
        groups: dict[int, tuple[Policy, list[int]]] = {}
        for i in active:
            policy = seats[i][states[i].current_player()]
            groups.setdefault(id(policy), (policy, []))[1].append(i)
        for policy, indices in groups.values():
            probs = policy.action_probabilities([states[i] for i in indices])
            for i, action in zip(indices, sample_actions(probs, rng), strict=True):
                states[i].apply_action(int(action))
        active = [i for i in active if not states[i].is_terminal()]
    return np.array([s.returns()[0] for s in states])


def play_match(
    game: pyspiel.Game,
    a: Policy,
    b: Policy,
    num_deals: int,
    seed: int = 0,
    batch_size: int = 512,
) -> MatchResult:
    """Puntos por mano de `a` contra `b` en `num_deals` repartos duplicados."""
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, num_deals, batch_size):
        deals = [sample_deal(game, rng) for _ in range(min(batch_size, num_deals - start))]
        seats = [(a, b)] * len(deals) + [(b, a)] * len(deals)
        returns = play_games(game, seats, deals + deals, rng)
        a_first, a_second = returns[: len(deals)], -returns[len(deals) :]
        values.append((a_first + a_second) / 2)
    values = np.concatenate(values)
    return MatchResult(
        float(values.mean()), float(values.std(ddof=1) / np.sqrt(num_deals)), num_deals
    )


def crossplay(
    game: pyspiel.Game, policies: list[Policy], num_deals: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de cruces: M[i, j] = puntos por mano de policies[i] contra policies[j].

    Devuelve (medias, errores estándar). Solo se juega cada par i < j; el resto sale de que el
    juego es de suma cero: M[j, i] = -M[i, j].
    """
    n = len(policies)
    means, stderrs = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            result = play_match(game, policies[i], policies[j], num_deals, seed + i * n + j)
            means[i, j], means[j, i] = result.mean, -result.mean
            stderrs[i, j] = stderrs[j, i] = result.stderr
    return means, stderrs
