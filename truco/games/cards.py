"""Mazo español de 40 cartas y cálculo de tantos de envido."""

from functools import cache
from itertools import combinations

import numpy as np

SUITS = ("espada", "basto", "oro", "copa")
RANKS = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)

# Una carta es un entero en [0, 40): carta = palo * 10 + índice del rango.
NUM_CARDS = len(SUITS) * len(RANKS)
HAND_SIZE = 3
MAX_TANTOS = 33


def rank(card: int) -> int:
    return RANKS[card % len(RANKS)]


def suit(card: int) -> int:
    return card // len(RANKS)


def card_to_string(card: int) -> str:
    return f"{rank(card)} de {SUITS[suit(card)]}"


def envido_value(card: int) -> int:
    """Valor de una carta para el envido: las figuras (10, 11, 12) valen 0."""
    r = rank(card)
    return 0 if r >= 10 else r


def tantos(hand: tuple[int, ...]) -> int:
    """Tantos de envido de una mano.

    Con dos o más cartas del mismo palo: 20 + las dos más altas de ese palo.
    Sin dos cartas del mismo palo: el valor de la carta más alta.
    """
    best = max(envido_value(c) for c in hand)
    for s in range(len(SUITS)):
        values = sorted((envido_value(c) for c in hand if suit(c) == s), reverse=True)
        if len(values) >= 2:
            best = max(best, 20 + values[0] + values[1])
    return best


@cache
def tantos_joint_counts() -> np.ndarray:
    """Cantidad de repartos con tantos (t0, t1), enumerando todos los repartos de 3 + 3 cartas.

    Devuelve una matriz de enteros de (MAX_TANTOS + 1) x (MAX_TANTOS + 1) donde la entrada
    [t0, t1] cuenta los pares ordenados (mano del jugador 0, mano del jugador 1) disjuntos.
    """
    hands = list(combinations(range(NUM_CARDS), HAND_SIZE))
    masks = np.array([sum(1 << c for c in h) for h in hands], dtype=np.int64)
    hand_tantos = np.array([tantos(h) for h in hands], dtype=np.int64)

    counts = np.zeros((MAX_TANTOS + 1, MAX_TANTOS + 1), dtype=np.int64)
    for mask, t0 in zip(masks, hand_tantos, strict=True):
        disjoint = (masks & mask) == 0
        counts[t0] += np.bincount(hand_tantos[disjoint], minlength=MAX_TANTOS + 1)
    return counts


@cache
def tantos_joint_distribution() -> np.ndarray:
    """Probabilidad exacta de cada par de tantos (t0, t1) al repartir 3 cartas a cada jugador."""
    counts = tantos_joint_counts()
    return counts / counts.sum()
