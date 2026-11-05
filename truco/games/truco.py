"""Una mano de truco sin envido como juego de OpenSpiel (etapas 2a y 2b).

Reglas en docs/reglas.md. Sin envido, el palo solo importa a través de la jerarquía del truco, así
que el azar reparte niveles de la jerarquía en lugar de cartas: es una abstracción sin pérdida. El
parámetro `niveles` deja en el mazo solo los niveles más altos (14 = mazo completo).
"""

import enum
from functools import cache
from itertools import combinations_with_replacement
from math import comb, prod

import numpy as np
import pyspiel

from truco.games.cards import HAND_SIZE, NUM_TRUCO_LEVELS, TRUCO_LEVEL_CARDS, TRUCO_LEVEL_NAMES


class Action(enum.IntEnum):
    """Acciones que no son tirar carta. Las acciones 0 a NUM_TRUCO_LEVELS - 1 tiran una carta de
    ese nivel."""

    PASO = NUM_TRUCO_LEVELS
    TRUCO = NUM_TRUCO_LEVELS + 1
    RETRUCO = NUM_TRUCO_LEVELS + 2
    VALE_CUATRO = NUM_TRUCO_LEVELS + 3
    QUIERO = NUM_TRUCO_LEVELS + 4
    NO_QUIERO = NUM_TRUCO_LEVELS + 5
    MAZO = NUM_TRUCO_LEVELS + 6


_CALL_VALUE = {Action.TRUCO: 2, Action.RETRUCO: 3, Action.VALE_CUATRO: 4}
# Canto que sube la mano desde su valor actual.
_NEXT_CALL = {1: Action.TRUCO, 2: Action.RETRUCO, 3: Action.VALE_CUATRO}
_ACTION_TOKENS = {
    Action.PASO: "-",
    Action.TRUCO: "T",
    Action.RETRUCO: "R",
    Action.VALE_CUATRO: "V",
    Action.QUIERO: "Q",
    Action.NO_QUIERO: "N",
    Action.MAZO: "M",
}

_NUM_PLAYERS = 2
_NUM_ACTIONS = NUM_TRUCO_LEVELS + len(Action)
_NUM_CARDS_PLAYED = _NUM_PLAYERS * HAND_SIZE
# Cada carta tirada, una oportunidad de cantar por jugador antes de cada carta y hasta tres
# respuestas a cantos.
_MAX_GAME_LENGTH = _NUM_CARDS_PLAYED * (1 + _NUM_PLAYERS) + len(_CALL_VALUE)
_MIN_LEVELS = 5  # con menos niveles no hay cartas para repartir 3 a cada jugador
_DEFAULT_PARAMS = {"niveles": NUM_TRUCO_LEVELS}

_GAME_TYPE = pyspiel.GameType(
    short_name="truco",
    long_name="Truco: una mano sin envido",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS,
)


@cache
def hands(niveles: int) -> tuple[tuple[int, ...], ...]:
    """Manos posibles con los `niveles` más altos, como tuplas ordenadas de niveles."""
    return tuple(
        h
        for h in combinations_with_replacement(range(niveles), HAND_SIZE)
        if all(h.count(level) <= len(TRUCO_LEVEL_CARDS[level]) for level in h)
    )


@cache
def deal_outcomes(niveles: int, dealt: tuple[int, ...]) -> tuple[tuple[int, float], ...]:
    """Probabilidad de cada mano (índice en `hands`) dados los niveles ya repartidos."""
    copies = [len(TRUCO_LEVEL_CARDS[level]) - dealt.count(level) for level in range(niveles)]
    total = comb(sum(copies), HAND_SIZE)
    outcomes = []
    for i, hand in enumerate(hands(niveles)):
        ways = prod(comb(copies[level], hand.count(level)) for level in set(hand))
        if ways:
            outcomes.append((i, ways / total))
    return tuple(outcomes)


def hand_string(hand: tuple[int, ...]) -> str:
    return ",".join(TRUCO_LEVEL_NAMES[level] for level in hand)


def hand_winner(results: list[int | None], mano: int) -> int | None:
    """Ganador de la mano según las bazas terminadas (None = parda), o None si no está definido."""
    for player in range(_NUM_PLAYERS):
        if results.count(player) == 2:
            return player
    decided = [r for r in results if r is not None]
    if None in results and decided:
        return decided[0]  # con alguna parda gana quien ganó la primera baza no empatada
    if len(results) == HAND_SIZE:
        return mano  # tres pardas
    return None


class TrucoGame(pyspiel.Game):
    def __init__(self, params=None):
        params = {**_DEFAULT_PARAMS, **(params or {})}
        niveles = params["niveles"]
        if not _MIN_LEVELS <= niveles <= NUM_TRUCO_LEVELS:
            raise ValueError(f"niveles debe estar entre {_MIN_LEVELS} y {NUM_TRUCO_LEVELS}")
        game_info = pyspiel.GameInfo(
            num_distinct_actions=_NUM_ACTIONS,
            max_chance_outcomes=len(hands(niveles)),
            num_players=_NUM_PLAYERS,
            min_utility=-4.0,
            max_utility=4.0,
            utility_sum=0.0,
            max_game_length=_MAX_GAME_LENGTH,
        )
        super().__init__(_GAME_TYPE, game_info, params)
        self.niveles = niveles

    def new_initial_state(self):
        return TrucoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return TrucoObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params
        )


class TrucoState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.niveles = game.niveles
        self.mano: int | None = None
        self.hands: list[tuple[int, ...]] = []  # manos repartidas, una por jugador
        self.remaining: list[list[int]] = []  # cartas que le quedan a cada jugador
        self.table: list[tuple[int, int]] = []  # (jugador, nivel) de la baza en curso
        self.results: list[int | None] = []  # ganador de cada baza terminada, None si fue parda
        self.points = 1  # valor de la mano según los cantos aceptados
        self.quiero: int | None = None  # quien puede subir el último canto aceptado
        self.pending: Action | None = None  # canto que espera respuesta
        self.caller: int | None = None  # quien hizo el canto pendiente
        self.window: list[int] = []  # jugadores que todavía no usaron su oportunidad en la ventana
        self.card_player: int | None = None  # quien tira al cerrar la ventana
        self.history_actions: list[int] = []
        self._game_over = False
        self._returns = [0.0, 0.0]

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if len(self.hands) < _NUM_PLAYERS:
            return pyspiel.PlayerId.CHANCE
        if self.pending is not None:
            return 1 - self.caller
        if self.window:
            return self.window[0]
        return self.card_player

    def _legal_actions(self, player):
        if self.pending is not None:
            raise_ = _NEXT_CALL.get(_CALL_VALUE[self.pending])
            return sorted([Action.QUIERO, Action.NO_QUIERO, *([raise_] if raise_ else [])])
        if self.window:
            actions = [Action.PASO, Action.MAZO]
            if self.points in _NEXT_CALL and self.quiero in (None, player):
                actions.append(_NEXT_CALL[self.points])
            return sorted(actions)
        return sorted(set(self.remaining[player]))

    def chance_outcomes(self):
        if self.mano is None:
            return [(0, 0.5), (1, 0.5)]
        return list(deal_outcomes(self.niveles, tuple(sorted(sum(self.hands, ())))))

    def _apply_action(self, action):
        if self.is_chance_node():
            if self.mano is None:
                self.mano = action
                return
            hand = hands(self.niveles)[action]
            self.hands.append(hand)
            self.remaining.append(list(hand))
            if len(self.hands) == _NUM_PLAYERS:
                self.window = [self.mano, 1 - self.mano]
                self.card_player = self.mano
            return

        player = self.current_player()
        self.history_actions.append(action)
        if action < NUM_TRUCO_LEVELS:
            self._play_card(player, action)
            return

        action = Action(action)
        if self.pending is None:
            self.window.pop(0)  # el jugador usa su oportunidad en la ventana
        if action == Action.MAZO:
            self._finish(1 - player, self.points)
        elif action == Action.QUIERO:
            self.points = _CALL_VALUE[self.pending]
            self.quiero = player
            self.pending = None
        elif action == Action.NO_QUIERO:
            self._finish(self.caller, _CALL_VALUE[self.pending] - 1)
        elif action in _CALL_VALUE:
            self.pending = action
            self.caller = player

    def _play_card(self, player: int, level: int):
        self.remaining[player].remove(level)
        self.table.append((player, level))
        if len(self.table) == 1:
            next_player = 1 - player
        else:
            (first, first_level), (_, second_level) = self.table
            if first_level == second_level:
                winner = None
            else:
                winner = first if first_level < second_level else 1 - first
            self.results.append(winner)
            self.table = []
            hand = hand_winner(self.results, self.mano)
            if hand is not None:
                self._finish(hand, self.points)
                return
            next_player = self.mano if winner is None else winner
        self.card_player = next_player
        self.window = [player, 1 - player]  # primero actúa quien tiró la última carta

    def history_string(self) -> str:
        """Historial: niveles de las cartas tiradas y cantos (- paso, T, R, V, Q, N, M mazo)."""
        return " ".join(
            TRUCO_LEVEL_NAMES[a] if a < NUM_TRUCO_LEVELS else _ACTION_TOKENS[a]
            for a in self.history_actions
        )

    def _finish(self, winner: int, points: int):
        self._game_over = True
        self._returns[winner] = float(points)
        self._returns[1 - winner] = -float(points)

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self.mano is None:
                return f"Mano:{action}"
            return f"Cartas:{hand_string(hands(self.niveles)[action])}"
        if action < NUM_TRUCO_LEVELS:
            return f"Tira:{TRUCO_LEVEL_NAMES[action]}"
        return Action(action).name

    def is_terminal(self):
        return self._game_over

    def returns(self):
        return list(self._returns)

    def __str__(self):
        cards = " | ".join(hand_string(h) for h in self.hands)
        return f"mano:{self.mano} cartas:{cards} {self.history_string()}"


class TrucoObserver:
    """Observador según la interfaz PyObserver de OpenSpiel.

    Como en el envido, la observación incluye siempre el historial completo, así que la observación
    con y sin memoria perfecta coinciden. Las cartas privadas son la mano repartida al jugador; las
    que ya tiró se deducen del historial.
    """

    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Parámetros de observación no soportados: {params}")

        pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
        if iig_obs_type.public_info:
            pieces.append(("mano", _NUM_PLAYERS, (_NUM_PLAYERS,)))
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("cartas", NUM_TRUCO_LEVELS, (NUM_TRUCO_LEVELS,)))
        if iig_obs_type.public_info:
            size = _MAX_GAME_LENGTH * _NUM_ACTIONS
            pieces.append(("historia", size, (_MAX_GAME_LENGTH, _NUM_ACTIONS)))

        self.tensor = np.zeros(sum(size for _, size, _ in pieces), np.float32)
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index : index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        self.tensor.fill(0)
        self.dict["player"][player] = 1
        if "mano" in self.dict and state.mano is not None:
            self.dict["mano"][state.mano] = 1
        if "cartas" in self.dict and len(state.hands) > player:
            for level in state.hands[player]:
                self.dict["cartas"][level] += 1
        if "historia" in self.dict:
            for turn, action in enumerate(state.history_actions):
                self.dict["historia"][turn, action] = 1

    def string_from(self, state, player):
        pieces = [f"p{player}"]
        if "mano" in self.dict and state.mano is not None:
            pieces.append(f"mano:{state.mano}")
        if "cartas" in self.dict and len(state.hands) > player:
            pieces.append(f"cartas:{hand_string(state.hands[player])}")
        if "historia" in self.dict:
            pieces.append(f"historia:{state.history_string()}")
        return " ".join(pieces)


def parse_information_state(info_state: str) -> dict:
    """Separa un information state string de TrucoObserver en sus partes."""
    pieces, history = info_state.split(" historia:")
    player, mano, cards = pieces.split(" ")
    return {
        "jugador": int(player.removeprefix("p")),
        "es_mano": int(mano.removeprefix("mano:") == player.removeprefix("p")),
        "cartas": cards.removeprefix("cartas:"),
        "historia": history,
    }


pyspiel.register_game(_GAME_TYPE, TrucoGame)
