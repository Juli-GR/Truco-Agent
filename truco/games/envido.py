"""Envido aislado como juego de OpenSpiel (etapa 1).

Reglas en docs/reglas.md. El azar elige primero quién es mano y después reparte directamente el par
de tantos (t0, t1) con su probabilidad exacta, en lugar de repartir cartas. Es una abstracción sin
pérdida: en el envido aislado la mano solo importa a través de los tantos.
"""

import enum

import numpy as np
import pyspiel

from truco.games.cards import MAX_TANTOS, tantos_joint_distribution


class Action(enum.IntEnum):
    PASO = 0
    ENVIDO = 1
    REAL_ENVIDO = 2
    QUIERO = 3
    NO_QUIERO = 4


_ACTION_LETTERS = "PERQN"
_CALL_POINTS = {Action.ENVIDO: 2, Action.REAL_ENVIDO: 3}
# Subidas legales según la secuencia de cantos hasta el momento (sin Falta Envido).
_RAISES = {
    (Action.ENVIDO,): [Action.ENVIDO, Action.REAL_ENVIDO],
    (Action.ENVIDO, Action.ENVIDO): [Action.REAL_ENVIDO],
}

_NUM_PLAYERS = 2
_NUM_TANTOS = MAX_TANTOS + 1
_MAX_GAME_LENGTH = 5  # P, E, E, R, Q

_GAME_TYPE = pyspiel.GameType(
    short_name="truco_envido",
    long_name="Truco: envido aislado",
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
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=_NUM_TANTOS * _NUM_TANTOS,
    num_players=_NUM_PLAYERS,
    min_utility=-7.0,
    max_utility=7.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH,
)


def accepted_points(calls: list[Action]) -> int:
    """Puntos en juego si se acepta la secuencia de cantos."""
    return sum(_CALL_POINTS[c] for c in calls)


def rejected_points(calls: list[Action]) -> int:
    """Puntos para quien cantó último si no se acepta: lo acumulado antes, con mínimo 1."""
    return max(1, accepted_points(calls[:-1]))


class EnvidoGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self):
        return EnvidoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return EnvidoObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params
        )


class EnvidoState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.mano: int | None = None
        self.tantos: tuple[int, int] | None = None
        self.calls: list[Action] = []
        self.history_actions: list[Action] = []
        self._next_player = 0
        self._game_over = False
        self._returns = [0.0, 0.0]

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self.mano is None or self.tantos is None:
            return pyspiel.PlayerId.CHANCE
        return self._next_player

    def _legal_actions(self, player):
        if not self.calls:
            return [Action.PASO, Action.ENVIDO, Action.REAL_ENVIDO]
        raises = _RAISES.get(tuple(self.calls), [])
        return sorted([Action.QUIERO, Action.NO_QUIERO, *raises])

    def chance_outcomes(self):
        if self.mano is None:
            return [(0, 0.5), (1, 0.5)]
        dist = tantos_joint_distribution()
        t0, t1 = np.nonzero(dist)
        return [(int(a * _NUM_TANTOS + b), float(dist[a, b])) for a, b in zip(t0, t1, strict=True)]

    def _apply_action(self, action):
        if self.is_chance_node():
            if self.mano is None:
                self.mano = action
            else:
                self.tantos = divmod(action, _NUM_TANTOS)
                self._next_player = self.mano
            return

        player = self._next_player
        other = 1 - player
        action = Action(action)
        self.history_actions.append(action)

        if action == Action.PASO:
            if player == self.mano:
                self._next_player = other
            else:
                self._game_over = True
        elif action in _CALL_POINTS:
            self.calls.append(action)
            self._next_player = other
        elif action == Action.QUIERO:
            self._finish(self._envido_winner(), accepted_points(self.calls))
        else:  # NO_QUIERO: gana quien cantó último.
            self._finish(other, rejected_points(self.calls))

    def history_string(self) -> str:
        """Historial de cantos: P(aso), E(nvido), R(eal envido), Q(uiero), N(o quiero)."""
        return "".join(_ACTION_LETTERS[a] for a in self.history_actions)

    def _envido_winner(self) -> int:
        t = self.tantos
        if t[0] == t[1]:
            return self.mano
        return 0 if t[0] > t[1] else 1

    def _finish(self, winner: int, points: int):
        self._game_over = True
        self._returns[winner] = float(points)
        self._returns[1 - winner] = -float(points)

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self.mano is None:
                return f"Mano:{action}"
            return "Tantos:{}-{}".format(*divmod(action, _NUM_TANTOS))
        return Action(action).name

    def is_terminal(self):
        return self._game_over

    def returns(self):
        return list(self._returns)

    def __str__(self):
        return f"mano:{self.mano} tantos:{self.tantos} {self.history_string()}"


class EnvidoObserver:
    """Observador según la interfaz PyObserver de OpenSpiel.

    El juego es tan chico que la observación incluye siempre el historial completo de cantos, así
    que la observación con y sin memoria perfecta coinciden.
    """

    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Parámetros de observación no soportados: {params}")

        pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
        if iig_obs_type.public_info:
            pieces.append(("mano", _NUM_PLAYERS, (_NUM_PLAYERS,)))
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("tantos", _NUM_TANTOS, (_NUM_TANTOS,)))
        if iig_obs_type.public_info:
            size = _MAX_GAME_LENGTH * len(Action)
            pieces.append(("historia", size, (_MAX_GAME_LENGTH, len(Action))))

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
        if "tantos" in self.dict and state.tantos is not None:
            self.dict["tantos"][state.tantos[player]] = 1
        if "historia" in self.dict:
            for turn, action in enumerate(state.history_actions):
                self.dict["historia"][turn, action] = 1

    def string_from(self, state, player):
        pieces = [f"p{player}"]
        if "mano" in self.dict and state.mano is not None:
            pieces.append(f"mano:{state.mano}")
        if "tantos" in self.dict and state.tantos is not None:
            pieces.append(f"tantos:{state.tantos[player]}")
        if "historia" in self.dict and state.history_actions:
            pieces.append(state.history_string())
        return " ".join(pieces)


pyspiel.register_game(_GAME_TYPE, EnvidoGame)
