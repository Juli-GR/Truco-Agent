"""One hand of truco with envido as an OpenSpiel game (stage 3).

Rules in docs/reglas.md. Unlike truco.py, chance deals real cards: the suit matters for the tantos.
The truco part is delegated to a state of the full-deck truco game (truco.py), which receives only
the truco actions, each card replaced by its level; this module adds the envido on top. So the
truco rules live in a single place, and the inner state (`HandState.truco`) is what the truco
heuristics already understand.

Actions 0 to NUM_CARDS - 1 play that card, then `Action`. QUIERO and NO_QUIERO answer the envido
while one is pending, the truco otherwise.
"""

import enum
from functools import cache
from itertools import combinations

import numpy as np
import pyspiel

from truco.games import envido, truco
from truco.games.cards import (
    HAND_SIZE,
    MAX_TANTOS,
    NUM_CARDS,
    NUM_TRUCO_LEVELS,
    SUITS,
    rank,
    suit,
    tantos,
    truco_level,
)


class Action(enum.IntEnum):
    """Actions that are not playing a card."""

    PASO = NUM_CARDS
    TRUCO = NUM_CARDS + 1
    RETRUCO = NUM_CARDS + 2
    VALE_CUATRO = NUM_CARDS + 3
    QUIERO = NUM_CARDS + 4
    NO_QUIERO = NUM_CARDS + 5
    MAZO = NUM_CARDS + 6
    ENVIDO = NUM_CARDS + 7
    REAL_ENVIDO = NUM_CARDS + 8


_TO_TRUCO = {a: truco.Action[a.name] for a in Action if a.name in truco.Action.__members__}
_FROM_TRUCO = {t: a for a, t in _TO_TRUCO.items()}
_TO_ENVIDO = {Action.ENVIDO: envido.Action.ENVIDO, Action.REAL_ENVIDO: envido.Action.REAL_ENVIDO}
_FROM_ENVIDO = {e: a for a, e in _TO_ENVIDO.items()}
_ACTION_TOKENS = {
    Action.PASO: "-",
    Action.TRUCO: "T",
    Action.RETRUCO: "R",
    Action.VALE_CUATRO: "V",
    Action.QUIERO: "Q",
    Action.NO_QUIERO: "N",
    Action.MAZO: "M",
    Action.ENVIDO: "E",
    Action.REAL_ENVIDO: "RE",
}

_NUM_PLAYERS = 2
_NUM_ACTIONS = NUM_CARDS + len(Action)
_NUM_TANTOS = MAX_TANTOS + 1
_MAX_ENVIDO_ACTIONS = 4  # E, E, R and the answer
_MAX_POINTS = 7 + 4  # E-E-R accepted plus vale cuatro

_GAME_TYPE = pyspiel.GameType(
    short_name="truco_con_envido",
    long_name="Truco: una mano con envido",
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


@cache
def hands() -> tuple[tuple[int, ...], ...]:
    """Every hand of 3 cards, as sorted tuples; a deal is an index into this tuple."""
    return tuple(combinations(range(NUM_CARDS), HAND_SIZE))


@cache
def _level_hand_index() -> dict[tuple[int, ...], int]:
    """Index of each hand of levels in truco.hands(NUM_TRUCO_LEVELS)."""
    return {h: i for i, h in enumerate(truco.hands(NUM_TRUCO_LEVELS))}


def levels(cards) -> tuple[int, ...]:
    """Sorted truco levels of some cards."""
    return tuple(sorted(truco_level(c) for c in cards))


def card_name(card: int) -> str:
    """Short name of a card: rank and the suit's initial, e.g. 7e = 7 de espada."""
    return f"{rank(card)}{SUITS[suit(card)][0]}"


def hand_string(hand) -> str:
    return ",".join(card_name(c) for c in hand)


class HandGame(pyspiel.Game):
    def __init__(self, params=None):
        self.truco_game = pyspiel.load_game("truco")
        game_info = pyspiel.GameInfo(
            num_distinct_actions=_NUM_ACTIONS,
            max_chance_outcomes=len(hands()),
            num_players=_NUM_PLAYERS,
            min_utility=-float(_MAX_POINTS),
            max_utility=float(_MAX_POINTS),
            utility_sum=0.0,
            max_game_length=self.truco_game.max_game_length() + _MAX_ENVIDO_ACTIONS,
        )
        super().__init__(_GAME_TYPE, game_info, params or {})

    def new_initial_state(self):
        return HandState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return HandObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params,
            self.max_game_length(),
        )


class _ClonedOnCopy:
    """Holds a pyspiel state so that deep copies clone it. Copying a HandState (clone, child)
    deep-copies its attributes, and copy.deepcopy of a Python pyspiel state gives a broken copy
    (its actions are not applied to it), while clone() works."""

    def __init__(self, state: pyspiel.State):
        self.state = state

    def __deepcopy__(self, memo):
        return _ClonedOnCopy(self.state.clone())


class HandState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._truco = _ClonedOnCopy(game.truco_game.new_initial_state())
        self.mano: int | None = None
        self.cards: list[tuple[int, ...]] = []  # dealt hands, one per player
        self.cards_left: list[list[int]] = []  # cards each player still holds
        self.history_actions: list[int] = []
        self.envido_calls: list[envido.Action] = []
        self.envido_caller: int | None = None  # who made the last envido call
        self.envido_winner: int | None = None  # set when the envido is resolved
        self.envido_points = 0
        self.declared: list[int | None] = [None, None]  # tantos each player declared (public)

    @property
    def truco(self) -> truco.TrucoState:
        """The truco part of the hand, on levels."""
        return self._truco.state

    # --- Envido ---

    @property
    def envido_pending(self) -> bool:
        return bool(self.envido_calls) and self.envido_winner is None

    def can_call_envido(self, player: int) -> bool:
        """Whether `player` may start the envido now (assuming it is in a window or answering a
        truco): only once, before playing a card and while no truco has been accepted."""
        return (
            not self.envido_calls
            and len(self.cards_left[player]) == HAND_SIZE
            and self.truco.points == 1
        )

    def _resolve_envido(self, accepted: bool):
        if not accepted:  # the last caller scores what was accumulated before
            self.envido_winner = self.envido_caller
            self.envido_points = envido.rejected_points(self.envido_calls)
            return
        mano, pie = self.mano, 1 - self.mano
        t = [tantos(self.cards[p]) for p in range(_NUM_PLAYERS)]
        self.declared[mano] = t[mano]
        if t[pie] > t[mano]:  # the pie declares only if it wins; otherwise "son buenas"
            self.declared[pie] = t[pie]
            self.envido_winner = pie
        else:
            self.envido_winner = mano
        self.envido_points = envido.accepted_points(self.envido_calls)

    # --- pyspiel.State ---

    def current_player(self):
        if self.truco.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if len(self.cards) < _NUM_PLAYERS:
            return pyspiel.PlayerId.CHANCE
        if self.envido_pending:
            return 1 - self.envido_caller
        return self.truco.current_player()

    def _legal_actions(self, player):
        if self.envido_pending:
            raises = envido._RAISES.get(tuple(self.envido_calls), [])
            return sorted([Action.QUIERO, Action.NO_QUIERO, *(_FROM_ENVIDO[r] for r in raises)])
        actions = []
        for a in self.truco.legal_actions():
            if a < NUM_TRUCO_LEVELS:
                actions.extend(c for c in self.cards_left[player] if truco_level(c) == a)
            else:
                actions.append(_FROM_TRUCO[a])
        calling = self.truco.pending is not None or self.truco.window
        if calling and self.can_call_envido(player):
            actions.extend(_TO_ENVIDO)
        return sorted(actions)

    def chance_outcomes(self):
        if self.mano is None:
            return [(0, 0.5), (1, 0.5)]
        dealt = set(sum(self.cards, ()))
        outcomes = [i for i, h in enumerate(hands()) if dealt.isdisjoint(h)]
        return [(i, 1 / len(outcomes)) for i in outcomes]

    def _apply_action(self, action):
        if self.is_chance_node():
            if self.mano is None:
                self.mano = action
                self.truco.apply_action(action)
                return
            hand = hands()[action]
            self.cards.append(hand)
            self.cards_left.append(list(hand))
            self.truco.apply_action(_level_hand_index()[levels(hand)])
            return

        player = self.current_player()
        self.history_actions.append(action)
        if action in _TO_ENVIDO:  # does not use the player's chance in the window
            self.envido_calls.append(_TO_ENVIDO[action])
            self.envido_caller = player
        elif self.envido_pending:
            self._resolve_envido(action == Action.QUIERO)
        elif action < NUM_CARDS:
            self.cards_left[player].remove(action)
            self.truco.apply_action(truco_level(action))
        else:
            self.truco.apply_action(_TO_TRUCO[action])

    def history_string(self) -> str:
        """History: cards played and calls (- paso, T, R, V, Q, N, M mazo, E, RE real envido)."""
        return " ".join(
            card_name(a) if a < NUM_CARDS else _ACTION_TOKENS[a] for a in self.history_actions
        )

    def declared_string(self) -> str:
        """Tantos declared in an accepted envido: mano's / pie's ("buenas" if it did not declare),
        empty if none."""
        mano = self.declared[self.mano] if self.mano is not None else None
        if mano is None:
            return ""
        pie = self.declared[1 - self.mano]
        return f"{mano}/{'buenas' if pie is None else pie}"

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self.mano is None:
                return f"Mano:{action}"
            return f"Cartas:{hand_string(hands()[action])}"
        if action < NUM_CARDS:
            return f"Tira:{card_name(action)}"
        return Action(action).name

    def is_terminal(self):
        return self.truco.is_terminal()

    def returns(self):
        if not self.is_terminal():
            return [0.0, 0.0]
        result = self.truco.returns()
        if self.envido_winner is not None:
            result[self.envido_winner] += self.envido_points
            result[1 - self.envido_winner] -= self.envido_points
        return result

    def __str__(self):
        cards = " | ".join(hand_string(h) for h in self.cards)
        return f"mano:{self.mano} cartas:{cards} {self.history_string()}"


class HandObserver:
    """Observer with OpenSpiel's PyObserver interface.

    As in the other games, the observation always includes the whole history, so the observation
    with and without perfect recall coincide. The private cards are the hand dealt to the player.
    """

    def __init__(self, iig_obs_type, params, max_game_length):
        if params:
            raise ValueError(f"Unsupported observation parameters: {params}")

        pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
        if iig_obs_type.public_info:
            pieces.append(("mano", _NUM_PLAYERS, (_NUM_PLAYERS,)))
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("cartas", NUM_CARDS, (NUM_CARDS,)))
        if iig_obs_type.public_info:
            size = max_game_length * _NUM_ACTIONS
            pieces.append(("historia", size, (max_game_length, _NUM_ACTIONS)))
            pieces.append(("tantos", 2 * _NUM_TANTOS, (_NUM_PLAYERS, _NUM_TANTOS)))

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
        if "cartas" in self.dict and len(state.cards) > player:
            self.dict["cartas"][list(state.cards[player])] = 1
        if "historia" in self.dict:
            for turn, action in enumerate(state.history_actions):
                self.dict["historia"][turn, action] = 1
        if "tantos" in self.dict:
            for p, t in enumerate(state.declared):
                if t is not None:
                    self.dict["tantos"][p, t] = 1

    def string_from(self, state, player):
        pieces = [f"p{player}"]
        if "mano" in self.dict and state.mano is not None:
            pieces.append(f"mano:{state.mano}")
        if "cartas" in self.dict and len(state.cards) > player:
            pieces.append(f"cartas:{hand_string(state.cards[player])}")
        if "historia" in self.dict:
            pieces.append(f"historia:{state.history_string()}")
            if state.declared_string():
                pieces.append(f"tantos:{state.declared_string()}")
        return " ".join(pieces)


pyspiel.register_game(_GAME_TYPE, HandGame)
