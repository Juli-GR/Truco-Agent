"""Self-play: turns the two-player truco game into a single-agent environment.

PPO and DQN learn in an environment with a single agent. To use them in a two-player game, the
opponent is frozen and becomes part of the environment: its moves happen between the agent's turns.
Each game samples which seat the agent takes (player 0 or 1) and which opponent it faces from an
`OpponentPool` (the latest frozen copy of the agent, or any earlier checkpoint).

`SelfPlayEnv` follows the interface of OpenSpiel's `SyncVectorEnv`, so OpenSpiel's PPO and DQN
use it unchanged. The agent always sees itself as player 0 of a one-player game: the observation,
legal actions and reward are those of its seat, whichever seat that is.

Opponents are `Policy` objects, like in the arena: they receive all the states where they have to
play at once, so a frozen network moves in every game with a single forward pass.
"""

import copy

import numpy as np
import pyspiel
import torch
from open_spiel.python.rl_environment import StepType, TimeStep
from torch import nn

from truco.evaluation.arena import Policy, sample_actions, sample_deal
from truco.games.truco import TrucoState

AGENT_ID = 0  # the agent's player_id in the single-agent view (whatever its seat)


class NetworkPolicy:
    """Policy given by a network that maps information state tensors to one score per action.

    With `greedy=False` the scores are logits and the policy is their softmax over the legal
    actions (PPO's actor). With `greedy=True` it plays the legal action with the highest score
    (DQN's Q-network).
    """

    def __init__(self, network: nn.Module, greedy: bool = False, device: str = "cpu"):
        self.network = network
        self.greedy = greedy
        self.device = device

    @classmethod
    def snapshot(cls, network: nn.Module, greedy: bool = False) -> "NetworkPolicy":
        """Frozen copy of `network`: later training does not change it."""
        frozen = copy.deepcopy(network).cpu().eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        return cls(frozen, greedy)

    def action_probabilities(self, states: list[TrucoState]) -> np.ndarray:
        obs = np.array(
            [s.information_state_tensor(s.current_player()) for s in states], dtype=np.float32
        )
        legal = np.zeros((len(states), states[0].num_distinct_actions()), dtype=bool)
        for i, state in enumerate(states):
            legal[i, state.legal_actions()] = True

        with torch.no_grad():
            scores = self.network(torch.as_tensor(obs, device=self.device)).cpu().numpy()
        scores = np.where(legal, scores.astype(np.float64), -np.inf)
        if self.greedy:
            probs = np.zeros_like(scores)
            probs[np.arange(len(states)), scores.argmax(axis=1)] = 1.0
            return probs
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)


class OpponentPool:
    """Opponents the agent trains against, sampled with geometric decay by age.

    The opponent added k additions before the last one is sampled with probability proportional
    to decay**k. `decay=0` always samples the last one (naive self-play); `decay=1` samples all of
    them uniformly (a league of past checkpoints). In between, recent opponents weigh more, and
    the weight halves every log(1/2) / log(decay) additions.
    """

    def __init__(self, opponents: list[Policy] | None = None, decay: float = 0.0):
        if not 0 <= decay <= 1:
            raise ValueError(f"decay must be between 0 and 1, got {decay}")
        self.opponents: list[Policy] = list(opponents or [])
        self.decay = decay

    def __len__(self) -> int:
        return len(self.opponents)

    def add(self, opponent: Policy):
        self.opponents.append(opponent)

    def probabilities(self) -> np.ndarray:
        """Probability of sampling each opponent, in the order they were added."""
        ages = np.arange(len(self.opponents))[::-1]  # the last one added has age 0
        weights = float(self.decay) ** ages  # numpy gives 0.0**0 = 1: decay = 0 keeps the last
        return weights / weights.sum()

    def sample(self, rng: np.random.Generator) -> Policy:
        if not self.opponents:
            raise ValueError("The opponent pool is empty")
        return self.opponents[rng.choice(len(self.opponents), p=self.probabilities())]


class SelfPlayEnv:
    """`num_envs` games played in parallel, each one against an opponent from the pool.

    Every game starts with a fresh deal, a random seat for the agent and an opponent sampled from
    the pool. The pool is read only when a game starts, so opponents added mid-game take effect
    from the next game on.
    """

    def __init__(self, game: pyspiel.Game, pool: OpponentPool, num_envs: int, seed: int = 0):
        self.game = game
        self.pool = pool
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.states: list[TrucoState] = [None] * num_envs
        self.seats = np.zeros(num_envs, dtype=int)  # agent's seat (player id) in each game
        self.opponents: list[Policy] = [None] * num_envs

    def __len__(self) -> int:
        return self.num_envs

    @property
    def info_state_shape(self) -> tuple[int]:
        return (self.game.information_state_tensor_size(),)

    @property
    def num_actions(self) -> int:
        return self.game.num_distinct_actions()

    def reset(self) -> list[TimeStep]:
        """Starts a new game in every environment."""
        self._start_games(list(range(self.num_envs)))
        return [self._time_step(i, StepType.FIRST) for i in range(self.num_envs)]

    def step(
        self, step_outputs, reset_if_done: bool = True
    ) -> tuple[list[TimeStep], list[list[float]], list[bool], list[TimeStep]]:
        """Applies one agent action per game, then plays the opponents until it is the agent's
        turn again or the game ends.

        `step_outputs` are OpenSpiel `StepOutput`s (or anything with an `action` attribute).
        Returns, like `SyncVectorEnv.step`: the time steps (already reset for finished games if
        `reset_if_done`), the rewards, the done flags and the time steps before resetting.
        """
        if len(step_outputs) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(step_outputs)}")
        for i, output in enumerate(step_outputs):
            state = self.states[i]
            if state.is_terminal():
                raise ValueError(f"Game {i} is over: call reset() or step with reset_if_done")
            action = int(output.action)
            if action not in state.legal_actions():
                raise ValueError(f"Illegal action {action} in game {i}: {state}")
            state.apply_action(action)

        self._play_opponents(list(range(self.num_envs)))
        time_steps = [
            self._time_step(i, StepType.LAST if s.is_terminal() else StepType.MID)
            for i, s in enumerate(self.states)
        ]
        rewards = [ts.rewards for ts in time_steps]
        dones = [ts.last() for ts in time_steps]
        unreset_time_steps = time_steps
        if reset_if_done:
            finished = [i for i, done in enumerate(dones) if done]
            self._start_games(finished)
            time_steps = list(time_steps)
            for i in finished:
                time_steps[i] = self._time_step(i, StepType.FIRST)
        return time_steps, rewards, dones, unreset_time_steps

    def _start_games(self, indices: list[int]):
        """New game in each of `indices`, played until the agent's first turn.

        If the opponent ends a game before the agent plays (it goes to the mazo), the agent made
        no decision in it, so that game is discarded and dealt again.
        """
        while indices:
            for i in indices:
                state = self.game.new_initial_state()
                for action in sample_deal(self.game, self.rng):
                    state.apply_action(action)
                self.states[i] = state
                self.seats[i] = self.rng.integers(2)
                self.opponents[i] = self.pool.sample(self.rng)
            self._play_opponents(indices)
            indices = [i for i in indices if self.states[i].is_terminal()]

    def _play_opponents(self, indices: list[int]):
        """Plays the opponents' moves until every game ends or reaches the agent's turn."""
        waiting = [i for i in indices if self._opponent_to_play(i)]
        while waiting:
            # Groups the games by opponent, to call each one once with all its states.
            groups: dict[int, tuple[Policy, list[int]]] = {}
            for i in waiting:
                groups.setdefault(id(self.opponents[i]), (self.opponents[i], []))[1].append(i)
            for opponent, group in groups.values():
                probs = opponent.action_probabilities([self.states[i] for i in group])
                for i, action in zip(group, sample_actions(probs, self.rng), strict=True):
                    self.states[i].apply_action(int(action))
            waiting = [i for i in waiting if self._opponent_to_play(i)]

    def _opponent_to_play(self, i: int) -> bool:
        state = self.states[i]
        return not state.is_terminal() and state.current_player() != self.seats[i]

    def _time_step(self, i: int, step_type: StepType) -> TimeStep:
        """The game seen by the agent, as a one-player game where it is player 0."""
        state, seat = self.states[i], int(self.seats[i])
        last = step_type == StepType.LAST
        observations = {
            "info_state": [state.information_state_tensor(seat)],
            "legal_actions": [[] if last else state.legal_actions(seat)],
            "current_player": pyspiel.PlayerId.TERMINAL if last else AGENT_ID,
        }
        reward = state.returns()[seat] if last else 0.0
        return TimeStep(observations, [reward], [0.0 if last else 1.0], step_type)
