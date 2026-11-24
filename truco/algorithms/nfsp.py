"""NFSP on truco: one OpenSpiel NFSP agent per seat, and the average policy they learn.

NFSP (Heinrich & Silver, 2016) gives each player two networks: a best response, trained with DQN
against the opponent's current behaviour, and an average policy, trained with supervised learning
on the agent's own best-response actions (kept in a reservoir buffer). In each episode the agent
plays its best response with probability `anticipatory_param` and its average policy otherwise. In
two-player zero-sum games the average policies approach a Nash equilibrium, so they are the policy
we evaluate.

Both seats learn at the same time, so the frozen-opponent `SelfPlayEnv` does not apply: training
uses OpenSpiel's `rl_environment.Environment`, one game at a time.
"""

from pathlib import Path

import pyspiel
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.dqn import MLP
from open_spiel.python.pytorch.nfsp import NFSP

from truco.agents.policy import NetworkPolicy, SeatPolicy


def make_agents(
    game: pyspiel.Game, hidden_layers: list[int], seed: int = 0, **kwargs
) -> list[NFSP]:
    """One NFSP agent per seat. `kwargs` go to OpenSpiel's `NFSP` (and from there to its DQN)."""
    return [
        NFSP(
            player,
            game.information_state_tensor_size(),
            game.num_distinct_actions(),
            hidden_layers,
            seed=seed + 10 * player,
            **kwargs,
        )
        for player in range(game.num_players())
    ]


def train_episode(env: rl_environment.Environment, agents: list[NFSP]) -> list[float]:
    """Plays one training game between the agents, which learn from it. Returns the returns."""
    time_step = env.reset()
    while not time_step.last():
        player = time_step.observations["current_player"]
        output = agents[player].step(time_step)
        time_step = env.step([output.action])
    for agent in agents:  # each agent sees the final reward and closes its episode
        agent.step(time_step)
    return time_step.rewards


def _average_network(agent: NFSP) -> torch.nn.Module:
    # OpenSpiel exposes no accessor for the average-policy network.
    return agent._avg_network


def average_policy(agents: list[NFSP], frozen: bool = True) -> SeatPolicy:
    """The agents' average policy, each seat played by its own agent.

    With `frozen`, a copy that later training does not change (to use as an opponent).
    """
    networks = [_average_network(agent) for agent in agents]
    if frozen:
        return SeatPolicy([NetworkPolicy.snapshot(network) for network in networks])
    device = next(networks[0].parameters()).device
    return SeatPolicy([NetworkPolicy(network, device=str(device)) for network in networks])


def save_average_policy(agents: list[NFSP], path: Path) -> None:
    """Saves the state_dicts of the average-policy networks, one per seat."""
    torch.save([_average_network(agent).state_dict() for agent in agents], path)


def load_nfsp_policy(path: Path, game: pyspiel.Game, hidden_layers: list[int]) -> SeatPolicy:
    """Frozen average policy from a `save_average_policy` checkpoint.

    `game` must have the `observacion` the networks were trained with.
    """
    policies = []
    for state_dict in torch.load(path, map_location="cpu"):
        network = MLP(
            game.information_state_tensor_size(), hidden_layers, game.num_distinct_actions()
        )
        network.load_state_dict(state_dict)
        policies.append(NetworkPolicy.snapshot(network))
    return SeatPolicy(policies)
