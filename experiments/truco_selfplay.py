"""Stage 2b: PPO in self-play on one hand of truco (full deck, no envido).

The agent trains against frozen copies of itself (`truco.algorithms.selfplay`). The pool starts
with the initial random network and gets a new copy every `--snapshot-every` PPO updates; `--decay`
sets how often old copies are sampled (0 = always the latest, 1 = all of them alike).
`--observacion` is the game parameter that sets what the network observes
(truco/games/observations.py).

Logs to MLflow (experiment "truco-selfplay"):
- PPO's losses after every update, and the mean reward of the training games.
- Every `--eval-every` updates, the points per hand against the fixed opponents
  (`truco.agents.truco_rivales`), with duplicate deals (`truco.evaluation.arena`).
- Every copy added to the pool, as a checkpoint in `checkpoints/` (the network's state_dict). The
  cross-play matrix of step 6 is built from them.

Usage, from the repo root:
    python experiments/truco_selfplay.py --steps 5000000 --decay 0.95 --observacion full
"""

import argparse
import functools
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pyspiel
import torch
from open_spiel.python.pytorch.ppo import PPO

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.policy import NetworkPolicy
from truco.agents.truco_rivales import RIVALES
from truco.algorithms.ppo import PPONetwork
from truco.algorithms.selfplay import AGENT_ID, OpponentPool, SelfPlayEnv
from truco.evaluation.arena import play_match
from truco.games.truco import OBSERVACIONES
from truco.tracking import setup_experiment


class MlflowWriter:
    """Stands in for the TensorBoard SummaryWriter that OpenSpiel's PPO logs its losses to."""

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        mlflow.log_metric(tag, float(value), step=step)


def save_checkpoint(network: torch.nn.Module, update: int, directory: Path) -> None:
    path = directory / f"update_{update:06d}.pt"
    torch.save(network.state_dict(), path)
    mlflow.log_artifact(str(path), "checkpoints")


def evaluate(game, policy: NetworkPolicy, num_deals: int, seed: int) -> dict[str, float]:
    """Points per hand of `policy` against each fixed opponent."""
    metrics = {}
    for name, rival in RIVALES.items():
        result = play_match(game, policy, rival, num_deals, seed)
        metrics[f"vs_{name}"] = result.mean
        metrics[f"vs_{name}_stderr"] = result.stderr
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2_000_000, help="agent decisions in total")
    parser.add_argument("--num-envs", type=int, default=64, help="games played in parallel")
    parser.add_argument("--steps-per-batch", type=int, default=128, help="per game, per update")
    parser.add_argument("--observacion", choices=OBSERVACIONES, default="full")
    parser.add_argument("--hidden", type=int, default=256, help="units per hidden layer")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--decay", type=float, default=0.0, help="pool decay, from 0 to 1")
    parser.add_argument("--snapshot-every", type=int, default=10, help="in PPO updates")
    parser.add_argument("--eval-every", type=int, default=25, help="in PPO updates")
    parser.add_argument("--eval-deals", type=int, default=500, help="duplicate deals per rival")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # OpenSpiel's PPO shuffles minibatches with np.random
    game = pyspiel.load_game("truco", {"observacion": args.observacion})
    pool = OpponentPool(decay=args.decay)
    env = SelfPlayEnv(game, pool, args.num_envs, seed=args.seed)

    setup_experiment("truco-selfplay")
    with mlflow.start_run(), tempfile.TemporaryDirectory() as tmp:
        checkpoints = Path(tmp)
        mlflow.log_params({"game": "truco", "algo": "ppo", **vars(args)})
        agent = PPO(
            env.info_state_shape,
            env.num_actions,
            num_players=1,
            player_id=AGENT_ID,
            num_envs=args.num_envs,
            steps_per_batch=args.steps_per_batch,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            device=args.device,
            writer=MlflowWriter(),
            agent_fn=functools.partial(PPONetwork, hidden=args.hidden),
        )
        num_updates = args.steps // agent.batch_size
        mlflow.log_param("num_updates", num_updates)

        env.pool.add(NetworkPolicy.snapshot(agent.network.actor))
        save_checkpoint(agent.network, 0, checkpoints)

        time_steps = env.reset()
        rewards_since_log: list[float] = []
        games = 0
        train_time = 0.0
        for update in range(1, num_updates + 1):
            start = time.perf_counter()
            for _ in range(args.steps_per_batch):
                outputs = agent.step(time_steps)
                time_steps, rewards, dones, _ = env.step(outputs)
                agent.post_step(rewards, dones)
                rewards_since_log += [r[0] for r, done in zip(rewards, dones, strict=True) if done]
            agent.learn(time_steps)
            train_time += time.perf_counter() - start

            step = agent.total_steps_done
            games += len(rewards_since_log)
            mlflow.log_metrics(
                {
                    "train_reward": float(np.mean(rewards_since_log)),
                    "games": games,
                    "train_games_per_s": games / train_time,
                },
                step=step,
            )
            rewards_since_log = []

            if update % args.snapshot_every == 0 or update == num_updates:
                env.pool.add(NetworkPolicy.snapshot(agent.network.actor))
                save_checkpoint(agent.network, update, checkpoints)
                mlflow.log_metric("pool_size", len(env.pool), step=step)

            if update % args.eval_every == 0 or update == num_updates:
                policy = NetworkPolicy(agent.network.actor, device=args.device)
                metrics = evaluate(game, policy, args.eval_deals, args.seed)
                mlflow.log_metrics(metrics, step=step)
                summary = "  ".join(
                    f"{k} {v:+.2f}" for k, v in metrics.items() if not k.endswith("stderr")
                )
                print(f"update {update:5d}/{num_updates}  games {games:9d}  {summary}", flush=True)


if __name__ == "__main__":
    main()
