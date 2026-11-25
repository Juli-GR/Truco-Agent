"""Stage 2b: NFSP on one hand of truco (full deck, no envido).

One NFSP agent per seat (`truco.algorithms.nfsp`), both learning at once, one game at a time. The
policy evaluated is the average policy, which is the one that approaches a Nash equilibrium. The
defaults follow OpenSpiel's NFSP example for Leduc poker, with larger networks.

Exploitability is not measured: computing it on the full deck is not feasible (see
notas_teoria.md). The average policy is evaluated against the fixed opponents instead.

Logs to MLflow (experiment "truco-nfsp"):
- Every `--log-every` episodes: each seat's supervised (average policy) and RL (DQN) losses, and
  the training speed.
- Every `--eval-every` episodes, the average policy's points per hand against the fixed opponents
  (`truco.agents.truco_rivales`), with duplicate deals (`truco.evaluation.arena`).
- Every `--checkpoint-every` episodes, the average policy as a checkpoint in `checkpoints/`
  (`truco.algorithms.nfsp.save_average_policy`).

Usage, from the repo root:
    python experiments/truco_nfsp.py --episodes 5000000 --observacion full
"""

import argparse
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pyspiel
import torch
from open_spiel.python import rl_environment

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.truco_rivales import RIVALES
from truco.algorithms.nfsp import average_policy, make_agents, save_average_policy, train_episode
from truco.evaluation.arena import play_match
from truco.games.truco import OBSERVACIONES
from truco.tracking import setup_experiment


def save_checkpoint(agents, episode: int, directory: Path) -> None:
    path = directory / f"episode_{episode:09d}.pt"
    save_average_policy(agents, path)
    mlflow.log_artifact(str(path), "checkpoints")


def evaluate(game, policy, num_deals: int, seed: int) -> dict[str, float]:
    """Points per hand of `policy` against each fixed opponent."""
    metrics = {}
    for name, rival in RIVALES.items():
        result = play_match(game, policy, rival, num_deals, seed)
        metrics[f"vs_{name}"] = result.mean
        metrics[f"vs_{name}_stderr"] = result.stderr
    return metrics


def loss_metrics(agents) -> dict[str, float]:
    """Each seat's last supervised and RL losses (missing until that network starts learning)."""
    metrics = {}
    for player, agent in enumerate(agents):
        sl_loss, rl_loss = agent.loss
        if sl_loss is not None:
            metrics[f"sl_loss_p{player}"] = float(sl_loss)
        if rl_loss is not None:
            metrics[f"rl_loss_p{player}"] = float(rl_loss)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=2_000_000, help="training games")
    parser.add_argument("--observacion", choices=OBSERVACIONES, default="full")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--anticipatory", type=float, default=0.1, help="P(best response)")
    parser.add_argument("--reservoir-capacity", type=int, default=2_000_000)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--rl-learning-rate", type=float, default=0.01)
    parser.add_argument("--sl-learning-rate", type=float, default=0.01)
    parser.add_argument("--epsilon-start", type=float, default=0.06)
    parser.add_argument("--epsilon-end", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learn-every", type=int, default=64, help="in agent steps")
    parser.add_argument("--update-target-every", type=int, default=19_200, help="in agent steps")
    parser.add_argument("--log-every", type=int, default=10_000, help="in episodes")
    parser.add_argument("--eval-every", type=int, default=100_000, help="in episodes")
    parser.add_argument("--checkpoint-every", type=int, default=250_000, help="in episodes")
    parser.add_argument("--eval-deals", type=int, default=500, help="duplicate deals per rival")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # NFSP samples its mode and DQN explores with np.random
    game = pyspiel.load_game("truco", {"observacion": args.observacion})
    env = rl_environment.Environment(game, seed=args.seed)
    agents = make_agents(
        game,
        args.hidden,
        seed=args.seed,
        anticipatory_param=args.anticipatory,
        reservoir_buffer_capacity=args.reservoir_capacity,
        replay_buffer_capacity=args.replay_capacity,
        rl_learning_rate=args.rl_learning_rate,
        sl_learning_rate=args.sl_learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        batch_size=args.batch_size,
        learn_every=args.learn_every,
        update_target_network_every=args.update_target_every,
        device=args.device,
    )

    setup_experiment("truco-nfsp")
    with mlflow.start_run(), tempfile.TemporaryDirectory() as tmp:
        checkpoints = Path(tmp)
        mlflow.log_params({"game": "truco", "algo": "nfsp", **vars(args)})
        save_checkpoint(agents, 0, checkpoints)

        train_time = 0.0  # evaluation and checkpoints are not training time
        for episode in range(1, args.episodes + 1):
            start = time.perf_counter()
            train_episode(env, agents)
            train_time += time.perf_counter() - start

            if episode % args.log_every == 0:
                mlflow.log_metrics(
                    {"train_episodes_per_s": episode / train_time, **loss_metrics(agents)},
                    step=episode,
                )

            if episode % args.checkpoint_every == 0 or episode == args.episodes:
                save_checkpoint(agents, episode, checkpoints)

            if episode % args.eval_every == 0 or episode == args.episodes:
                metrics = evaluate(game, average_policy(agents), args.eval_deals, args.seed)
                mlflow.log_metrics(metrics, step=episode)
                summary = "  ".join(
                    f"{k} {v:+.2f}" for k, v in metrics.items() if not k.endswith("stderr")
                )
                print(f"episode {episode:9d}/{args.episodes}  {summary}", flush=True)


if __name__ == "__main__":
    main()
