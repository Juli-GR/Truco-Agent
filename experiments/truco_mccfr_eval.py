"""Evaluates an MCCFR run (experiments/truco_mccfr.py) against the fixed opponents.

Loads the last checkpoint of the run dir, plays its average strategy on the exact OpenSpiel game
against each fixed opponent (`truco.agents.truco_rivales`) with duplicate deals
(`truco.evaluation.arena`), and logs the points per hand to the run's MLflow run, as
vs_<opponent> and vs_<opponent>_stderr (the same names as the NFSP and PPO scripts) at the step of
the checkpoint's iteration.

The checkpoint is mapped read-only instead of loaded into RAM, so it can run next to the training
(only the parts used are read). Since the training keeps only its last checkpoint, `--watch`
evaluates every new one as it appears, which builds the learning curve; run it in its own tmux
window and stop it with Ctrl+C.

Usage, from the repo root:
    python experiments/truco_mccfr_eval.py results/mccfr/<run dir>
    python experiments/truco_mccfr_eval.py results/mccfr/<run dir> --watch
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import pyspiel
from truco_mccfr import CONFIG_FILE, STATE_FILE, last_checkpoint

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.truco_rivales import RIVALES
from truco.algorithms import mccfr
from truco.evaluation.arena import play_match
from truco.tracking import setup_experiment


def evaluate(checkpoint: Path, config: dict, num_deals: int, seed: int) -> None:
    iteration = json.loads((checkpoint / STATE_FILE).read_text())["iteration"]
    print(f"{checkpoint.name} (iteration {iteration:,}), {num_deals:,} duplicate deals", flush=True)
    start = time.perf_counter()
    tables = mccfr.Tables.load(checkpoint, read_only=True)
    policy = mccfr.MCCFRPolicy(tables, mccfr.KEY_MODES[config["key"]])
    game = pyspiel.load_game("truco", {"niveles": config["niveles"]})

    client = mlflow.MlflowClient()
    run_id = config["mlflow_run_id"]
    for name, rival in RIVALES.items():
        result = play_match(game, policy, rival, num_deals, seed)
        client.log_metric(run_id, f"vs_{name}", result.mean, step=iteration)
        client.log_metric(run_id, f"vs_{name}_stderr", result.stderr, step=iteration)
        print(f"  vs {name:12s} {result.mean:+.4f} ± {result.stderr:.4f}", flush=True)
    seconds = time.perf_counter() - start
    client.log_metric(run_id, "eval_seconds", seconds, step=iteration)
    print(f"  [{seconds:.0f} s]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--deals", type=int, default=5000, help="duplicate deals per opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", action="store_true", help="evaluate each new checkpoint")
    parser.add_argument("--interval-minutes", type=float, default=10.0)
    args = parser.parse_args()

    config = json.loads((args.run_dir / CONFIG_FILE).read_text())
    setup_experiment("truco-mccfr")  # points MLflow to the repo's database

    evaluated = None
    while True:
        checkpoint = last_checkpoint(args.run_dir)
        if checkpoint is not None and checkpoint.name != evaluated:
            try:
                evaluate(checkpoint, config, args.deals, args.seed)
                evaluated = checkpoint.name
            except FileNotFoundError:  # replaced by a newer one while loading: take that one
                continue
        elif checkpoint is None:
            print("No checkpoint yet.", flush=True)
        if not args.watch:
            break
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
