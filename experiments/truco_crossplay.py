"""Stage 2b: cross-play matrix between the checkpoints of a self-play run.

Loads `--checkpoints` checkpoints evenly spaced along the training run `--run-id` (from
experiments/truco_selfplay.py) and plays every pair, plus the fixed opponents of
`truco.agents.truco_rivales`, with duplicate deals. Nothing is trained.

Logs to MLflow (experiment "truco-crossplay"), with the training run's id as a parameter:
- `crossplay.csv` and `crossplay_stderr.csv`: M[i, j] = points per hand of row i against column j.
- `heatmap.png`: the matrix. A red lower triangle means newer checkpoints beat older ones;
  a pale block means they play alike (convergence); stripes or a checkerboard mean cycling.
- `strength.png`, and per checkpoint (step = PPO update): points against the other
  checkpoints, against the previous one, against the first one and against each fixed opponent.
- `intransitive_triads`: triples a > b > c > a (rock-paper-scissors) among the checkpoints,
  each by more than two standard errors. 0 means the checkpoints can be ranked.

Usage, from the repo root:
    python experiments/truco_crossplay.py --run-id <training run id> --checkpoints 20
"""

import argparse
import csv
import tempfile
from math import comb
from pathlib import Path

import mlflow
import numpy as np
import pyspiel

import truco.games.truco  # noqa: F401  (registers the game)
from truco.agents.truco_rivales import RIVALES
from truco.algorithms.ppo import load_ppo_policy
from truco.evaluation.crossplay import (
    crossplay,
    intransitive_triads,
    mean_against_others,
    select_evenly,
)
from truco.plots import crossplay_heatmap_figure, crossplay_strength_figure
from truco.tracking import setup_experiment


def update_of(path: Path) -> int:
    return int(path.stem.removeprefix("update_"))


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *(round(float(v), 4) for v in row)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="truco-selfplay run to analyse")
    parser.add_argument("--checkpoints", type=int, default=20, help="evenly spaced, at least 2")
    parser.add_argument("--deals", type=int, default=1000, help="duplicate deals per pair")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_experiment("truco-crossplay")
    training = mlflow.get_run(args.run_id)
    params = training.data.params
    # Runs logged before the encoding moved into the game call it "encoding".
    observacion = params.get("observacion", params.get("encoding"))
    if observacion is None:
        raise SystemExit("The run predates PPONetwork and the encodings: it cannot be loaded")
    game = pyspiel.load_game("truco", {"observacion": observacion})

    run_name = f"crossplay {args.run_id[:8]}"
    with mlflow.start_run(run_name=run_name), tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.log_params(
            {
                "training_run_id": args.run_id,
                "training_observacion": observacion,
                **{f"training_{k}": params[k] for k in ("decay", "hidden", "steps")},
                **vars(args),
            }
        )
        directory = Path(
            mlflow.artifacts.download_artifacts(
                run_id=args.run_id, artifact_path="checkpoints", dst_path=str(tmp)
            )
        )
        paths = select_evenly(
            sorted(directory.glob("update_*.pt"), key=update_of), args.checkpoints
        )
        updates = [update_of(p) for p in paths]
        policies = [load_ppo_policy(p, game, int(params["hidden"])) for p in paths]
        labels = [f"u{u}" for u in updates] + list(RIVALES)
        print(f"{len(paths)} checkpoints + {len(RIVALES)} fixed opponents", flush=True)

        means, stderrs = crossplay(
            game, policies + list(RIVALES.values()), args.deals, args.seed, verbose=True
        )

        n = len(paths)
        block, block_err = means[:n, :n], stderrs[:n, :n]
        triads = intransitive_triads(block, block_err)
        mlflow.log_metrics(
            {
                "intransitive_triads": triads,
                "intransitive_fraction": triads / comb(n, 3) if n >= 3 else 0.0,
            }
        )

        series = {"vs other checkpoints": mean_against_others(block)}
        series.update({f"vs {name}": means[:n, n + k] for k, name in enumerate(RIVALES)})
        for k, update in enumerate(updates):
            metrics = {name.replace(" ", "_"): float(values[k]) for name, values in series.items()}
            metrics["vs_first"] = float(block[k, 0])
            if k > 0:
                metrics["vs_previous"] = float(block[k, k - 1])
            mlflow.log_metrics(metrics, step=update)

        for name, matrix in (("crossplay.csv", means), ("crossplay_stderr.csv", stderrs)):
            write_matrix(tmp / name, labels, matrix)
            mlflow.log_artifact(str(tmp / name))
        mlflow.log_figure(crossplay_heatmap_figure(means, labels, n), "heatmap.png")
        shown = {k: v for k, v in series.items() if k != "vs aleatorio"}  # other scale
        mlflow.log_figure(crossplay_strength_figure(updates, shown), "strength.png")
        print(f"intransitive triads: {triads} of {comb(n, 3)}", flush=True)


if __name__ == "__main__":
    main()
