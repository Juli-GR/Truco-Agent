"""Genera los gráficos de estrategia del envido para runs ya existentes y los guarda en MLflow.

Lee el artefacto estrategia.csv de cada run y agrega los gráficos en graficos/ del mismo run, junto
con las métricas de valor esperado.

Uso, desde la raíz del repo:
    python experiments/envido_plots.py <run_id> [<run_id> ...]
"""

import argparse
import csv

import mlflow
import pyspiel
from envido_cfr import log_strategy_analysis

from truco.games.envido import Action
from truco.tracking import setup_experiment

_INT_COLUMNS = ("jugador", "es_mano", "tantos")


def load_strategy(run_id: str) -> list[dict]:
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="estrategia.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in _INT_COLUMNS:
            r[col] = int(r[col])
        for a in Action:
            r[a.name] = float(r[a.name])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--jugador", type=int, default=0)
    args = parser.parse_args()

    game = pyspiel.load_game("truco_envido")
    setup_experiment("envido-cfr")
    for run_id in args.run_ids:
        rows = load_strategy(run_id)
        with mlflow.start_run(run_id=run_id):
            log_strategy_analysis(game, rows, args.jugador)
        print(f"{run_id}: gráficos guardados")


if __name__ == "__main__":
    main()
