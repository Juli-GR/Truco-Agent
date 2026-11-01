"""Etapa 1: CFR sobre el envido aislado.

Registra en MLflow (experimento "envido-cfr") la curva de exploitabilidad, la estrategia promedio
final como CSV (una fila por information set), un gráfico de la estrategia por cada historia y el
valor esperado según los tantos.

Uso, desde la raíz del repo:
    python experiments/envido_cfr.py --iterations 500 --eval-every 10
"""

import argparse
import csv
import tempfile
import time
from pathlib import Path

import mlflow
import pyspiel
from open_spiel.python import policy as policy_lib

from truco.evaluation.envido import expected_value_by_tantos, strategy_from_rows
from truco.games.envido import Action
from truco.plots import envido_strategy_figures, envido_value_figure
from truco.tracking import setup_experiment

SOLVERS = {"cfr": pyspiel.CFRSolver, "cfr_plus": pyspiel.CFRPlusSolver}


def policy_rows(game: pyspiel.Game, policy: pyspiel.Policy) -> list[dict]:
    """Estrategia por information set: quién juega, si es mano, sus tantos, el historial y
    la probabilidad de cada acción."""
    rows = []
    for state in policy_lib.TabularPolicy(game).states:
        player = state.current_player()
        probs = policy.action_probabilities(state)
        row = {
            "jugador": player,
            "es_mano": int(state.mano == player),
            "tantos": state.tantos[player],
            "historia": state.history_string(),
        }
        row.update({a.name: round(probs.get(a, 0.0), 4) for a in Action})
        rows.append(row)
    return sorted(rows, key=lambda r: (r["jugador"], -r["es_mano"], r["historia"], r["tantos"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def log_strategy_analysis(game: pyspiel.Game, rows: list[dict], jugador: int = 0) -> None:
    """Guarda en el run activo los gráficos de estrategia y de valor esperado de `jugador`."""
    for name, fig in envido_strategy_figures(rows, jugador).items():
        mlflow.log_figure(fig, f"graficos/{name}.png")
    by_tantos, total = expected_value_by_tantos(game, strategy_from_rows(rows), jugador)
    mlflow.log_figure(envido_value_figure(by_tantos, total), "graficos/valor_esperado.png")
    mlflow.log_metrics({"valor_mano": total["mano"], "valor_pie": total["pie"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solver", choices=SOLVERS, default="cfr")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    args = parser.parse_args()

    game = pyspiel.load_game("truco_envido")
    solver = SOLVERS[args.solver](game)

    setup_experiment("envido-cfr")
    with mlflow.start_run():
        mlflow.log_params({"game": "truco_envido", **vars(args)})
        start = time.perf_counter()
        for it in range(1, args.iterations + 1):
            solver.evaluate_and_update_policy()
            if it == 1 or it % args.eval_every == 0 or it == args.iterations:
                expl = pyspiel.exploitability(game, solver.average_policy())
                mlflow.log_metric("exploitability", expl, step=it)
                mlflow.log_metric("elapsed_s", time.perf_counter() - start, step=it)
                print(f"iter {it:5d}  exploitability {expl:.5f}")

        rows = policy_rows(game, solver.average_policy())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estrategia.csv"
            write_csv(path, rows)
            mlflow.log_artifact(str(path))
        log_strategy_analysis(game, rows)


if __name__ == "__main__":
    main()
