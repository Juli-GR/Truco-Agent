"""Etapa 2a: CFR exacto sobre una mano de truco sin envido con mazo reducido.

El juego se exporta una sola vez a formato EFG (en results/efg/, ignorado por git) y CFR corre sobre
la versión en C++ (efg_game), decenas de veces más rápido que sobre el juego en Python. El nombre
del archivo incluye un hash del código de las reglas y de la exportación, así que si cambian se
vuelve a exportar.

Registra en MLflow (experimento "truco-cfr") la curva de exploitabilidad y la estrategia promedio
final como CSV comprimido, con una fila por information set. Medir la exploitabilidad recorre el
árbol completo y cuesta tanto como decenas de iteraciones, así que se mide en puntos espaciados en
escala logarítmica (1, 2, 5, 10, 20, 50, ...).

Uso, desde la raíz del repo:
    python experiments/truco_cfr.py --niveles 6 --iterations 500
"""

import argparse
import csv
import gzip
import hashlib
import tempfile
import time
from pathlib import Path

import mlflow
import pyspiel

import truco.games.truco  # noqa: F401  (registra el juego)
from truco.games.cards import TRUCO_LEVEL_NAMES
from truco.games.efg import EFGMapping, export_efg, load_efg, original_strategy
from truco.games.truco import Action, parse_information_state
from truco.tracking import REPO_ROOT, setup_experiment

SOLVERS = {"cfr": pyspiel.CFRSolver, "cfr_plus": pyspiel.CFRPlusSolver}
# Código del que depende el archivo EFG: las reglas y la exportación.
RULES_FILES = [REPO_ROOT / "truco" / "games" / name for name in ("truco.py", "cards.py", "efg.py")]
ACTION_NAMES = [*TRUCO_LEVEL_NAMES, *(a.name for a in Action)]


def checkpoints(iterations: int) -> set[int]:
    """Iteraciones 1, 2, 5, 10, 20, 50, ... hasta `iterations`, más la última."""
    points = {iterations}
    scale = 1
    while scale <= iterations:
        points.update(m * scale for m in (1, 2, 5) if m * scale <= iterations)
        scale *= 10
    return points


def rules_hash() -> str:
    digest = hashlib.sha256()
    for path in RULES_FILES:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def load_or_export(niveles: int, efg_dir: Path) -> tuple[pyspiel.Game, EFGMapping]:
    """Carga el EFG del juego con `niveles`, exportándolo antes si no existe."""
    path = efg_dir / f"truco_niveles{niveles}_{rules_hash()}.efg"
    mapping_path = path.with_suffix(".json")
    # La correspondencia se guarda al final: si existe, la exportación terminó.
    if not mapping_path.exists():
        efg_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exportando {path.name}...")
        start = time.perf_counter()
        mapping = export_efg(pyspiel.load_game("truco", {"niveles": niveles}), path)
        mapping.save(mapping_path)
        mlflow.log_metric("export_s", time.perf_counter() - start)
    start = time.perf_counter()
    efg = load_efg(path)
    mlflow.log_metric("carga_s", time.perf_counter() - start)
    return efg, EFGMapping.load(mapping_path)


def write_strategy(path: Path, policy: pyspiel.Policy, mapping: EFGMapping) -> None:
    """Estrategia por information set: quién juega, si es mano, sus cartas, el historial y la
    probabilidad de cada acción (vacía si no es legal)."""
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["jugador", "es_mano", "cartas", "historia", *ACTION_NAMES])
        for _, info_state, probs in original_strategy(policy, mapping):
            row = parse_information_state(info_state)
            action_probs = [
                round(probs[a], 4) if a in probs else "" for a in range(len(ACTION_NAMES))
            ]
            writer.writerow([*row.values(), *action_probs])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--niveles", type=int, default=6, help="niveles más altos del mazo")
    parser.add_argument("--solver", choices=SOLVERS, default="cfr")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--efg-dir", type=Path, default=REPO_ROOT / "results" / "efg")
    args = parser.parse_args()

    setup_experiment("truco-cfr")
    with mlflow.start_run():
        mlflow.log_params({"game": "truco", **vars(args)})
        efg, mapping = load_or_export(args.niveles, args.efg_dir)
        mlflow.log_params({f"infosets_jugador{p}": len(s) for p, s in enumerate(mapping.infosets)})

        solver = SOLVERS[args.solver](efg)
        evals = checkpoints(args.iterations)
        start = time.perf_counter()
        for it in range(1, args.iterations + 1):
            solver.evaluate_and_update_policy()
            if it in evals:
                expl = pyspiel.exploitability(efg, solver.average_policy())
                mlflow.log_metric("exploitability", expl, step=it)
                mlflow.log_metric("elapsed_s", time.perf_counter() - start, step=it)
                print(f"iter {it:5d}  exploitability {expl:.5f}", flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estrategia.csv.gz"
            write_strategy(path, solver.average_policy(), mapping)
            mlflow.log_artifact(str(path))


if __name__ == "__main__":
    main()
