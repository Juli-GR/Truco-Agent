"""Configuración común de MLflow: todos los experimentos se guardan en la raíz del repo.

Para ver los resultados, desde la raíz del repo:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
ARTIFACTS_DIR = REPO_ROOT / "mlartifacts"


def setup_experiment(name: str) -> None:
    """Apunta MLflow a la base local del repo y activa (o crea) el experimento `name`."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ARTIFACTS_DIR / name).as_uri())
    mlflow.set_experiment(name)
