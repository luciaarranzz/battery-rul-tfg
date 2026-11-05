"""Entrena el XGBoost final con las baterías de entrenamiento y lo guarda para la app.

Se entrena igual que en las gráficas y la evaluación (mismos hiperparámetros
y semilla) y solo con las baterías de entrenamiento, de modo que las de test,
con las que se hacen las demostraciones de la app, siguen sin haberse usado.

Uso:  python -m src.export_model
"""
from pathlib import Path

import joblib
import numpy as np

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.classical_models import XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)

MODEL_PATH = Path("models/xgboost_soh.joblib")


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, _ = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    X_train, y_train, _ = build_feature_matrix(train_df, target=target)

    model = XGBoostModel(**cfg["models"]["xgboost"])
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": model, "columnas": list(X_train.columns)}, MODEL_PATH)
    log.info(
        "Modelo guardado en %s (%d variables, %.0f KB)",
        MODEL_PATH,
        X_train.shape[1],
        MODEL_PATH.stat().st_size / 1024,
    )

    # Comprobación: el modelo cargado del fichero debe predecir lo mismo
    guardado = joblib.load(MODEL_PATH)
    muestra = X_train.head(20)
    assert np.allclose(model.predict(muestra), guardado["modelo"].predict(muestra))
    log.info("Comprobado: el modelo guardado da las mismas predicciones que el original.")


if __name__ == "__main__":
    main()