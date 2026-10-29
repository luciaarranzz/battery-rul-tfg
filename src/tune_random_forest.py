"""Búsqueda de hiperparámetros para Random Forest, usando la misma validación
cruzada agrupada por batería que src/cross_validate.py y src/tune_xgboost.py.

Se varía la profundidad máxima de los árboles, que es lo que más controla el
sobreajuste con pocas baterías. Los resultados se guardan en
results/random_forest_tuning.csv.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.classical_models import RandomForestModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)


CANDIDATE_PARAMS = [
    {"n_estimators": 300, "max_depth": 3},
    {"n_estimators": 300, "max_depth": 4},
    {"n_estimators": 300, "max_depth": 6},
    {"n_estimators": 300, "max_depth": 8},
    {"n_estimators": 300, "max_depth": 12},    # actual (config.yaml)
    {"n_estimators": 300, "max_depth": None},  # sin límite de profundidad
]


def evaluate_params(train_df: pd.DataFrame, params: dict, target: str = "soh", n_splits: int = 5) -> dict:
    X, y, groups = build_feature_matrix(train_df, target=target)
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = RandomForestModel(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_metrics.append(model.evaluate(y_val, y_pred))

    df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in df.columns:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()
    return summary


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    log.info("Cargando datos (fuente=%s)...", cfg["data"]["source"])
    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, _ = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    log.info("Baterías en train (para CV): %d", train_df.battery_id.nunique())

    results = []
    for params in CANDIDATE_PARAMS:
        log.info("Probando hiperparámetros: %s", params)
        metrics = evaluate_params(train_df, params, target=target)
        results.append({**params, **metrics})

    results_df = pd.DataFrame(results).sort_values("R2_mean", ascending=False)
    log.info("Resultados ordenados por R2_mean (mejor primero):\n%s", results_df.round(3))

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "random_forest_tuning.csv", index=False)
    return results_df


if __name__ == "__main__":
    main()