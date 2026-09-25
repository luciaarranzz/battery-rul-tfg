"""Validación cruzada agrupada por batería, sobre el conjunto de train.

Uso:
    python -m src.cross_validate

Por qué: con un único reparto train/test (9 baterías de test) el resultado
depende mucho de qué baterías tocan por azar. Aquí se hacen varios repartos
distintos dentro de las baterías de TRAIN (GroupKFold, agrupado por
battery_id) y se reporta la media y la desviación de las métricas en cada
fold, para tener una estimación fiable antes de tocar hiperparámetros.
Las baterías de test (configs/config.yaml, data.test_batteries) no se
tocan aquí -- se reservan para la evaluación final en train.py.
"""
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.classical_models import LinearBaselineModel, RandomForestModel, XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_classical_models(cfg: dict) -> dict:
    mcfg = cfg["models"]
    return {
        "linear_baseline": LinearBaselineModel(),
        "random_forest": RandomForestModel(**mcfg["random_forest"]),
        "xgboost": XGBoostModel(**mcfg["xgboost"]),
    }


def cross_validate(train_df: pd.DataFrame, cfg: dict, target: str = "soh", n_splits: int = 5) -> dict:
    """Entrena y evalúa cada modelo en `n_splits` repartos distintos
    (GroupKFold por battery_id) dentro de `train_df`. Devuelve
    {nombre_modelo: [metrics_fold_0, metrics_fold_1, ...]}."""
    X, y, groups = build_feature_matrix(train_df, target=target)
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = {name: [] for name in build_classical_models(cfg)}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, model in build_classical_models(cfg).items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics = model.evaluate(y_val, y_pred)
            fold_metrics[name].append(metrics)
            log.info("fold %d - %s -> %s", fold_idx, name, {k: round(v, 3) for k, v in metrics.items()})

    return fold_metrics


def summarize(fold_metrics: dict) -> pd.DataFrame:
    """Convierte {modelo: [metrics_fold_0, ...]} en una tabla con la media
    y la desviación típica de cada métrica, una fila por modelo."""
    rows = []
    for name, folds in fold_metrics.items():
        df = pd.DataFrame(folds)
        row = {"model": name}
        for col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    log.info("Cargando datos (fuente=%s)...", cfg["data"]["source"])
    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, _ = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    log.info("Baterías en train (para CV): %d", train_df.battery_id.nunique())

    fold_metrics = cross_validate(train_df, cfg, target=target, n_splits=5)
    summary = summarize(fold_metrics)
    log.info("Resumen validación cruzada (media ± std por modelo):\n%s", summary.round(3))
    return summary


if __name__ == "__main__":
    main()