"""Importancia de las variables de entrada para Random Forest y XGBoost.

Calcula dos medidas con la configuración final de configs/config.yaml:
  - importancia interna del modelo (reparto del 100 % entre variables),
  - importancia por permutación sobre las baterías de test: aumento del RMSE
    al barajar cada variable. Cuanto mayor, más depende el modelo de ella.

Los resultados se guardan en results/feature_importance.csv.
"""
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.classical_models import RandomForestModel, XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)

N_REPEATS = 20


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    target = cfg.get("target", "soh")

    log.info("Cargando datos (fuente=%s)...", cfg["data"]["source"])
    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, test_df = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    X_train, y_train, _ = build_feature_matrix(train_df, target=target)
    X_test, y_test, _ = build_feature_matrix(test_df, target=target)
    log.info("Train: %d filas, test: %d filas, %d variables", len(X_train), len(X_test), X_train.shape[1])

    models = {
        "Random Forest": RandomForestModel(**cfg["models"]["random_forest"]),
        "XGBoost": XGBoostModel(**cfg["models"]["xgboost"]),
    }

    tables = []
    for name, model in models.items():
        log.info("Entrenando %s...", name)
        model.fit(X_train, y_train)

        interna = pd.Series(model.feature_importance())
        interna = interna / interna.sum() * 100

        perm = permutation_importance(
            model.model, X_test, y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=N_REPEATS, random_state=seed,
        )
        tabla = pd.DataFrame({
            "variable": X_test.columns,
            "modelo": name,
            "importancia_interna_pct": interna.reindex(X_test.columns).values,
            "permutacion_rmse_media": perm.importances_mean,
            "permutacion_rmse_std": perm.importances_std,
        }).sort_values("permutacion_rmse_media", ascending=False)
        tables.append(tabla)
        log.info("Importancia de variables, %s:\n%s", name, tabla.round(3).to_string(index=False))

    result = pd.concat(tables, ignore_index=True)
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "feature_importance.csv", index=False)
    return result


if __name__ == "__main__":
    main()