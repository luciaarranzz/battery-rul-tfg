"""Experimento sin las variables que dependen de la duración de la descarga.

Uso:
    python -m src.ablation_duration

La duración de la descarga es casi proporcional a la capacidad, y la caída de
voltaje por segundo se calcula dividiendo por esa duración, así que ambas
variables pueden "leer" el SOH. Este script repite la validación cruzada
agrupada por batería (sobre train) y la evaluación en test quitando grupos de
variables, con los hiperparámetros de config.yaml sin volver a ajustarlos, para
ver cuánto empeoran los modelos. La LSTM no se incluye.

Resultado: una tabla en results/ablation_duration.csv con una fila por variante
y modelo.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from src.cross_validate import build_classical_models, load_config, summarize
from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.utils import get_logger, set_seed

log = get_logger(__name__)

DURACION = ["discharge_duration", "discharge_duration_rel"]
CAIDA_VOLTAJE = ["voltage_drop_rate", "voltage_drop_rate_rel"]
CARGA = ["charge_time", "charge_time_rel"]

# Variante -> variables que se quitan.
VARIANTES = {
    "todas": [],
    "sin_duracion": DURACION,
    "sin_duracion_ni_caida": DURACION + CAIDA_VOLTAJE,
    "solo_temperatura": DURACION + CAIDA_VOLTAJE + CARGA,
}


def cv_variante(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cfg: dict, n_splits: int = 5) -> pd.DataFrame:
    """Validación cruzada agrupada por batería de los modelos clásicos.
    Devuelve la media y la desviación típica de cada métrica por modelo."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = {name: [] for name in build_classical_models(cfg)}
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        for name, model in build_classical_models(cfg).items():
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X.iloc[val_idx])
            fold_metrics[name].append(model.evaluate(y.iloc[val_idx], y_pred))
    return summarize(fold_metrics)


def test_variante(X_train, y_train, X_test, y_test, cfg: dict) -> pd.DataFrame:
    """Entrena con todo train y evalúa en test. Una fila por modelo."""
    filas = {}
    for name, model in build_classical_models(cfg).items():
        model.fit(X_train, y_train)
        filas[name] = model.evaluate(y_test, model.predict(X_test))
    return pd.DataFrame(filas).T


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, test_df = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])

    X_train, y_train, groups_train = build_feature_matrix(train_df, target=target)
    X_test, y_test, _ = build_feature_matrix(test_df, target=target)

    tablas = []
    for nombre, quitar in VARIANTES.items():
        cols = [c for c in X_train.columns if c not in quitar]
        log.info("Variante %s: %d variables (%s)", nombre, len(cols), ", ".join(cols))
        cv = cv_variante(X_train[cols], y_train, groups_train, cfg)
        test = test_variante(X_train[cols], y_train, X_test[cols], y_test, cfg)
        tabla = cv.join(test.add_prefix("test_"))
        tabla.insert(0, "variante", nombre)
        tablas.append(tabla.reset_index())

    resultado = pd.concat(tablas, ignore_index=True)
    salida = Path(cfg["output"]["results_dir"]) / "ablation_duration.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida, index=False)
    log.info("Resultados guardados en %s", salida)
    log.info("\n%s", resultado.round(3).to_string())
    return resultado


if __name__ == "__main__":
    main()