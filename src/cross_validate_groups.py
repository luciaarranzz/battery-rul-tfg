"""Validación cruzada dejando fuera un grupo experimental completo cada vez.

Comprueba si los modelos generalizan a condiciones de ensayo que no han visto
en el entrenamiento (temperatura, corriente de descarga, voltaje de corte).
"""
from sklearn.model_selection import LeaveOneGroupOut

from src.cross_validate import build_classical_models, load_config, summarize
from src.data.loader import DataLoader
from src.eda import BATTERY_TO_GROUP
from src.features import build_feature_matrix
from src.utils import get_logger, set_seed

log = get_logger(__name__)


def cross_validate_by_group(df, cfg, target="soh"):
    X, y, _ = build_feature_matrix(df, target=target)
    assert len(X) == len(df), "build_feature_matrix ha descartado filas"
    exp_groups = df["battery_id"].map(BATTERY_TO_GROUP).to_numpy()

    logo = LeaveOneGroupOut()
    fold_metrics = {name: [] for name in build_classical_models(cfg)}
    for train_idx, val_idx in logo.split(X, y, groups=exp_groups):
        left_out = exp_groups[val_idx][0]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        for name, model in build_classical_models(cfg).items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics = model.evaluate(y_val, y_pred)
            fold_metrics[name].append(metrics)
            log.info("grupo %s - %s -> %s", left_out, name,
                     {k: round(v, 3) for k, v in metrics.items()})
    return fold_metrics


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    log.info("Cargando datos (fuente=%s)...", cfg["data"]["source"])
    cycles = DataLoader(cfg).load_cycles()
    fold_metrics = cross_validate_by_group(cycles, cfg, target=target)
    summary = summarize(fold_metrics)
    log.info("Resumen dejando un grupo fuera (media y desviación):\n%s", summary.round(3))


if __name__ == "__main__":
    main()