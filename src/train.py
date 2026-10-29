"""Punto de entrada: entrena y compara los modelos definidos en config.yaml.

Uso:
    python -m src.train
    python -m src.train --config configs/config.yaml
"""
import argparse
from pathlib import Path

import yaml

from src.data.loader import DataLoader
from src.evaluate import metrics_table, plot_predictions, save_metrics
from src.features import build_feature_matrix
from src.models.classical_models import LinearBaselineModel, RandomForestModel, XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)

# Carpeta de la que se permite leer la configuración (raíz del proyecto/configs).
CONFIG_DIR = (Path(__file__).resolve().parent.parent / "configs").resolve()


def load_config(path: str) -> dict:
    """Carga un archivo YAML de configuración.

    Solo se aceptan archivos .yaml/.yml situados dentro de la carpeta
    `configs/` del proyecto, para evitar leer rutas arbitrarias pasadas por
    línea de comandos.
    """
    config_path = Path(path).resolve()
    if config_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"El archivo de configuración debe ser .yaml o .yml: {path}")
    if not config_path.is_relative_to(CONFIG_DIR):
        raise ValueError(f"El archivo de configuración debe estar dentro de {CONFIG_DIR}: {path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_classical_models(cfg: dict) -> dict:
    mcfg = cfg["models"]
    return {
        "linear_baseline": LinearBaselineModel(),
        "random_forest": RandomForestModel(**mcfg["random_forest"]),
        "xgboost": XGBoostModel(**mcfg["xgboost"]),
    }


def maybe_build_lstm(cfg: dict):
    """El LSTM es opcional: si PyTorch no está instalado, se omite con un
    aviso en vez de romper el resto del pipeline."""
    try:
        from src.models.deep_models import LSTMModel
        return LSTMModel(**cfg["models"]["lstm"])
    except ImportError as exc:
        log.warning("Omitiendo el modelo LSTM (%s). Instala PyTorch para incluirlo.", exc)
        return None


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    log.info("Cargando datos (fuente=%s)...", cfg["data"]["source"])
    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, test_df = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    log.info("Baterías train=%d, test=%d", train_df.battery_id.nunique(), test_df.battery_id.nunique())

    log.info("Extrayendo features (target=%s)...", target)
    X_train, y_train, groups_train = build_feature_matrix(train_df, target=target)
    X_test, y_test, groups_test = build_feature_matrix(test_df, target=target)

    results = {}

    for name, model in build_classical_models(cfg).items():
        log.info("Entrenando %s...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = model.evaluate(y_test, y_pred)
        results[name] = metrics
        plot_predictions(y_test, y_pred, name, target, cfg["output"]["results_dir"])
        log.info("%s -> %s", name, {k: round(v, 4) for k, v in metrics.items()})

    lstm = maybe_build_lstm(cfg)
    if lstm is not None:
        from src.models.deep_models import align_targets

        log.info("Entrenando lstm...")
        lstm.fit(X_train, y_train, groups=groups_train)
        y_pred_lstm = lstm.predict(X_test, groups=groups_test)
        y_test_lstm = align_targets(y_test, groups_test, lstm.sequence_length)
        metrics = lstm.evaluate(y_test_lstm, y_pred_lstm)
        results["lstm"] = metrics
        plot_predictions(y_test_lstm, y_pred_lstm, "lstm", target, cfg["output"]["results_dir"])
        log.info("lstm -> %s", {k: round(v, 4) for k, v in metrics.items()})

    save_metrics(results, cfg["output"]["results_dir"])
    log.info("\n%s", metrics_table(results))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)