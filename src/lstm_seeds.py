"""Repite el entrenamiento de la LSTM con varias semillas y resume su variabilidad en test."""
from pathlib import Path

import pandas as pd

from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.deep_models import LSTMModel, align_targets
from src.train import load_config
from src.utils import get_logger, set_seed

log = get_logger(__name__)

# La semilla 42 es la de los resultados principales: sirve de comprobación.
SEEDS = (42, 43, 44, 45, 46)


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, test_df = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    X_train, y_train, groups_train = build_feature_matrix(train_df, target=target)
    X_test, y_test, groups_test = build_feature_matrix(test_df, target=target)

    rows = []
    for seed in SEEDS:
        log.info("Entrenando la LSTM con la semilla %d...", seed)
        set_seed(seed)
        model = LSTMModel(**cfg["models"]["lstm"])
        model.fit(X_train, y_train, groups=groups_train)
        y_pred = model.predict(X_test, groups=groups_test)
        y_true = align_targets(y_test, groups_test, model.sequence_length)
        metrics = model.evaluate(y_true, y_pred)
        rows.append({"semilla": seed, **metrics})
        log.info("semilla %d -> %s", seed, {k: round(v, 4) for k, v in metrics.items()})

    results = pd.DataFrame(rows)
    out_path = Path(cfg["output"]["results_dir"]) / "lstm_seeds.csv"
    results.to_csv(out_path, index=False)
    log.info("Resultados guardados en %s", out_path)

    summary = results.drop(columns="semilla").agg(["mean", "std", "min", "max"]).round(4)
    print()
    print(results.round(4).to_string(index=False))
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()