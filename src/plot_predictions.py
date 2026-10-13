"""Gráfica de predicción vs. valor real del SOH sobre las baterías de test."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.classical_models import XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, test_df = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])

    X_train, y_train, _ = build_feature_matrix(train_df, target=target)
    X_test, y_test, _ = build_feature_matrix(test_df, target=target)
    assert len(X_test) == len(test_df), "build_feature_matrix ha descartado filas"

    model = XGBoostModel(**cfg["models"]["xgboost"])
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_test))
    y_real = np.asarray(y_test)

    batteries = list(test_df["battery_id"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, battery in zip(axes, batteries):
        mask = (test_df["battery_id"] == battery).to_numpy()
        cycle = test_df.loc[mask, "cycle"].to_numpy()
        real = y_real[mask]
        pred = y_pred[mask]
        rmse = np.sqrt(np.mean((real - pred) ** 2))
        ax.plot(cycle, real, label="real")
        ax.plot(cycle, pred, label="predicho")
        ax.set_title(f"{battery} (RMSE {rmse:.1f})")
        ax.set_xlabel("ciclo")
    axes[0].set_ylabel("SOH (%)")
    axes[0].legend()
    fig.tight_layout()

    out_dir = Path(cfg["output"]["results_dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "predicciones_xgboost.png", dpi=150)
    plt.close(fig)
    log.info("Gráfica guardada en %s", out_dir / "predicciones_xgboost.png")


if __name__ == "__main__":
    main()