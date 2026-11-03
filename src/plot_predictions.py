"""Gráfica de predicción vs. valor real del SOH sobre las baterías de test."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.classical_models import XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)

COLOR_REAL = "#1f4e79"  # azul oscuro: SOH real
COLOR_PRED = "#d9731a"  # naranja: SOH predicho
COLOR_EOL = "#6b6b6b"  # gris: umbral de fin de vida
COLOR_GRID = "#e5e5e5"
N_COLS = 3
# Tamaño final en la memoria (~13 cm de ancho): las fuentes se leen sin reducir.
FIG_WIDTH = 6.6
ROW_HEIGHT = 1.95


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")
    eol_soh = cfg.get("eol_threshold", 0.80) * 100

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
    n_rows = int(np.ceil(len(batteries) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(FIG_WIDTH, ROW_HEIGHT * n_rows),
                             sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, battery) in enumerate(zip(axes, batteries)):
        mask = (test_df["battery_id"] == battery).to_numpy()
        cycle = test_df.loc[mask, "cycle"].to_numpy()
        real = y_real[mask]
        pred = y_pred[mask]
        rmse = np.sqrt(np.mean((real - pred) ** 2))

        ax.axhline(eol_soh, color=COLOR_EOL, linewidth=0.8, linestyle=":",
                   label=f"Fin de vida ({eol_soh:.0f} %)")
        ax.plot(cycle, real, color=COLOR_REAL, linewidth=1.2, label="SOH real")
        ax.plot(cycle, pred, color=COLOR_PRED, linewidth=1.2, linestyle="--",
                label="SOH predicho")

        rmse_txt = f"{rmse:.1f}".replace(".", ",")
        ax.set_title(f"{battery} (RMSE {rmse_txt})", fontsize=8)
        ax.grid(True, color=COLOR_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        if i % N_COLS == 0:
            ax.set_ylabel("SOH (%)", fontsize=8)
        if i >= len(batteries) - N_COLS:
            ax.set_xlabel("Ciclo", fontsize=8)

    for ax in axes[len(batteries):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in ("SOH real", "SOH predicho")] + [
        i for i, name in enumerate(labels) if name.startswith("Fin de vida")
    ]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = Path(cfg["output"]["results_dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predicciones_xgboost.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfica guardada en %s", out_path)


if __name__ == "__main__":
    main()