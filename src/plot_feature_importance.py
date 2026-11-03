"""Gráfica de importancia de las variables (por permutación) para Random Forest y XGBoost."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cross_validate import load_config
from src.utils import get_logger

log = get_logger(__name__)

COLOR_BAR = "#2f7f79"  # verde azulado: importancia por permutación
COLOR_ERR = "#444444"  # gris oscuro: barras de error y línea del cero
COLOR_GRID = "#e5e5e5"
MODELS = ("Random Forest", "XGBoost")


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    results_dir = Path(cfg["output"]["results_dir"])
    df = pd.read_csv(results_dir / "feature_importance.csv")

    # Orden común a los dos paneles: la variable más importante, arriba.
    order = (
        df.groupby("variable")["permutacion_rmse_media"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    x_max = (df["permutacion_rmse_media"] + df["permutacion_rmse_std"]).max() * 1.2

    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.6, 3.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df[df["modelo"] == model].set_index("variable").loc[order]
        values = sub["permutacion_rmse_media"].to_numpy()
        errors = sub["permutacion_rmse_std"].to_numpy()
        y = np.arange(len(order))

        ax.barh(y, values, xerr=errors, color=COLOR_BAR, height=0.6,
                error_kw={"ecolor": COLOR_ERR, "elinewidth": 0.8, "capsize": 2})
        ax.axvline(0, color=COLOR_ERR, linewidth=0.7)
        for yi, value, err in zip(y, values, errors):
            ax.text(max(value, 0) + err + 0.1, yi, f"{value:.2f}".replace(".", ","),
                    va="center", fontsize=6.5)

        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(-0.6, x_max)
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Aumento del RMSE (puntos de SOH)", fontsize=7.5)
        ax.grid(True, axis="x", color=COLOR_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    fig.tight_layout()

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "importancia_variables.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfica guardada en %s", out_path)


if __name__ == "__main__":
    main()