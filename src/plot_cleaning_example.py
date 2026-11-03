"""Ejemplo de limpieza: capacidad de descarga de una batería antes y después del filtrado."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.cross_validate import load_config
from src.data.nasa_loader import load_nasa_dataset
from src.utils import get_logger

log = get_logger(__name__)

RAW_DIR = Path("data/raw/cleaned_dataset")
BATTERY = "B0036"
COLOR_KEPT = "#1f4e79"  # azul oscuro: descargas conservadas
COLOR_REMOVED = "#d9731a"  # naranja: descargas descartadas
COLOR_GRID = "#e5e5e5"


def raw_capacities(raw_dir: Path, battery: str) -> np.ndarray:
    """Capacidades de las descargas de una batería, en orden de medida, sin filtrar."""
    meta = pd.read_csv(raw_dir / "metadata.csv")
    group = meta[(meta["battery_id"] == battery) & (meta["type"] == "discharge")]
    group = group.sort_values("test_id")
    return pd.to_numeric(group["Capacity"], errors="coerce").to_numpy(dtype=float)


def match_kept(raw: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Marca qué descargas en bruto sobreviven a la limpieza (la limpieza solo quita, no reordena)."""
    is_kept = np.zeros(len(raw), dtype=bool)
    j = 0
    for i, value in enumerate(raw):
        if j < len(kept) and np.isclose(value, kept[j]):
            is_kept[i] = True
            j += 1
    if j != len(kept):
        raise ValueError("No se han podido alinear las descargas en bruto con las limpias.")
    return is_kept


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    results_dir = Path(cfg["output"]["results_dir"])

    raw = raw_capacities(RAW_DIR, BATTERY)
    cleaned = load_nasa_dataset(str(RAW_DIR))
    kept = cleaned.loc[cleaned["battery_id"] == BATTERY, "capacity"].to_numpy(dtype=float)
    is_kept = match_kept(raw, kept)

    x = np.arange(len(raw))
    first_kept = int(np.argmax(is_kept))
    valid = ~np.isnan(raw)
    initial = ~is_kept & valid & (x < first_kept)
    other = ~is_kept & valid & (x >= first_kept)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(x[is_kept], raw[is_kept], color=COLOR_KEPT, linewidth=1.2,
            label="Descargas conservadas")
    ax.scatter(x[initial], raw[initial], marker="x", color=COLOR_REMOVED, s=30,
               linewidths=1.4, label="Descartadas: régimen inicial")
    ax.scatter(x[other], raw[other], marker="D", facecolors="none",
               edgecolors=COLOR_REMOVED, s=28, linewidths=1.2,
               label="Descartadas: fuera de rango")

    ax.set_title(BATTERY, fontsize=9)
    ax.set_xlabel("Descarga (orden de medida)", fontsize=8)
    ax.set_ylabel("Capacidad (Ah)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color=COLOR_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ejemplo_limpieza.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(
        "%s: %d descargas en bruto, %d conservadas, %d descartadas "
        "(%d de régimen inicial, %d fuera de rango, %d sin capacidad)",
        BATTERY, len(raw), int(is_kept.sum()), int((~is_kept).sum()),
        int(initial.sum()), int(other.sum()), int((~valid).sum()),
    )
    log.info("Gráfica guardada en %s", out_path)


if __name__ == "__main__":
    main()