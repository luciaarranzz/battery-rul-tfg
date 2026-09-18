"""Análisis exploratorio (EDA) sobre el dataset real de baterías.

Uso:
    python -m src.eda

No entrena ningún modelo: solo carga los datos con el mismo DataLoader que
usa train.py y genera gráficas + una tabla resumen, para entender el
dataset antes de seguir afinando features y modelos. Pensado sobre todo
para explicar por qué el primer entrenamiento (18 sept) dio un R² negativo.

Genera:
    - Resumen por consola: nº de baterías, ciclos por batería, cuántas
      filas de descarga se descartaron por datos inválidos.
    - results/eda_summary.csv -- una fila por batería (grupo experimental,
      nº de ciclos, SOH inicial/final, capacidad máxima), lista para tablas
      de la memoria.
    - results/figures/eda/capacity_fade.png -- capacidad (Ah) frente al
      ciclo, una línea por batería, una subgráfica por grupo experimental
      del dataset NASA.
    - results/figures/eda/soh_por_bateria.png -- SOH de cada batería
      (train en azul, test en naranja): si los rangos de SOH de train y
      test no se solapan bien, el modelo va a extrapolar mal -- es la
      explicación más probable del R² negativo del primer entrenamiento.
    - results/figures/eda/correlacion_features.png -- correlación de cada
      feature (las mismas que usa el modelo) con el SOH.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no requiere pantalla; guarda directamente a fichero
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data.loader import DataLoader
from src.features import FEATURE_COLUMNS, build_feature_matrix
from src.utils import get_logger, set_seed

log = get_logger(__name__)

# Grupos experimentales originales del dataset NASA Battery Dataset (según
# los ficheros extra_infos/README_*.txt que trae la descarga de Kaggle).
# Cada grupo se probó con su propia temperatura ambiente y condiciones de
# carga/descarga, así que agrupar por aquí ayuda a ver si el dataset
# mezcla comportamientos muy distintos entre sí.
BATTERY_GROUPS = {
    "05_06_07_18": ["B0005", "B0006", "B0007", "B0018"],
    "25_26_27_28": ["B0025", "B0026", "B0027", "B0028"],
    "29_30_31_32": ["B0029", "B0030", "B0031", "B0032"],
    "33_34_36": ["B0033", "B0034", "B0036"],
    "38_39_40": ["B0038", "B0039", "B0040"],
    "41_42_43_44": ["B0041", "B0042", "B0043", "B0044"],
    "45_46_47_48": ["B0045", "B0046", "B0047", "B0048"],
    "49_50_51_52": ["B0049", "B0050", "B0051", "B0052"],
    "53_54_55_56": ["B0053", "B0054", "B0055", "B0056"],
}
BATTERY_TO_GROUP = {b: g for g, batteries in BATTERY_GROUPS.items() for b in batteries}


def battery_group(battery_id: str) -> str:
    """Grupo experimental de una batería, o "otros" si no está en el mapa
    de arriba (p. ej. datos sintéticos, o una versión futura del dataset
    con más baterías)."""
    return BATTERY_TO_GROUP.get(battery_id, "otros")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_summary(cycles: pd.DataFrame) -> None:
    n_batteries = cycles.battery_id.nunique()
    per_battery = cycles.groupby("battery_id").size()
    log.info("Baterías: %d", n_batteries)
    log.info(
        "Ciclos por batería: min=%d, mediana=%d, max=%d (total=%d filas)",
        per_battery.min(), int(per_battery.median()), per_battery.max(), len(cycles),
    )
    log.info(
        "SOH: min=%.1f%%, mediana=%.1f%%, max=%.1f%%",
        cycles.soh.min(), cycles.soh.median(), cycles.soh.max(),
    )


def save_summary_csv(cycles: pd.DataFrame, output_dir: str = "results") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for battery_id, group in cycles.groupby("battery_id"):
        group = group.sort_values("cycle")
        rows.append({
            "battery_id": battery_id,
            "grupo_experimental": battery_group(battery_id),
            "n_ciclos": len(group),
            "soh_inicial": round(float(group["soh"].iloc[0]), 1),
            "soh_final": round(float(group["soh"].iloc[-1]), 1),
            "capacidad_max_ah": round(float(group["capacity"].max()), 3),
        })
    summary = pd.DataFrame(rows).sort_values(["grupo_experimental", "battery_id"])
    out_path = out_dir / "eda_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def plot_capacity_fade(cycles: pd.DataFrame, out_dir: Path) -> Path:
    groups = sorted(cycles.battery_id.map(battery_group).unique())
    n = len(groups)
    ncols = 3
    nrows = -(-n // ncols)  # división entera hacia arriba
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, group_name in zip(axes, groups):
        battery_ids = [b for b in cycles.battery_id.unique() if battery_group(b) == group_name]
        for battery_id in battery_ids:
            sub = cycles[cycles.battery_id == battery_id].sort_values("cycle")
            ax.plot(sub["cycle"], sub["capacity"], linewidth=1, alpha=0.8, label=battery_id)
        ax.set_title(f"Grupo {group_name}", fontsize=10)
        ax.set_xlabel("ciclo")
        ax.legend(fontsize=6, loc="upper right")
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("capacidad (Ah)")

    fig.suptitle("Caída de capacidad por grupo experimental (NASA Battery Dataset)")
    fig.tight_layout()
    out_path = out_dir / "capacity_fade.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_soh_por_bateria(cycles: pd.DataFrame, test_batteries: list, out_dir: Path) -> Path:
    order = sorted(cycles.battery_id.unique(), key=lambda b: (battery_group(b), b))
    data = [cycles.loc[cycles.battery_id == b, "soh"].to_numpy() for b in order]
    colors = ["#B5652B" if b in test_batteries else "#2B5C86" for b in order]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(order)), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel("SOH (%)")
    ax.set_title("SOH por batería -- azul = train, naranja = test")
    fig.tight_layout()

    out_path = out_dir / "soh_por_bateria.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_correlations(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> Path:
    corr = X.corrwith(y).sort_values()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(corr.index, corr.to_numpy(), color="#2B5C86")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"correlación con {y.name}")
    ax.set_title("Correlación de cada feature con el SOH")
    fig.tight_layout()

    out_path = out_dir / "correlacion_features.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    log.info("Cargando datos (fuente=%s)...", cfg["data"]["source"])
    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    print_summary(cycles)

    summary_path = save_summary_csv(cycles, cfg["output"]["results_dir"])
    log.info("Tabla resumen por batería guardada en %s", summary_path)

    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cap_path = plot_capacity_fade(cycles, fig_dir)
    log.info("Gráfica de caída de capacidad guardada en %s", cap_path)

    soh_path = plot_soh_por_bateria(cycles, cfg["data"]["test_batteries"], fig_dir)
    log.info("Gráfica de SOH por batería guardada en %s", soh_path)

    X, y, _ = build_feature_matrix(cycles, target=target)
    corr_path = plot_feature_correlations(X, y, fig_dir)
    log.info("Gráfica de correlación de features guardada en %s", corr_path)

    log.info("EDA completo. Revisa results/figures/eda/ y results/eda_summary.csv")


if __name__ == "__main__":
    main()
