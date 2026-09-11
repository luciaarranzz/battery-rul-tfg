"""Evaluación y visualización de resultados: tabla de métricas, gráfica de
predicción vs. real, e interpretabilidad (SHAP, opcional)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no requiere pantalla; guarda directamente a fichero
import matplotlib.pyplot as plt
import pandas as pd


def metrics_table(results: dict) -> pd.DataFrame:
    """`results` es {nombre_modelo: {"RMSE": ..., "MAE": ..., ...}}."""
    return pd.DataFrame(results).T.sort_values("RMSE")


def save_metrics(results: dict, output_dir: str = "results", filename: str = "metrics.json") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return out_path


def plot_predictions(y_true, y_pred, model_name: str, target: str, output_dir: str = "results") -> Path:
    out_dir = Path(output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlabel(f"{target} real")
    ax.set_ylabel(f"{target} predicho")
    ax.set_title(f"{model_name}: predicción vs. real")
    fig.tight_layout()

    out_path = out_dir / f"{model_name}_{target}_pred_vs_real.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def shap_analysis(model, X, output_dir: str = "results"):
    """Interpretabilidad con SHAP sobre un modelo de tipo árbol (Random
    Forest / XGBoost). Dependencia opcional: `pip install shap`."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap_analysis necesita el paquete `shap` (pip install shap)."
        ) from exc

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)

    out_dir = Path(output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    out_path = out_dir / f"{model.name}_shap_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path