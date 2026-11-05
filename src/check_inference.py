"""Comprueba la predicción de la app de extremo a extremo con una batería de test.

Predice el SOH de ``demo_vehiculo_nuevo.csv`` con el modelo guardado y lo
compara con las predicciones de XGBoost para B0036 de results/predictions.csv,
que se hicieron sin pasar por el CSV. Deben coincidir salvo por el redondeo
a 4 decimales del CSV.

Uso:  python -m src.check_inference
"""
import pandas as pd

from src.data.bms_csv import procesar_csv
from src.inference import cargar_modelo, indicadores_por_ciclo, predecir_soh
from src.utils import get_logger

log = get_logger(__name__)

CSV_DEMO = "data/demo/demo_vehiculo_nuevo.csv"
PREDICCIONES = "results/predictions.csv"
BATERIA = "B0036"


def main():
    resultado = procesar_csv(CSV_DEMO)
    modelo, columnas = cargar_modelo()
    app = predecir_soh(indicadores_por_ciclo(resultado.ciclos), modelo, columnas)

    ref = pd.read_csv(PREDICCIONES)
    ref = ref[(ref["bateria"] == BATERIA) & (ref["modelo"] == "XGBoost")]
    ref = ref.sort_values("ciclo").head(len(app)).reset_index(drop=True)

    comparacion = pd.DataFrame(
        {
            "ciclo_app": app["ciclo"],
            "ciclo_resultados": ref["ciclo"],
            "soh_app": app["soh_predicho"],
            "soh_resultados": ref["soh_predicho"],
        }
    )
    comparacion["diferencia"] = (comparacion["soh_app"] - comparacion["soh_resultados"]).abs()

    log.info("Ciclos comparados: %d", len(comparacion))
    log.info(
        "Diferencia máxima: %.4f puntos de SOH; media: %.4f",
        comparacion["diferencia"].max(),
        comparacion["diferencia"].mean(),
    )
    log.info("Primeros ciclos:\n%s", comparacion.head(5).round(2).to_string(index=False))
    log.info("Últimos ciclos:\n%s", comparacion.tail(3).round(2).to_string(index=False))


if __name__ == "__main__":
    main()