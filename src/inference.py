"""Predicción del SOH de un vehículo a partir de los ciclos de su historial.

Los indicadores de cada ciclo (los de ``extract_health_indicators``) son lo
que se guardará en la base de datos. Las variables relativas se recalculan
siempre sobre el historial completo del vehículo, porque dependen de sus
primeras ``N_REF_CYCLES`` descargas.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    N_REF_CYCLES,
    add_relative_features,
    extract_health_indicators,
)

MODEL_PATH = Path("models/xgboost_soh.joblib")  # lo guarda src/export_model.py


def cargar_modelo(ruta: Path = MODEL_PATH):
    """Carga el modelo guardado y la lista de columnas que espera, en orden."""
    guardado = joblib.load(ruta)
    return guardado["modelo"], guardado["columnas"]


def indicadores_por_ciclo(ciclos: pd.DataFrame) -> pd.DataFrame:
    """Calcula los indicadores de salud de cada ciclo (una fila por ciclo)."""
    filas = [
        {"ciclo": int(fila["ciclo"]), **extract_health_indicators(fila)}
        for _, fila in ciclos.iterrows()
    ]
    return pd.DataFrame(filas, columns=["ciclo", *FEATURE_COLUMNS])


def predecir_soh(indicadores: pd.DataFrame, modelo, columnas) -> pd.DataFrame:
    """Predice el SOH de cada ciclo del historial de un vehículo.

    ``indicadores`` tiene que ser el historial completo del vehículo, no solo
    los ciclos nuevos. Devuelve ``ciclo`` y ``soh_predicho`` ordenados por ciclo.
    """
    if len(indicadores) < N_REF_CYCLES:
        raise ValueError(
            f"Se necesitan al menos {N_REF_CYCLES} ciclos para fijar la referencia "
            f"del vehículo y hay {len(indicadores)}."
        )
    ordenado = indicadores.sort_values("ciclo").reset_index(drop=True)
    X = add_relative_features(ordenado[FEATURE_COLUMNS], pd.Series(0, index=ordenado.index))
    soh = np.asarray(modelo.predict(X[columnas]))
    return pd.DataFrame({"ciclo": ordenado["ciclo"], "soh_predicho": soh})