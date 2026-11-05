"""Pruebas de la predicción del SOH a partir de los ciclos de un vehículo."""
import numpy as np
import pandas as pd
import pytest

from src.features import N_REF_CYCLES
from src.inference import indicadores_por_ciclo, predecir_soh


class ModeloFalso:
    """Guarda lo que recibe para poder revisarlo; predice siempre cero."""

    def predict(self, X):
        self.recibido = X
        return np.zeros(len(X))


COLUMNAS = ["discharge_duration_rel", "t_mean_delta", "discharge_duration", "charge_time"]


def _ciclos(n_ciclos):
    """Ciclos con descargas cada vez más cortas (un 2 % menos en cada ciclo)."""
    filas = []
    for ciclo in range(1, n_ciclos + 1):
        muestras = int(100 * 0.98**ciclo) + 10
        filas.append(
            {
                "ciclo": ciclo,
                "time_arr": np.arange(muestras) * 20.0,
                "voltage_arr": np.linspace(4.2, 2.7, muestras),
                "temp_arr": np.linspace(25.0, 35.0, muestras),
                "charge_time": 3600.0,
                "ambient_temp": 24.0,
            }
        )
    return pd.DataFrame(filas)


def test_indicadores_por_ciclo_tiene_una_fila_por_ciclo():
    indicadores = indicadores_por_ciclo(_ciclos(5))
    assert list(indicadores["ciclo"]) == [1, 2, 3, 4, 5]
    assert indicadores["discharge_duration"].is_monotonic_decreasing


def test_predecir_soh_devuelve_un_valor_por_ciclo_ordenado():
    indicadores = indicadores_por_ciclo(_ciclos(8)).sample(frac=1, random_state=1)
    resultado = predecir_soh(indicadores, ModeloFalso(), COLUMNAS)
    assert list(resultado["ciclo"]) == list(range(1, 9))
    assert len(resultado) == 8


def test_las_variables_relativas_usan_los_primeros_ciclos_como_referencia():
    modelo = ModeloFalso()
    indicadores = indicadores_por_ciclo(_ciclos(8))
    predecir_soh(indicadores, modelo, COLUMNAS)
    relativa = modelo.recibido["discharge_duration_rel"]
    referencia = indicadores["discharge_duration"].iloc[:N_REF_CYCLES].median()
    assert relativa.iloc[1] == pytest.approx(1.0)  # el ciclo 2 es la mediana de los 3 primeros
    assert relativa.iloc[-1] == pytest.approx(indicadores["discharge_duration"].iloc[-1] / referencia)


def test_se_respeta_el_orden_de_columnas_del_modelo():
    modelo = ModeloFalso()
    predecir_soh(indicadores_por_ciclo(_ciclos(5)), modelo, COLUMNAS)
    assert list(modelo.recibido.columns) == COLUMNAS


def test_con_menos_ciclos_que_la_referencia_da_error():
    indicadores = indicadores_por_ciclo(_ciclos(N_REF_CYCLES - 1))
    with pytest.raises(ValueError, match="al menos"):
        predecir_soh(indicadores, ModeloFalso(), COLUMNAS)