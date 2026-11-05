"""Pruebas del validador de CSV con lecturas de un BMS."""
import io

import numpy as np
import pandas as pd

from src.data.bms_csv import MIN_MUESTRAS, procesar_csv
from src.features import extract_health_indicators


def _ciclo(numero, muestras=50, voltaje_inicial=4.2, tiempo_carga=3600.0):
    return pd.DataFrame(
        {
            "ciclo": numero,
            "tiempo_s": np.arange(muestras) * 20.0,
            "voltaje_v": np.linspace(voltaje_inicial, 2.7, muestras),
            "temperatura_c": np.linspace(25.0, 35.0, muestras),
            "tiempo_carga_s": tiempo_carga,
            "temp_ambiente_c": 24.0,
        }
    )


def _csv(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def _cinco_ciclos():
    return pd.concat([_ciclo(n) for n in range(1, 6)], ignore_index=True)


def test_csv_valido_no_da_avisos_ni_errores():
    resultado = procesar_csv(_csv(_cinco_ciclos()))
    assert resultado.errores == []
    assert resultado.avisos == []
    assert list(resultado.ciclos["ciclo"]) == [1, 2, 3, 4, 5]


def test_los_ciclos_permiten_calcular_los_indicadores():
    resultado = procesar_csv(_csv(_cinco_ciclos()))
    indicadores = extract_health_indicators(resultado.ciclos.iloc[0])
    assert indicadores["discharge_duration"] == 49 * 20.0
    assert indicadores["charge_time"] == 3600.0
    assert indicadores["ambient_temp"] == 24.0


def test_filas_con_datos_que_faltan_se_descartan_con_aviso():
    df = _cinco_ciclos()
    df.loc[3, "voltaje_v"] = np.nan
    resultado = procesar_csv(_csv(df))
    assert resultado.errores == []
    assert len(resultado.ciclos) == 5
    assert len(resultado.avisos) == 1
    assert "1 de 250 filas" in resultado.avisos[0]


def test_ciclo_que_falta_en_la_numeracion_da_aviso():
    df = pd.concat([_ciclo(1), _ciclo(2), _ciclo(5)], ignore_index=True)
    resultado = procesar_csv(_csv(df))
    assert any("3 a 4" in aviso for aviso in resultado.avisos)


def test_un_solo_ciclo_que_falta_se_avisa_en_singular():
    df = pd.concat([_ciclo(1), _ciclo(3)], ignore_index=True)
    resultado = procesar_csv(_csv(df))
    assert resultado.avisos == ["Falta el ciclo 2 en el CSV."]


def test_voltaje_fuera_de_rango_descarta_el_ciclo():
    df = pd.concat([_ciclo(1), _ciclo(2, voltaje_inicial=9.0)], ignore_index=True)
    resultado = procesar_csv(_csv(df))
    assert list(resultado.ciclos["ciclo"]) == [1]
    assert any("Ciclo 2 descartado" in aviso and "voltaje" in aviso for aviso in resultado.avisos)


def test_ciclo_con_pocas_muestras_se_descarta():
    df = pd.concat([_ciclo(1), _ciclo(2, muestras=MIN_MUESTRAS - 1)], ignore_index=True)
    resultado = procesar_csv(_csv(df))
    assert list(resultado.ciclos["ciclo"]) == [1]
    assert any("muestras" in aviso for aviso in resultado.avisos)


def test_ciclo_con_tiempo_no_creciente_se_descarta():
    malo = _ciclo(2)
    malo["tiempo_s"] = 5.0
    resultado = procesar_csv(_csv(pd.concat([_ciclo(1), malo], ignore_index=True)))
    assert list(resultado.ciclos["ciclo"]) == [1]
    assert any("tiempo" in aviso for aviso in resultado.avisos)


def test_ciclo_con_tiempo_de_carga_no_positivo_se_descarta():
    df = pd.concat([_ciclo(1), _ciclo(2, tiempo_carga=-5.0)], ignore_index=True)
    resultado = procesar_csv(_csv(df))
    assert list(resultado.ciclos["ciclo"]) == [1]


def test_faltan_columnas_es_un_error():
    df = _cinco_ciclos().drop(columns=["voltaje_v", "tiempo_carga_s"])
    resultado = procesar_csv(_csv(df))
    assert resultado.ciclos.empty
    assert resultado.errores == ["Faltan columnas: voltaje_v, tiempo_carga_s"]


def test_ciclo_no_entero_es_un_error():
    df = _cinco_ciclos()
    df["ciclo"] = df["ciclo"] + 0.5
    resultado = procesar_csv(_csv(df))
    assert resultado.ciclos.empty
    assert len(resultado.errores) == 1


def test_sin_ningun_ciclo_valido_es_un_error():
    resultado = procesar_csv(_csv(_ciclo(1, muestras=3)))
    assert resultado.ciclos.empty
    assert resultado.errores == ["No hay ningún ciclo válido en el CSV."]


def test_csv_vacio_es_un_error():
    resultado = procesar_csv(io.StringIO(""))
    assert resultado.ciclos.empty
    assert len(resultado.errores) == 1