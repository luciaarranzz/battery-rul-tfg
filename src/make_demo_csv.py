"""Genera los CSV de demostración de la app a partir de una batería de test.

Cada CSV tiene el formato de ``src/data/bms_csv.py`` (una fila por muestra
de la descarga). Tras escribirlos se pasan por ``procesar_csv``: los CSV
limpios no deberían dar avisos, y el que tiene errores a propósito debería
dar uno por cada problema introducido.

Uso:  python -m src.make_demo_csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bms_csv import procesar_csv
from src.data.loader import DataLoader
from src.train import load_config
from src.utils import get_logger

log = get_logger(__name__)

DEMO_DIR = Path("data/demo")
BATERIA = "B0036"  # debe ser una batería de test: el modelo no la vio al entrenar
# (fichero, primer ciclo, último ciclo); los ciclos se numeran desde 1
LIMPIOS = [
    ("demo_vehiculo_nuevo.csv", 1, 60),
    ("demo_continuacion.csv", 61, 120),
]
CON_ERRORES = ("demo_con_errores.csv", 1, 12)


def ciclos_a_muestras(cycles: pd.DataFrame, bateria: str, primero: int, ultimo: int) -> pd.DataFrame:
    """Pasa los ciclos de una batería (numerados desde 1 por su orden) a una fila por muestra."""
    una = cycles[cycles["battery_id"] == bateria].reset_index(drop=True)
    if una.empty:
        raise ValueError(f"No hay datos de la batería {bateria}")
    partes = []
    for i, fila in una.iloc[primero - 1 : ultimo].iterrows():
        n = len(fila["time_arr"])
        if not len(fila["voltage_arr"]) == len(fila["temp_arr"]) == n:
            raise ValueError(f"Ciclo {i + 1}: las señales no tienen la misma longitud")
        partes.append(
            pd.DataFrame(
                {
                    "ciclo": i + 1,
                    "tiempo_s": fila["time_arr"],
                    "voltaje_v": fila["voltage_arr"],
                    "temperatura_c": fila["temp_arr"],
                    "tiempo_carga_s": fila["charge_time"],
                    "temp_ambiente_c": fila["ambient_temp"],
                }
            )
        )
    if not partes:
        raise ValueError(f"{bateria} no tiene ciclos entre el {primero} y el {ultimo}")
    return pd.concat(partes, ignore_index=True)


def inyectar_errores(df: pd.DataFrame) -> pd.DataFrame:
    """Estropea a propósito un CSV limpio para probar las advertencias."""
    df = df.copy()
    df = df[df["ciclo"] != 5]  # falta el ciclo 5
    df.loc[df["ciclo"] == 3, "voltaje_v"] = 9.9  # voltaje imposible
    df = df.drop(df[df["ciclo"] == 7].index[5:])  # el ciclo 7 se queda con 5 muestras
    df.loc[df[df["ciclo"] == 9].index[:3], "temperatura_c"] = np.nan  # tres lecturas vacías
    return df


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    if BATERIA not in cfg["data"]["test_batteries"]:
        raise ValueError(f"{BATERIA} no es una batería de test")
    cycles = DataLoader(cfg).load_cycles()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    ficheros = {}
    for nombre, primero, ultimo in LIMPIOS:
        ficheros[nombre] = ciclos_a_muestras(cycles, BATERIA, primero, ultimo)
    nombre, primero, ultimo = CON_ERRORES
    ficheros[nombre] = inyectar_errores(ciclos_a_muestras(cycles, BATERIA, primero, ultimo))

    for nombre, df in ficheros.items():
        ruta = DEMO_DIR / nombre
        df.round(4).to_csv(ruta, index=False)
        resultado = procesar_csv(ruta)
        log.info(
            "%s: %d filas, %.0f KB, %d ciclos válidos, %d avisos, %d errores",
            nombre,
            len(df),
            ruta.stat().st_size / 1024,
            len(resultado.ciclos),
            len(resultado.avisos),
            len(resultado.errores),
        )
        for aviso in resultado.avisos:
            log.info("    aviso: %s", aviso)
        for error in resultado.errores:
            log.info("    ERROR: %s", error)


if __name__ == "__main__":
    main()