"""Lectura y validación de los CSV con lecturas simuladas de un BMS.

Formato del CSV: una fila por muestra de una descarga, con las columnas

    ciclo, tiempo_s, voltaje_v, temperatura_c, tiempo_carga_s, temp_ambiente_c

``tiempo_carga_s`` y ``temp_ambiente_c`` son datos del ciclo completo: se
repiten en todas sus filas y se usa el valor de la primera.

El resultado tiene el mismo formato interno que producen los cargadores
(una fila por ciclo con ``time_arr``, ``voltage_arr``, ``temp_arr``,
``charge_time`` y ``ambient_temp``), de modo que se le puede aplicar
``extract_health_indicators`` de ``src.features`` igual que en el
entrenamiento.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUMNAS = [
    "ciclo",
    "tiempo_s",
    "voltaje_v",
    "temperatura_c",
    "tiempo_carga_s",
    "temp_ambiente_c",
]
COLUMNAS_CICLOS = ["ciclo", "time_arr", "voltage_arr", "temp_arr", "charge_time", "ambient_temp"]

MIN_MUESTRAS = 10
RANGO_VOLTAJE = (1.0, 4.5)  # V, para una celda de ion-litio
RANGO_TEMPERATURA = (-20.0, 80.0)  # °C


@dataclass
class ResultadoCSV:
    """Ciclos válidos del CSV, más los errores (impiden usarlo) y los avisos."""

    ciclos: pd.DataFrame
    errores: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


def _ciclos_vacios() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNAS_CICLOS)


def _motivo_descarte(g: pd.DataFrame) -> str | None:
    """Devuelve por qué hay que descartar un ciclo, o None si es válido."""
    if len(g) < MIN_MUESTRAS:
        return f"solo tiene {len(g)} muestras (mínimo {MIN_MUESTRAS})"
    if (np.diff(g["tiempo_s"].to_numpy()) <= 0).any():
        return "el tiempo no es siempre creciente"
    v_min, v_max = g["voltaje_v"].min(), g["voltaje_v"].max()
    if v_min < RANGO_VOLTAJE[0] or v_max > RANGO_VOLTAJE[1]:
        return f"voltaje fuera de rango ({v_min:.1f} a {v_max:.1f} V)"
    t_min, t_max = g["temperatura_c"].min(), g["temperatura_c"].max()
    if t_min < RANGO_TEMPERATURA[0] or t_max > RANGO_TEMPERATURA[1]:
        return f"temperatura fuera de rango ({t_min:.1f} a {t_max:.1f} °C)"
    if g["tiempo_carga_s"].iloc[0] <= 0:
        return "el tiempo de carga no es positivo"
    return None


def procesar_csv(origen) -> ResultadoCSV:
    """Lee un CSV (ruta o fichero subido) y devuelve sus ciclos válidos.

    Los problemas graves van a ``errores`` y los demás a ``avisos``: las
    filas con datos que faltan y los ciclos incorrectos se descartan y se
    avisa, y también se avisa si faltan ciclos en la numeración.
    """
    try:
        df = pd.read_csv(origen)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        return ResultadoCSV(_ciclos_vacios(), errores=[f"No se pudo leer el CSV: {exc}"])

    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        return ResultadoCSV(_ciclos_vacios(), errores=["Faltan columnas: " + ", ".join(faltan)])

    errores: list[str] = []
    avisos: list[str] = []

    df = df[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    n_filas = len(df)
    df = df.dropna()
    if len(df) < n_filas:
        avisos.append(
            f"{n_filas - len(df)} de {n_filas} filas tienen datos que faltan "
            "o no son números; se han descartado."
        )
    if df.empty:
        return ResultadoCSV(_ciclos_vacios(), errores=["El CSV no tiene filas con datos válidos."])
    if not (df["ciclo"] % 1 == 0).all():
        return ResultadoCSV(_ciclos_vacios(), errores=["La columna 'ciclo' debe contener enteros."])

    ids = sorted(int(c) for c in df["ciclo"].unique())
    for anterior, siguiente in zip(ids[:-1], ids[1:]):
        if siguiente - anterior == 2:
            avisos.append(f"Falta el ciclo {anterior + 1} en el CSV.")
        elif siguiente - anterior > 2:
            avisos.append(f"Faltan los ciclos {anterior + 1} a {siguiente - 1} en el CSV.")

    filas = []
    for ciclo, g in df.groupby("ciclo", sort=True):
        motivo = _motivo_descarte(g)
        if motivo:
            avisos.append(f"Ciclo {int(ciclo)} descartado: {motivo}.")
            continue
        filas.append(
            {
                "ciclo": int(ciclo),
                "time_arr": g["tiempo_s"].to_numpy(),
                "voltage_arr": g["voltaje_v"].to_numpy(),
                "temp_arr": g["temperatura_c"].to_numpy(),
                "charge_time": float(g["tiempo_carga_s"].iloc[0]),
                "ambient_temp": float(g["temp_ambiente_c"].iloc[0]),
            }
        )

    if not filas:
        errores.append("No hay ningún ciclo válido en el CSV.")
    return ResultadoCSV(pd.DataFrame(filas, columns=COLUMNAS_CICLOS), errores, avisos)