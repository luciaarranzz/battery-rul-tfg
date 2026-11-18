"""Extracción de indicadores de salud por ciclo de descarga.

Los indicadores se calculan solo a partir de señales que un BMS real
proporcionaría durante la descarga (tiempo, voltaje y temperatura), de la
duración de la carga previa y de la temperatura ambiente. No se utilizan
datos de laboratorio como la espectroscopía de impedancia o la curva
capacidad-voltaje.

Como el SOH se calcula respecto a la capacidad inicial de cada batería, a
las variables absolutas se añaden variables relativas a los primeros ciclos
de esa misma batería. En un vehículo real equivalen a comparar cada lectura
con las primeras lecturas del propio vehículo.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["discharge_duration", "charge_time", "t_mean", "t_max", "voltage_drop_rate", "ambient_temp"]

# Variables relativas: se calculan respecto a la mediana de los primeros
# N_REF_CYCLES ciclos de cada batería, la misma referencia que usa el SOH.
N_REF_CYCLES = 3
RATIO_COLUMNS = ["discharge_duration", "voltage_drop_rate", "charge_time"]
DELTA_COLUMNS = ["t_mean"]


def discharge_duration(time_arr) -> float:
    """Duración total del ciclo de descarga, en segundos."""
    time_arr = np.asarray(time_arr, dtype=float)
    if len(time_arr) < 2:
        return 0.0
    return float(time_arr[-1] - time_arr[0])


def voltage_drop_rate(voltage_arr, time_arr) -> float:
    """Velocidad media de caída del voltaje durante la descarga (V/s): la
    diferencia entre el voltaje inicial y el final dividida por la duración."""
    duration = discharge_duration(time_arr)
    if duration <= 0:
        return 0.0
    voltage_arr = np.asarray(voltage_arr, dtype=float)
    return float((voltage_arr[0] - voltage_arr[-1]) / duration)


def extract_health_indicators(cycle_row: pd.Series) -> dict:
    """Calcula los indicadores de salud de un ciclo de descarga a partir de
    sus señales (time_arr, voltage_arr, temp_arr) y de charge_time y
    ambient_temp, que se obtienen al cargar los datos."""
    temp_arr = np.asarray(cycle_row["temp_arr"], dtype=float)
    return {
        "discharge_duration": discharge_duration(cycle_row["time_arr"]),
        "charge_time": float(cycle_row["charge_time"]),
        "t_mean": float(np.mean(temp_arr)) if len(temp_arr) else 0.0,
        "t_max": float(np.max(temp_arr)) if len(temp_arr) else 0.0,
        "voltage_drop_rate": voltage_drop_rate(cycle_row["voltage_arr"], cycle_row["time_arr"]),
        "ambient_temp": float(cycle_row["ambient_temp"]),
    }


def add_relative_features(X: pd.DataFrame, battery_ids: pd.Series) -> pd.DataFrame:
    """Añade variables relativas a los primeros ciclos de cada batería:
    cocientes (duración de la descarga, caída de voltaje y duración de la
    carga) y una diferencia (temperatura media). Las filas de X deben estar
    ordenadas por ciclo dentro de cada batería, y X y battery_ids deben
    compartir índice."""
    X = X.copy()
    for col in RATIO_COLUMNS:
        ref = X.groupby(battery_ids)[col].transform(lambda s: s.iloc[:N_REF_CYCLES].median())
        X[f"{col}_rel"] = (X[col] / ref.replace(0, np.nan)).fillna(1.0)
    for col in DELTA_COLUMNS:
        ref = X.groupby(battery_ids)[col].transform(lambda s: s.iloc[:N_REF_CYCLES].median())
        X[f"{col}_delta"] = X[col] - ref
    return X


def build_feature_matrix(df: pd.DataFrame, target: str = "soh"):
    """Construye la matriz de variables X, el vector objetivo y (SOH o RUL)
    y los grupos (batería de cada fila) a partir del DataFrame de ciclos."""
    if target not in {"soh", "rul"}:
        raise ValueError("target debe ser 'soh' o 'rul'")

    rows = [extract_health_indicators(cycle_row) for _, cycle_row in df.iterrows()]

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = df[target].reset_index(drop=True)
    y.name = target
    groups = df["battery_id"].reset_index(drop=True)
    groups.name = "battery_id"
    X = add_relative_features(X, groups)
    return X, y, groups