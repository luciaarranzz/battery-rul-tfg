"""Extracción de indicadores de salud (Health Indicators, HI) por ciclo.

Estas funciones operan sobre el formato interno de datos (ver README), así
que funcionan igual con los datos sintéticos que con un dataset real, en
cuanto este último tenga un loader que produzca ese mismo formato.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["delta_q_var", "charge_time", "t_mean", "t_max", "discharge_slope"]


def delta_q_variance(q_curve_ref: np.ndarray, q_curve_cur: np.ndarray) -> float:
    """Varianza de ΔQ(V) entre un ciclo de referencia (p. ej. el ciclo 10)
    y el ciclo actual (Severson et al., 2019): el indicador individual más
    predictivo de la vida útil restante encontrado en la literatura."""
    delta_q = np.asarray(q_curve_cur) - np.asarray(q_curve_ref)
    return float(np.var(delta_q))


def discharge_slope(v_curve: np.ndarray, q_curve: np.ndarray) -> float:
    """Pendiente de la curva Q(V): resume qué tan "empinada" es la curva de
    capacidad frente a voltaje en ese ciclo."""
    return float(np.polyfit(v_curve, q_curve, 1)[0])


def extract_health_indicators(cycle_row: pd.Series, ref_q_curve: np.ndarray) -> dict:
    """Calcula los indicadores de salud de un ciclo, dado el ciclo de
    referencia (normalmente uno de los primeros ciclos de esa batería)."""
    return {
        "delta_q_var": delta_q_variance(ref_q_curve, cycle_row["q_curve"]),
        "charge_time": cycle_row["charge_time"],
        "t_mean": cycle_row["t_mean"],
        "t_max": cycle_row["t_max"],
        "discharge_slope": discharge_slope(cycle_row["v_curve"], cycle_row["q_curve"]),
    }


def build_feature_matrix(df: pd.DataFrame, target: str = "soh", ref_cycle: int = 5):
    """Construye la matriz X (features) y el vector y (SOH o RUL) a partir
    del DataFrame de ciclos. El ciclo de referencia para ΔQ(V) es, por
    batería, el ciclo `ref_cycle` (o el primero disponible si hay menos)."""
    if target not in {"soh", "rul"}:
        raise ValueError("target debe ser 'soh' o 'rul'")

    rows = []
    targets = []
    groups = []
    for battery_id, group in df.groupby("battery_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        ref_idx = min(ref_cycle, len(group) - 1)
        ref_q_curve = group.loc[ref_idx, "q_curve"]
        for _, cycle_row in group.iterrows():
            rows.append(extract_health_indicators(cycle_row, ref_q_curve))
            targets.append(cycle_row[target])
            groups.append(battery_id)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = pd.Series(targets, name=target)
    groups = pd.Series(groups, name="battery_id")
    return X, y, groups