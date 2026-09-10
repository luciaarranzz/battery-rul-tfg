"""Generador de datos sintéticos de degradación de baterías.

Es un PLACEHOLDER: sirve para tener el pipeline completo funcionando de
principio a fin sin depender de descargar el dataset real. Simula, para cada
batería, una curva de degradación de capacidad con forma realista
(decaimiento exponencial + lineal + ruido) y, para cada ciclo, una curva
Q(V) sintética de la que se pueden extraer indicadores de salud (health
indicators) del mismo tipo que se usarían con datos reales (p. ej. NASA).

Sustituir por un loader real es el primer paso natural del TFG una vez
elegido el dataset definitivo (ver README.md).
"""
import numpy as np
import pandas as pd

NOMINAL_CAPACITY_AH = 2.0
V_GRID = np.linspace(3.0, 4.2, 50)


def _soh_curve(n_cycles: int, rng: np.random.Generator) -> np.ndarray:
    """Genera una curva de SOH (0-1) con forma de degradación realista:
    una fase inicial casi plana y una caída que se acelera con el uso,
    con parámetros aleatorios por batería para que no todas degraden igual."""
    a = rng.uniform(0.15, 0.35)         # amplitud de la caída "rápida" inicial
    b = rng.uniform(0.004, 0.012)       # velocidad de esa caída
    c = rng.uniform(0.0003, 0.0009)     # pendiente lineal de degradación lenta
    cycles = np.arange(n_cycles)
    soh = 1.0 - a * (1 - np.exp(-b * cycles)) - c * cycles
    noise = rng.normal(0, 0.004, size=n_cycles)
    soh = np.clip(soh + noise, 0.4, 1.0)
    # Aseguramos monotonía no creciente suavizando picos de ruido hacia arriba.
    soh = np.minimum.accumulate(soh)
    return soh


def _q_curve_for_cycle(soh: float, rng: np.random.Generator) -> np.ndarray:
    """Curva Q(V) sintética para un ciclo dado su SOH: una sigmoide escalada
    por la capacidad disponible, con un ligero desplazamiento de la meseta
    de voltaje a medida que la celda degrada (efecto observado en literatura)."""
    v_mid = 3.7 - 0.05 * (1 - soh)   # el punto medio se desplaza al degradar
    spread = 0.12
    shape = 1 / (1 + np.exp(-(V_GRID - v_mid) / spread))
    noise = rng.normal(0, 0.005, size=V_GRID.shape)
    return NOMINAL_CAPACITY_AH * soh * shape + noise


def generate_dataset(n_batteries: int = 24, n_cycles: int = 300, seed: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por ciclo, en el formato interno
    descrito en el README (battery_id, cycle, capacity, soh, charge_time,
    t_mean, t_max, v_curve, q_curve)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_batteries):
        battery_id = f"battery_{i:02d}"
        soh_curve = _soh_curve(n_cycles, rng)
        base_charge_time = rng.uniform(45, 65)  # minutos
        base_temp = rng.uniform(24, 32)
        for cycle, soh in enumerate(soh_curve):
            q_curve = _q_curve_for_cycle(soh, rng)
            charge_time = base_charge_time * (2 - soh) + rng.normal(0, 1.5)
            t_mean = base_temp + 3 * (1 - soh) + rng.normal(0, 0.5)
            t_max = t_mean + rng.uniform(2, 5)
            rows.append({
                "battery_id": battery_id,
                "cycle": cycle,
                "capacity": NOMINAL_CAPACITY_AH * soh,
                "soh": soh * 100,
                "charge_time": charge_time,
                "t_mean": t_mean,
                "t_max": t_max,
                "v_curve": V_GRID,
                "q_curve": q_curve,
            })
    return pd.DataFrame(rows)