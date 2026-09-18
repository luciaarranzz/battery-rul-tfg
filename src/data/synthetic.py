"""Generador de datos sintéticos de degradación de baterías.

Es un PLACEHOLDER: sirve para tener el pipeline completo funcionando de
principio a fin sin depender de descargar el dataset real, y también como
comprobación rápida del pipeline sin esperar a leer cientos de CSV reales.
Simula, para cada batería, una curva de degradación de capacidad con forma
realista (decaimiento exponencial + lineal + ruido) y, para cada ciclo de
descarga, una señal temporal de voltaje/corriente/temperatura del mismo
tipo que produciría un BMS real -- el mismo formato que genera el loader
del dataset real (src/data/nasa_loader.py), para que features.py funcione
exactamente igual con ambas fuentes.

El dataset real ya está integrado (ver src/data/nasa_loader.py); este
generador se mantiene como modo rápido de comprobación del pipeline.
"""
import numpy as np
import pandas as pd

NOMINAL_CAPACITY_AH = 2.0
N_SAMPLES = 60             # muestras por ciclo de descarga simulado
DISCHARGE_SECONDS = 3000   # duración típica de una descarga completa (~50 min)


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


def _discharge_signal(soh: float, rng: np.random.Generator, base_temp: float):
    """Señal simulada de un ciclo de descarga (voltaje, corriente y
    temperatura frente al tiempo), con forma parecida a una descarga real:
    el voltaje cae de ~4.2V a ~2.7V, la corriente es aprox. constante, y
    la temperatura sube algo más cuanto peor está la batería."""
    duration = DISCHARGE_SECONDS * (0.55 + 0.45 * soh)  # peor SOH -> descargas más cortas
    time_arr = np.linspace(0, duration, N_SAMPLES)
    v_start, v_end = 4.2, 2.7 + 0.3 * (1 - soh)
    shape = (time_arr / duration) ** (0.7 + 0.6 * (1 - soh))
    voltage_arr = v_start + (v_end - v_start) * shape + rng.normal(0, 0.01, size=N_SAMPLES)
    current_arr = -2.0 + rng.normal(0, 0.03, size=N_SAMPLES)
    temp_arr = base_temp + 6 * (1 - soh) * (time_arr / duration) + rng.normal(0, 0.3, size=N_SAMPLES)
    return time_arr, voltage_arr, current_arr, temp_arr


def generate_dataset(n_batteries: int = 24, n_cycles: int = 300, seed: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por ciclo de descarga, en el
    formato interno descrito en el README (battery_id, cycle, capacity,
    soh, charge_time, time_arr, voltage_arr, current_arr, temp_arr)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_batteries):
        battery_id = f"battery_{i:02d}"
        soh_curve = _soh_curve(n_cycles, rng)
        base_charge_time = rng.uniform(2700, 3900)  # segundos
        base_temp = rng.uniform(24, 32)
        for cycle, soh in enumerate(soh_curve):
            time_arr, voltage_arr, current_arr, temp_arr = _discharge_signal(soh, rng, base_temp)
            charge_time = base_charge_time * (2 - soh) + rng.normal(0, 90)
            rows.append({
                "battery_id": battery_id,
                "cycle": cycle,
                "capacity": NOMINAL_CAPACITY_AH * soh,
                "soh": soh * 100,
                "ambient_temp": base_temp,
                "charge_time": charge_time,
                "time_arr": time_arr,
                "voltage_arr": voltage_arr,
                "current_arr": current_arr,
                "temp_arr": temp_arr,
            })
    return pd.DataFrame(rows)
