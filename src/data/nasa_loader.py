"""Loader del dataset real NASA Battery Dataset (Li-ion PCoE), en su
versión "cleaned" descargada de Kaggle (patrickfleith/nasa-battery-dataset):
un `metadata.csv` que indexa cada ciclo (charge/discharge/impedance) y una
carpeta `data/` con un CSV por ciclo (medidas frente al tiempo).

Traduce ese formato real al mismo "formato interno" que usa synthetic.py,
para que el resto del pipeline (features, modelos, evaluación) no necesite
saber de dónde vienen los datos (ver README, "Formato interno de datos").

Solo se usan señales que un BMS real proporcionaría durante la descarga
(voltaje, corriente, temperatura, tiempo) -- nunca impedancia (EIS) ni la
curva completa capacidad-voltaje de laboratorio, siguiendo la indicación
del tutor (César, correo de septiembre 2026).
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _read_cycle_csv(raw_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "data" / filename)


def load_nasa_dataset(raw_dir: str = "data/raw/cleaned_dataset") -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por ciclo de descarga, en el
    mismo formato interno que synthetic.generate_dataset: battery_id,
    cycle, capacity, soh, charge_time, time_arr, voltage_arr, current_arr,
    temp_arr.

    `raw_dir` debe apuntar a la carpeta `cleaned_dataset` (la que contiene
    metadata.csv y la subcarpeta data/), tal y como queda tras descomprimir
    el dataset de Kaggle en data/raw/ (ver README, "Puesta en marcha").
    """
    raw_dir = Path(raw_dir)
    meta = pd.read_csv(raw_dir / "metadata.csv")

    rows = []
    for battery_id, group in meta.groupby("battery_id"):
        group = group.sort_values("test_id").reset_index(drop=True)
        discharge = group[group["type"] == "discharge"].copy()

        # Unas pocas filas de descarga traen la capacidad vacía en el CSV
        # (aparece literalmente como el texto "[]", herencia del .mat
        # original cuando esa medida no se llegó a registrar). Sin un
        # valor numérico de capacidad no podemos calcular el SOH de ese
        # ciclo, así que se descarta en vez de fallar todo el loader.
        discharge["Capacity"] = pd.to_numeric(discharge["Capacity"], errors="coerce")
        discharge = discharge.dropna(subset=["Capacity"])
        if discharge.empty:
            continue

        # También hay algún ciclo aislado con capacidad ~0 Ah (fallo del
        # sensor durante esa descarga, no una batería muerta de verdad:
        # ninguna celda pasa de un 70-80% de SOH a 0 Ah de un ciclo a otro
        # dentro del rango estudiado). Se descartan como lecturas inválidas
        # -- el mismo criterio que con la capacidad vacía de arriba -- para
        # que no contaminen las etiquetas de SOH.
        max_capacity = discharge["Capacity"].max()
        discharge = discharge[discharge["Capacity"] >= 0.05 * max_capacity]
        discharge = discharge.sort_values("test_id").reset_index(drop=True)
        if discharge.empty:
            continue

        # Duración (segundos) de cada ciclo de carga de esta batería, para
        # poder asignar a cada descarga la duración de la carga previa más
        # cercana (equivalente a `charge_time` en el dataset sintético).
        charge_durations = {}
        for _, charge_row in group[group["type"] == "charge"].iterrows():
            cdf = _read_cycle_csv(raw_dir, charge_row["filename"])
            if len(cdf) >= 2 and "Time" in cdf.columns:
                charge_durations[charge_row["test_id"]] = float(
                    cdf["Time"].iloc[-1] - cdf["Time"].iloc[0]
                )

        # Referencia para el SOH: la capacidad MÁXIMA observada en esta
        # batería (ya limpia de lecturas inválidas), no la del primer
        # ciclo registrado. Se probó con la del primer ciclo y en varios
        # grupos de baterías NASA (p. ej. B0033, B0039, B0041) esa primera
        # lectura es ella misma anómala -- mucho más baja que el resto --
        # así que normalizar contra ella disparaba el SOH por encima del
        # 1000%. Usar el máximo observado es además la definición habitual
        # de SOH (capacidad relativa a la capacidad de pico de esa celda).
        reference_capacity = float(discharge["Capacity"].max())
        battery_rows = []
        cycle_idx = 0

        for _, row in discharge.iterrows():
            cdf = _read_cycle_csv(raw_dir, row["filename"])
            signal_cols = ["Time", "Voltage_measured", "Current_measured", "Temperature_measured"]
            cdf[signal_cols] = cdf[signal_cols].apply(pd.to_numeric, errors="coerce")
            cdf = cdf.dropna(subset=signal_cols)
            if len(cdf) < 2:
                # Ciclo sin señales válidas suficientes (mismo problema que
                # la capacidad vacía, pero en el CSV del ciclo): se salta
                # en vez de romper el resto de la batería.
                continue

            time_arr = cdf["Time"].to_numpy(dtype=float)
            voltage_arr = cdf["Voltage_measured"].to_numpy(dtype=float)
            current_arr = cdf["Current_measured"].to_numpy(dtype=float)
            temp_arr = cdf["Temperature_measured"].to_numpy(dtype=float)

            # Carga previa más cercana: la de mayor test_id por debajo del
            # test_id de esta descarga.
            prior_charges = [tid for tid in charge_durations if tid < row["test_id"]]
            charge_time = charge_durations[max(prior_charges)] if prior_charges else np.nan

            capacity = float(row["Capacity"])
            battery_rows.append({
                "battery_id": battery_id,
                "cycle": cycle_idx,
                "capacity": capacity,
                # SOH normalizado respecto a la capacidad MÁXIMA observada
                # en esta batería (ver nota más arriba): cada grupo de
                # baterías NASA tiene una capacidad nominal ligeramente
                # distinta, así que no usamos una constante global como en
                # el dataset sintético (documentar esta decisión en la
                # memoria).
                "soh": capacity / reference_capacity * 100,
                "charge_time": charge_time,
                "time_arr": time_arr,
                "voltage_arr": voltage_arr,
                "current_arr": current_arr,
                "temp_arr": temp_arr,
            })
            cycle_idx += 1

        # Si el primer ciclo no tiene carga previa registrada (caso
        # habitual), rellenamos su charge_time con la mediana de esa misma
        # batería en vez de descartar el ciclo.
        median_ct = pd.Series([r["charge_time"] for r in battery_rows]).median()
        for r in battery_rows:
            if pd.isna(r["charge_time"]):
                r["charge_time"] = median_ct if not pd.isna(median_ct) else 0.0

        rows.extend(battery_rows)

    return pd.DataFrame(rows)
