"""Carga y organiza los datos de ciclos de baterías por celda.

DataLoader es el único punto del pipeline que conoce de dónde vienen los
datos. El resto del código (features, modelos, evaluación) solo trabaja
con el DataFrame en el "formato interno" descrito en el README, así que
conectar un dataset nuevo es tan sencillo como añadir un loader más aquí.
"""
from pathlib import Path

import pandas as pd

from src.data.synthetic import generate_dataset


class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]

    def load_cycles(self) -> pd.DataFrame:
        """Devuelve un DataFrame con una fila por ciclo (ver formato interno
        en el README). Fuentes disponibles: "synthetic" (placeholder) y
        "nasa" (dataset real NASA Battery Dataset)."""
        source = self.data_cfg.get("source", "synthetic")
        if source == "synthetic":
            df = generate_dataset(
                n_batteries=self.data_cfg.get("n_batteries", 24),
                n_cycles=self.data_cfg.get("n_cycles", 300),
                seed=self.config.get("seed", 42),
            )
        elif source == "nasa":
            from src.data.nasa_loader import load_nasa_dataset
            df = load_nasa_dataset(raw_dir=self.data_cfg.get("raw_dir", "data/raw/cleaned_dataset"))
        else:
            raise NotImplementedError(
                f"Fuente de datos '{source}' no implementada todavía. "
                "Añade tu loader real (p. ej. para CALCE/Oxford) en src/data/ "
                "y regístralo aquí."
            )
        return self.add_rul_labels(df)

    def add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade la columna `rul` (ciclos restantes hasta cruzar el umbral
        de EOL) a partir de la columna `soh`, calculada por batería."""
        threshold_pct = self.data_cfg.get("eol_threshold", 0.80) * 100
        out = []
        for battery_id, group in df.groupby("battery_id"):
            group = group.sort_values("cycle").reset_index(drop=True)
            below = group.index[group["soh"] <= threshold_pct]
            eol_cycle = group.loc[below[0], "cycle"] if len(below) else group["cycle"].max()
            group["rul"] = eol_cycle - group["cycle"]
            group["rul"] = group["rul"].clip(lower=0)
            out.append(group)
        return pd.concat(out, ignore_index=True)

    def split_by_battery(self, df: pd.DataFrame, test_batteries: list):
        """Divide train/test por batería completa (nunca por ciclo suelto),
        para no filtrar información de una misma celda entre train y test."""
        test_batteries = set(test_batteries)
        train = df[~df.battery_id.isin(test_batteries)].reset_index(drop=True)
        test = df[df.battery_id.isin(test_batteries)].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(
                "El split train/test ha quedado vacío: revisa "
                "`data.test_batteries` en configs/config.yaml."
            )
        return train, test

    def cache_processed(self, df: pd.DataFrame, name: str = "cycles.csv") -> Path:
        """Guarda una versión sin las señales temporales (no tabulares) en
        data/processed/, útil para inspección rápida en un notebook."""
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / name
        cols_to_drop = ["time_arr", "voltage_arr", "current_arr", "temp_arr"]
        df.drop(columns=cols_to_drop, errors="ignore").to_csv(out_path, index=False)
        return out_path
