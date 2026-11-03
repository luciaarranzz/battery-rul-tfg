"""Recuento exacto de las descargas descartadas por la limpieza, batería a batería."""
import numpy as np
import pandas as pd

from src.data.nasa_loader import load_nasa_dataset
from src.plot_cleaning_example import RAW_DIR, match_kept, raw_capacities


def main():
    meta = pd.read_csv(RAW_DIR / "metadata.csv")
    cleaned = load_nasa_dataset(str(RAW_DIR))

    rows = []
    for battery in sorted(meta["battery_id"].unique()):
        raw = raw_capacities(RAW_DIR, battery)
        if len(raw) == 0:
            continue
        kept = cleaned.loc[cleaned["battery_id"] == battery, "capacity"].to_numpy(dtype=float)
        is_kept = match_kept(raw, kept)
        valid = ~np.isnan(raw)
        x = np.arange(len(raw))
        first_kept = int(np.argmax(is_kept)) if is_kept.any() else len(raw)
        rows.append({
            "bateria": battery,
            "en_bruto": len(raw),
            "sin_capacidad": int((~valid).sum()),
            "regimen_inicial": int((~is_kept & valid & (x < first_kept)).sum()),
            "fuera_de_rango": int((~is_kept & valid & (x >= first_kept)).sum()),
            "conservadas": int(is_kept.sum()),
        })

    summary = pd.DataFrame(rows)
    removed = summary[["sin_capacidad", "regimen_inicial", "fuera_de_rango"]].sum(axis=1)
    print("Baterías con descargas descartadas:")
    print(summary[removed > 0].to_string(index=False))
    print()
    print("Totales:")
    print(summary.drop(columns="bateria").sum().to_string())


if __name__ == "__main__":
    main()