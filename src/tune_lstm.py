"""Ajuste de hiperparámetros de la LSTM con validación cruzada agrupada.

Uso:
    python -m src.tune_lstm

Solo usa las baterías de TRAIN: el test no se toca. Cada variante cambia
únicamente los parámetros indicados respecto a configs/config.yaml.
"""
import copy

from src.cross_validate import cross_validate_lstm, load_config, summarize
from src.data.loader import DataLoader
from src.utils import get_logger, set_seed

log = get_logger(__name__)

# Cada variante modifica solo lo que se indica; el resto sale del YAML.
VARIANTES = {
    "base": {},
    "seq_10": {"sequence_length": 10},
    "hidden_32": {"hidden_size": 32},
    "epochs_40": {"epochs": 40},
    "lr_0.001": {"lr": 0.001},
}

def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, _ = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])

    resultados = {}
    for nombre, cambios in VARIANTES.items():
        cfg_v = copy.deepcopy(cfg)
        cfg_v["models"]["lstm"].update(cambios)
        set_seed(seed)  # misma semilla para todas: comparación justa
        log.info("Probando '%s': %s", nombre, cfg_v["models"]["lstm"])
        resultados[nombre] = cross_validate_lstm(train_df, cfg_v, target=target, n_splits=5)

    tabla = summarize(resultados)
    log.info("Resumen del ajuste de la LSTM:\n%s", tabla.round(3))
    tabla.to_csv(f"{cfg['output']['results_dir']}/lstm_tuning.csv")
    return tabla


if __name__ == "__main__":
    main()