# battery-rul-tfg

TFG — Aplicación de Machine Learning para la estimación y predicción de la vida útil de una batería.

Universidad de Burgos · Grado en Ingeniería Informática · Lucia Arranz Martinez

## Qué es esto

Un pipeline de Machine Learning en Python para dos problemas relacionados:

- **SOH (State of Health):** estimar el estado de salud actual de una batería (capacidad relativa, en %).
- **RUL (Remaining Useful Life):** predecir cuántos ciclos le quedan a una batería antes de llegar a su umbral de fin de vida (por defecto, 80% de la capacidad nominal).

El repositorio está organizado como un pipeline modular (carga de datos → features → modelos → evaluación), pensado para poder explicarse capítulo a capítulo en la memoria del TFG.

La dirección final acordada con el tutor (César) es un **dashboard de monitorización** que simula la conexión con un vehículo eléctrico y estima el SOH a partir de datos tipo BMS (Battery Management System) — no una app de "sube un CSV y te doy una predicción". Por eso todo el pipeline, desde la carga de datos hasta las features, usa **solo señales que un BMS real proporcionaría** (voltaje, corriente, temperatura y tiempo durante la descarga): nunca datos exclusivos de laboratorio como la espectroscopía de impedancia (EIS) o la curva completa capacidad-voltaje.

## Puesta en marcha rápida

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ejecuta el pipeline completo (por defecto usa el dataset real, ver abajo):
python -m src.train
```

Esto entrena varios modelos sobre los ciclos de descarga del dataset, imprime sus métricas por consola y guarda resultados en `results/`.

```bash
# Lanza los tests:
pytest -q
```

### Dataset real (NASA Battery Dataset)

El dataset ya está integrado y es la fuente por defecto (`data.source: nasa` en `configs/config.yaml`).

1. Descargar la versión "cleaned" del NASA Battery Dataset desde Kaggle (`patrickfleith/nasa-battery-dataset`).
2. Descomprimirlo dentro de `data/raw/`, de forma que quede `data/raw/cleaned_dataset/metadata.csv` y `data/raw/cleaned_dataset/data/*.csv`.
3. Listo: `src/data/nasa_loader.py` traduce ese formato al formato interno del pipeline (ver más abajo).

Son 34 baterías repartidas en 9 grupos experimentales; `configs/config.yaml` reserva una batería de cada grupo para test (`data.test_batteries`), así el conjunto de test cubre condiciones distintas a las de entrenamiento.

Para probar el pipeline sin el dataset descargado (p. ej. para depurar rápido), se puede cambiar `data.source` a `synthetic` en `configs/config.yaml`: `src/data/synthetic.py` genera baterías simuladas con una curva de degradación realista, en el mismo formato interno.

## Formato interno de datos

Cada fila del DataFrame que produce `DataLoader.load_cycles()` es un ciclo de descarga:

| Campo | Descripción |
|---|---|
| `battery_id` | identificador de la celda (str) |
| `cycle` | número de ciclo de descarga, empezando en 0 (int) |
| `capacity` | capacidad medida en ese ciclo, en Ah |
| `soh` | capacidad relativa respecto al primer ciclo registrado de esa batería, en % |
| `rul` | ciclos restantes hasta cruzar el umbral de fin de vida (lo añade `DataLoader.add_rul_labels`) |
| `charge_time` | duración de la carga previa más cercana, en segundos |
| `time_arr`, `voltage_arr`, `current_arr`, `temp_arr` | series temporales de la descarga (tiempo, voltaje, corriente, temperatura) — señales tipo BMS |

`src/features.py` calcula, a partir de esas series, los indicadores de salud (Health Indicators) que realmente entrenan los modelos:

| Feature | Descripción |
|---|---|
| `discharge_duration` | duración total de la descarga (s) |
| `charge_time` | duración de la carga previa (s) |
| `t_mean`, `t_max` | temperatura media/máxima durante la descarga (°C) |
| `voltage_drop_rate` | velocidad media de caída de voltaje durante la descarga (V/s) |

## Estructura del proyecto

```text
battery_rul_tfg/
├── configs/
│   └── config.yaml          # rutas, hiperparámetros, semillas
├── data/
│   ├── raw/                 # dataset original (NASA/CALCE), sin tocar
│   │   └── cleaned_dataset/ # metadata.csv + data/*.csv (NASA, desde Kaggle)
│   └── processed/           # datos intermedios cacheados
├── notebooks/
│   └── 01_eda.ipynb         # análisis exploratorio de partida
├── src/
│   ├── data/
│   │   ├── synthetic.py     # generador de datos sintéticos (placeholder)
│   │   ├── nasa_loader.py   # loader del dataset real NASA Battery Dataset
│   │   └── loader.py        # DataLoader: elige fuente, split por batería, RUL
│   ├── features.py          # extracción de indicadores de salud (HI), solo señales tipo BMS
│   ├── models/
│   │   ├── base_model.py    # interfaz común (patrón Strategy)
│   │   ├── classical_models.py  # RandomForest, XGBoost
│   │   └── deep_models.py       # LSTM (PyTorch)
│   ├── train.py             # entrena y compara todos los modelos
│   ├── evaluate.py          # métricas y gráficas
│   └── utils.py             # semillas, logging
├── tests/
│   └── test_features.py     # tests de las funciones de features
├── results/                  # métricas, gráficas y modelos entrenados
└── requirements.txt
```

## Modelos incluidos

- **Baseline:** regresión lineal / Ridge.
- **Ensemble:** Random Forest, XGBoost.
- **Secuencial (opcional):** LSTM en PyTorch, sobre secuencias de vectores de features por batería.

Todos implementan la misma interfaz (`fit`, `predict`, `evaluate`) definida en `src/models/base_model.py`, así que `train.py` puede entrenarlos y compararlos en un único bucle sin código específico por modelo.

## Métricas

RMSE, MAE, MAPE y R², calculadas en `src/evaluate.py` sobre un conjunto de test dividido **por batería completa** (nunca por ciclo suelto), para evitar fuga de datos entre train y test.

## Próximos pasos sugeridos

- Construir el dashboard de monitorización (Streamlit) que simula la conexión con el vehículo y muestra el SOH estimado en tiempo real, sobre los modelos ya entrenados.
- Ampliar `features.py` con más indicadores de salud calculables desde señales tipo BMS (p. ej. pendiente de descarga por tramos, tiempo hasta un umbral de voltaje).
- Añadir interpretabilidad (SHAP) sobre el mejor modelo ensemble.
- Documentar en la memoria (plantilla oficial, LaTeX) el formato interno, la decisión de restringir las features a señales tipo BMS, y los resultados por modelo.
