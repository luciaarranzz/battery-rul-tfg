# battery-rul-tfg

Estimación del estado de salud (SOH) y de la vida útil restante (RUL) de baterías de ion-litio con Machine Learning, y panel de monitorización.

Trabajo de Fin de Grado · Grado en Ingeniería Informática · Universidad de Burgos
Autora: Lucía Arranz Martínez · Tutor: César Represa Pérez

## Qué es

Un pipeline en Python (carga de datos → características → modelos → evaluación) que estima el SOH de una batería a partir de señales que proporcionaría un BMS real (tensión, corriente, temperatura y tiempo durante la descarga), y un dashboard en Streamlit que muestra las predicciones.

- **SOH (State of Health):** capacidad actual de la batería, en %, respecto a su capacidad de referencia.
- **RUL (Remaining Useful Life):** ciclos que le quedan hasta llegar al fin de vida, fijado en un SOH del 80 %.

Los datos son los del NASA Prognostics Center of Excellence (34 baterías en 9 grupos experimentales).

## Instalación

Requisitos: Python 3.14 y Git.

```bash
git clone <url-del-repositorio>
cd battery-rul-tfg
python -m venv .venv
.venv\Scripts\activate             # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Datos

1. Descargar la versión «cleaned» del NASA Battery Dataset desde Kaggle (`patrickfleith/nasa-battery-dataset`).
2. Descomprimirla en `data/raw/` de forma que existan `data/raw/cleaned_dataset/metadata.csv` y `data/raw/cleaned_dataset/data/*.csv`.

El cargador (`src/data/nasa_loader.py`) traduce ese formato al interno del proyecto y descarta los ciclos con capacidad vacía o prácticamente nula.
<!-- TODO: describir aquí el filtrado definitivo de ciclos anómalos y la referencia del SOH cuando estén cerrados -->

Las baterías de test se fijan en `configs/config.yaml` (`data.test_batteries`): una por cada grupo experimental.

## Uso

Con el entorno virtual activado, desde la raíz del proyecto:

| Comando | Qué hace |
|---|---|
| `python -m src.train` | Entrena los cuatro modelos y evalúa en las baterías de test |
| `python -m src.cross_validate` | Validación cruzada agrupada por batería (solo con las baterías de entrenamiento) |
| `python -m src.tune_lstm` | Ajuste de hiperparámetros de la LSTM con validación cruzada |
| `python -m src.make_predictions_table` | Genera `results/predictions.csv` con las predicciones de los cuatro modelos |
| `streamlit run app/dashboard.py` | Lanza el dashboard en `http://localhost:8501` |
| `python -m src.eda` | Análisis exploratorio: gráficas y tabla resumen en `results/` |
| `pytest -q` | Ejecuta los tests |

## Modelos

Regresión lineal (línea base), Random Forest, XGBoost y LSTM (PyTorch). Todos implementan la misma interfaz (`fit`, `predict`, `evaluate`) definida en `src/models/base_model.py`. Se evalúan con RMSE, MAE, MAPE y R² sobre baterías completas que no intervienen en el entrenamiento, para evitar fugas de datos entre ciclos de una misma batería.

<!-- TODO: añadir aquí la tabla de resultados cuando estén recalculados -->

## Características de entrada

Calculadas en `src/features.py` a partir de las series de cada ciclo de descarga: duración de la descarga, duración de la carga previa, temperatura media y máxima, y velocidad media de caída de tensión.
<!-- TODO: actualizar esta lista cuando se cierren las variables de entrada -->

## Estructura del proyecto

```text
battery-rul-tfg/
├── app/
│   ├── dashboard.py          # dashboard de Streamlit
│   └── cars.py               # siluetas de vehículos (ilustrativas)
├── configs/
│   └── config.yaml           # rutas, baterías de test, hiperparámetros, semilla
├── data/
│   └── raw/                  # dataset original (no se sube al repositorio)
├── results/                  # métricas, gráficas y predictions.csv
├── src/
│   ├── data/                 # carga de datos (loader.py, nasa_loader.py)
│   ├── models/               # base_model.py, classical_models.py, deep_models.py
│   ├── features.py           # indicadores de salud a partir de señales tipo BMS
│   ├── train.py              # entrenamiento y evaluación en test
│   ├── cross_validate.py     # validación cruzada agrupada por batería
│   ├── tune_lstm.py          # ajuste de la LSTM
│   ├── make_predictions_table.py
│   ├── evaluate.py           # métricas y gráficas
│   ├── eda.py                # análisis exploratorio
│   └── utils.py              # semillas y logging
├── tests/
├── .streamlit/config.toml    # temas claro y oscuro del dashboard
└── requirements.txt
```

## Tecnologías

Python · pandas · NumPy · scikit-learn · XGBoost · PyTorch · Matplotlib · Streamlit · pytest