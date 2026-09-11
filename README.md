# battery-rul-tfg

TFG — Aplicación de Machine Learning para la estimación y predicción de la vida útil de una batería.

Universidad de Burgos · Grado en Ingeniería Informática · Lucia Arranz Martinez

## Qué es esto

Un pipeline de Machine Learning en Python para dos problemas relacionados:

- **SOH (State of Health):** estimar el estado de salud actual de una batería (capacidad relativa, en %).
- **RUL (Remaining Useful Life):** predecir cuántos ciclos le quedan a una batería antes de llegar a su umbral de fin de vida (por defecto, 80% de la capacidad nominal).

El repositorio está organizado como un pipeline modular (carga de datos → features → modelos → evaluación), pensado para poder explicarse capítulo a capítulo en la memoria del TFG.

## Puesta en marcha rápida

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ejecuta el pipeline completo sobre datos sintéticos (no hace falta descargar nada):
python -m src.train
```

Esto genera un dataset sintético de baterías (ver más abajo por qué), entrena varios modelos, imprime sus métricas por consola y guarda resultados en `results/`.

```bash
# Lanza los tests:
pytest -q
```

## Por qué hay un generador de datos sintéticos

Para poder tener el pipeline funcionando de principio a fin desde el primer día —sin depender de descargar y adaptar el dataset real—, `src/data/synthetic.py` genera un conjunto de baterías simuladas con una curva de degradación de capacidad realista (decaimiento tipo exponencial + ruido) y una curva Q(V) sintética por ciclo.

**Esto es un placeholder, no el resultado final del TFG.** El siguiente paso natural es sustituirlo por el dataset real elegido (ver `docs/` o la propuesta en Word entregada aparte, sección 4): normalmente el **NASA Battery Dataset** como punto de partida. Para ello solo hay que:

1. Descargar el dataset (p. ej. desde Kaggle: `patrickfleith/nasa-battery-dataset`) en `data/raw/`.
2. Escribir un loader específico en `src/data/loader.py` (o uno nuevo, p. ej. `nasa_loader.py`) que traduzca el formato real del dataset al mismo formato interno que ya usa `synthetic.py` (ver la sección "Formato interno de datos" más abajo).
3. Cambiar en `configs/config.yaml` el `data_source` de `synthetic` a `nasa` (o el nombre que le des).

El resto del pipeline (features, modelos, evaluación) no necesita tocarse: solo depende del formato interno, no de dónde vienen los datos.

## Formato interno de datos

Cada ciclo se representa como un diccionario/fila con:

| Campo | Descripción |
|---|---|
| `battery_id` | identificador de la celda (str) |
| `cycle` | número de ciclo (int) |
| `capacity` | capacidad medida en ese ciclo, en Ah |
| `soh` | capacidad relativa respecto a la nominal, en % |
| `charge_time` | duración de la fase de carga, en minutos |
| `t_mean`, `t_max` | temperatura media/máxima durante el ciclo, en °C |
| `v_curve`, `q_curve` | arrays de voltaje y capacidad acumulada del ciclo (para health indicators tipo ΔQ(V)) |

## Estructura del proyecto

```text
battery_rul_tfg/
├── configs/
│   └── config.yaml          # rutas, hiperparámetros, semillas
├── data/
│   ├── raw/                 # dataset original (NASA/CALCE), sin tocar
│   └── processed/           # datos intermedios cacheados
├── notebooks/
│   └── 01_eda.ipynb         # análisis exploratorio de partida
├── src/
│   ├── data/
│   │   ├── synthetic.py     # generador de datos sintéticos (placeholder)
│   │   └── loader.py        # DataLoader: carga y split por batería
│   ├── features.py          # extracción de indicadores de salud (HI)
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

- Sustituir `synthetic.py` por un loader del dataset real (NASA/CALCE).
- Ampliar `features.py` con más indicadores de salud (curvas de capacidad incremental, resistencia interna).
- Añadir interpretabilidad (SHAP) sobre el mejor modelo ensemble.
- Documentar resultados y decisiones en la memoria del TFG.