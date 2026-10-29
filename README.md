# 🔋 Battery RUL

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

**Battery RUL** estima el **estado de salud (SOH)** y la **vida útil restante (RUL)** de baterías de iones de litio mediante aprendizaje automático, a partir de los datos de envejecimiento del repositorio NASA PCoE. Incluye un **panel de monitorización** en Streamlit donde se compara la predicción de varios algoritmos con el valor real, ciclo a ciclo.

Trabajo de Fin de Grado del Grado en Ingeniería Informática, Universidad de Burgos.

## 🚀 Despliegue

### Requisitos previos

1. Git *(para clonar el repositorio)*
2. Python 3.10 o superior

### Pasos de instalación

#### 1. Descargar el código fuente

```bash
git clone https://github.com/USUARIO/battery-rul-tfg.git
cd battery-rul-tfg
```

#### 2. Crear el entorno virtual e instalar las dependencias

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

#### 3. Obtener los datos y generar las predicciones

Descarga el dataset [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) y descomprímelo en `data/raw/`, de forma que exista la carpeta `data/raw/cleaned_dataset/` (la ruta se puede cambiar en `configs/config.yaml`). Después, entrena los modelos y genera la tabla de predicciones que lee el panel:

```bash
python -m src.train
python -m src.make_predictions_table
```

#### 4. Arrancar el panel

```bash
streamlit run app/dashboard.py
```

Abre el navegador en [http://localhost:8501](http://localhost:8501).

> 💡 Los hiperparámetros, las baterías de test y el umbral de fin de vida (80 %) se configuran en `configs/config.yaml`.

## 🛠️ Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)