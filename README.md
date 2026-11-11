# 🔋 EVcheck

[![Quality gate](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=alert_status)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=bugs)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Code smells](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=code_smells)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=vulnerabilities)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Duplicated lines](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=duplicated_lines_density)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Security](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=security_rating)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=reliability_rating)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=sqale_rating)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)
[![Lines of code](https://sonarcloud.io/api/project_badges/measure?project=luciaarranzz_battery-rul-tfg&metric=ncloc)](https://sonarcloud.io/project/overview?id=luciaarranzz_battery-rul-tfg)

**EVcheck** estima el **estado de salud (SOH)** y la **vida útil restante (RUL)** de baterías de iones de litio mediante aprendizaje automático, a partir de los datos de envejecimiento del repositorio NASA PCoE.

Incluye una **aplicación web** en Streamlit con dos vistas según el rol de la cuenta:

- **Usuario** (propietario del vehículo): consulta el estado de la batería de su vehículo por matrícula y sube las lecturas de un BMS en formato CSV para obtener el SOH y la RUL estimados.
- **Administrador** (personal del taller): panel de monitorización donde se compara la predicción de varios algoritmos con el valor real, ciclo a ciclo.

Trabajo de Fin de Grado del Grado en Ingeniería Informática, Universidad de Burgos.

> ⚠️ Los modelos se han entrenado con datos de laboratorio de celdas NASA, no con lecturas de un BMS real. La aplicación es un **simulador** que muestra cómo funcionaría el sistema, no una herramienta de diagnóstico de vehículos.

## 🚀 Despliegue

### Requisitos previos

1. Git *(para clonar el repositorio)*
2. Python 3.10 o superior

### Pasos de instalación

#### 1. Descargar el código fuente

```bash
git clone https://github.com/luciaarranzz/battery-rul-tfg.git
cd battery-rul-tfg
```

#### 2. Crear el entorno virtual e instalar las dependencias

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

#### 3. Arrancar la aplicación

```bash
python -m streamlit run app/main.py
```

Abre el navegador en [http://localhost:8501](http://localhost:8501). La primera vez se crea la base de datos local `data/app.db` con las cuentas y el vehículo de demostración.

El repositorio ya incluye el modelo entrenado de la aplicación (`models/xgboost_soh.joblib`) y los CSV de demostración (`data/demo/`). Los pasos siguientes solo hacen falta para regenerarlos o para el panel del administrador.

#### 4. (Opcional) Regenerar datos, modelos y predicciones

Descarga el dataset [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) y descomprímelo en `data/raw/`, de forma que exista la carpeta `data/raw/cleaned_dataset/` (la ruta se puede cambiar en `configs/config.yaml`). Después:

```bash
python -m src.train                  # entrena y evalúa los modelos
python -m src.make_predictions_table # genera la tabla que lee el panel del administrador
python -m src.export_model           # guarda el modelo XGBoost que usa la app
python -m src.make_demo_csv          # genera los CSV de demostración
```

> 💡 Los hiperparámetros, las baterías de test y el umbral de fin de vida (80 %) se configuran en `configs/config.yaml`.
>
> El modelo se guarda con `joblib`, que depende de las versiones de las librerías. Si cargas un modelo con versiones distintas a las de `requirements.txt`, vuelve a generarlo con `python -m src.export_model`.

## 🧪 Cómo probar la aplicación

Las cuentas y la matrícula de demostración son ficticias.

| Cuenta | Usuario | Contraseña |
|---|---|---|
| Usuario | `usuario_demo` | `demo-usuario-2026` |
| Administrador | `admin_demo` | `demo-admin-2026` |

### Vista de usuario

1. Entra con `usuario_demo` e introduce la matrícula `1234BCD`. Es un vehículo con historial de 60 ciclos: se muestran el SOH estimado, el estado, la autonomía ilustrativa y la gráfica de degradación.
2. Sube `data/demo/demo_continuacion.csv` (ciclos 61 a 120 del mismo vehículo) y pulsa **Guardar lecturas y calcular**. El historial pasa a 120 ciclos y aparece una estimación de la RUL.
3. Cierra sesión, entra de nuevo y prueba una matrícula nueva, por ejemplo `5678XYZ`. Al no tener historial, la aplicación pide directamente el CSV: sube `data/demo/demo_vehiculo_nuevo.csv`.
4. Con otra matrícula nueva, sube `data/demo/demo_con_errores.csv` para ver los avisos de validación (ciclos que faltan, valores fuera de rango, muestras insuficientes y filas descartadas). Usa una matrícula nueva porque estos ciclos sustituyen a los que ya estén guardados.

### Vista de administrador

Entra con `admin_demo`: se abre el panel de monitorización con la comparación de modelos sobre las baterías de test.

### Reiniciar los datos de demostración

Cierra la aplicación, borra el fichero `data/app.db` y vuelve a arrancarla: se regenera con las cuentas y el vehículo de demostración.

### Formato del CSV de lecturas

Una fila por cada muestra del BMS durante una descarga, con las columnas:

| Columna | Descripción |
|---|---|
| `ciclo` | Número de ciclo (entero, empezando en 1) |
| `tiempo_s` | Tiempo desde el inicio de la descarga (s) |
| `voltaje_v` | Tensión de la batería (V) |
| `temperatura_c` | Temperatura de la batería (°C) |
| `tiempo_carga_s` | Duración de la carga previa (s); se repite en todas las filas del ciclo |
| `temp_ambiente_c` | Temperatura ambiente (°C); se repite en todas las filas del ciclo |

### Limitaciones

- Un vehículo nuevo necesita al menos 3 ciclos, porque el SOH se calcula respecto a los 3 primeros.
- Si se registra por primera vez un vehículo con la batería ya envejecida, el SOH saldrá más alto del real, porque la referencia son sus primeros ciclos registrados.
- La RUL es una extrapolación lineal de la tendencia de los últimos 60 ciclos y solo se da cuando esa tendencia es clara frente al ruido.

## 🛠️ Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)