"""Dashboard de monitorización del SOH de baterías (versión 1: tabla)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

CSV_PATH = Path(__file__).resolve().parents[1] / "results" / "predictions.csv"

st.set_page_config(page_title="Monitorización de baterías", layout="wide")


@st.cache_data
def load_predictions(path):
    return pd.read_csv(path)


df = load_predictions(CSV_PATH)

st.title("Monitorización del estado de salud (SOH) de baterías")
st.caption(
    "Predicciones de XGBoost sobre las 34 baterías NASA. "
    "'test' son baterías que el modelo no vio al entrenar; "
    "'validación cruzada' son las de entrenamiento, predichas por un modelo que no las vio."
)

st.sidebar.header("Filtros")
todos_grupos = sorted(df["grupo"].unique())
grupos = st.sidebar.multiselect("Grupo experimental", todos_grupos, default=todos_grupos)
todos_origenes = sorted(df["origen"].unique())
origenes = st.sidebar.multiselect("Origen", todos_origenes, default=todos_origenes)
disponibles = sorted(df.loc[df["grupo"].isin(grupos), "bateria"].unique())
baterias = st.sidebar.multiselect("Batería", disponibles, default=disponibles)

filtrado = df[
    df["grupo"].isin(grupos) & df["origen"].isin(origenes) & df["bateria"].isin(baterias)
]
if filtrado.empty:
    st.warning("No hay datos con estos filtros.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Filas", len(filtrado))
col2.metric("RMSE (puntos de SOH)", f"{np.sqrt((filtrado['error'] ** 2).mean()):.2f}")
col3.metric("Error absoluto medio", f"{filtrado['error_abs'].mean():.2f}")

st.dataframe(filtrado, width="stretch", hide_index=True)