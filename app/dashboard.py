"""Dashboard de monitorización del SOH de baterías (versión 1: tabla)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from cars import CARS, car_html


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
col1.metric("Filas", len(filtrado), border=True)
col2.metric("RMSE (puntos de SOH)", f"{np.sqrt((filtrado['error'] ** 2).mean()):.2f}", border=True)
col3.metric("Error absoluto medio", f"{filtrado['error_abs'].mean():.2f}", border=True)


st.subheader("SOH real frente a predicho")
bateria_sel = st.selectbox("Batería a mostrar", sorted(filtrado["bateria"].unique()))
una = filtrado[filtrado["bateria"] == bateria_sel].set_index("ciclo")
rmse_una = np.sqrt((una["error"] ** 2).mean())
st.caption(f"{bateria_sel} · {una['origen'].iloc[0]} · RMSE {rmse_una:.2f} puntos de SOH")
tipo = st.selectbox("Tipo de vehículo (ilustrativo)", list(CARS))
tema = st.context.theme.type or "light"
components.html(car_html(tipo, tema), height=190)

soh_final = una["soh_predicho"].iloc[-1]
estado = "Fin de vida" if soh_final < 80 else "Aviso" if soh_final < 90 else "OK"
autonomia = CARS[tipo]["range_km"] * soh_final / 100
c1, c2, c3 = st.columns(3)
c1.metric("SOH predicho (último ciclo)", f"{soh_final:.0f} %", border=True)
c2.metric("Estado", estado, border=True)
c3.metric("Autonomía ilustrativa", f"{autonomia:.0f} km", border=True)
st.line_chart(una[["soh_real", "soh_predicho"]])

st.dataframe(filtrado, width="stretch", hide_index=True)