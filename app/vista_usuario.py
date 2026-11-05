"""Vista del usuario (dueño del vehículo): matrícula, historial, subida de lecturas y predicción."""
import altair as alt
import pandas as pd
import streamlit as st
from cars import CARS, car_html

from src import db, servicio
from src.data.bms_csv import COLUMNAS, procesar_csv
from src.features import N_REF_CYCLES
from src.inference import cargar_modelo
from src.rul import EOL_SOH, VENTANA, estimar_rul


@st.cache_resource
def modelo_guardado():
    return cargar_modelo()


def pedir_matricula() -> str | None:
    """Pide la matrícula y la devuelve normalizada, o None si falta o no es válida."""
    with st.form("form_matricula"):
        entrada = st.text_input(
            "Introduce la matrícula de tu vehículo",
            value=st.session_state.get("matricula_actual", ""),
            placeholder="1234 BCD",
        )
        enviado = st.form_submit_button("Continuar")
    if enviado:
        st.session_state["matricula_actual"] = entrada

    texto = st.session_state.get("matricula_actual", "")
    if not texto:
        st.info("Introduce tu matrícula para ver el estado de la batería.")
        return None
    try:
        return db.normalizar_matricula(texto)
    except ValueError as exc:
        st.error(str(exc))
        return None


def mostrar_coche(tipo: str):
    tema = st.context.theme.type or "light"
    st.iframe(car_html(tipo, tema), height=200)


def grafica(historial: pd.DataFrame):
    if (st.context.theme.type or "light") == "dark":
        color_soh, color_tendencia, color_umbral = "#FFB020", "#ECEAE4", "#FF5A4E"
    else:
        color_soh, color_tendencia, color_umbral = "#E8590C", "#1B1B1F", "#D64545"

    linea = (
        alt.Chart(historial)
        .mark_line(strokeWidth=2.5, color=color_soh)
        .encode(
            x=alt.X("ciclo:Q", title="Ciclo"),
            y=alt.Y("soh_predicho:Q", title="SOH estimado (%)", scale=alt.Scale(zero=False)),
            tooltip=["ciclo", alt.Tooltip("soh_predicho:Q", format=".1f", title="SOH estimado")],
        )
    )
    tendencia = (
        alt.Chart(historial.tail(VENTANA))
        .transform_regression("ciclo", "soh_predicho")
        .mark_line(strokeDash=[6, 4], color=color_tendencia)
        .encode(x="ciclo:Q", y="soh_predicho:Q")
    )
    umbral = (
        alt.Chart(pd.DataFrame({"y": [EOL_SOH]}))
        .mark_rule(strokeDash=[6, 4], color=color_umbral)
        .encode(y="y:Q")
    )
    st.altair_chart((linea + tendencia + umbral).properties(height=340), width="stretch")
    st.caption(
        f"Línea continua: SOH estimado por el modelo. Discontinua: tendencia de los últimos "
        f"{VENTANA} ciclos. Roja: fin de vida ({EOL_SOH:.0f} %)."
    )


def mostrar_historial(conn, matricula: str, tipo: str):
    try:
        modelo, columnas = modelo_guardado()
    except FileNotFoundError:
        st.error("No se encuentra el modelo guardado. Hay que ejecutar: python -m src.export_model")
        return
    historial = servicio.historial_con_soh(conn, matricula, modelo, columnas)
    if historial.empty:
        st.warning("Este vehículo todavía no tiene lecturas suficientes.")
        return

    soh_actual = historial["soh_predicho"].iloc[-1]
    rul = estimar_rul(historial["ciclo"], historial["soh_predicho"])
    autonomia = CARS.get(tipo, CARS["Urbano"])["range_km"] * soh_actual / 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SOH estimado", f"{soh_actual:.0f} %", border=True)
    c2.metric("Estado", servicio.estado_bateria(soh_actual), border=True)
    c3.metric("Autonomía ilustrativa", f"{autonomia:.0f} km", border=True)
    restantes = "—" if rul.ciclos_restantes is None else f"{rul.ciclos_restantes:.0f}"
    c4.metric(f"Ciclos hasta el {EOL_SOH:.0f} %", restantes, border=True)
    st.caption(rul.mensaje)

    grafica(historial)
    with st.expander("Historial de mediciones"):
        tabla = historial.rename(
            columns={
                "ciclo": "Ciclo",
                "soh_predicho": "SOH estimado (%)",
                "discharge_duration": "Duración de la descarga (s)",
                "charge_time": "Duración de la carga (s)",
                "t_mean": "Temperatura media (°C)",
                "t_max": "Temperatura máxima (°C)",
            }
        )[["Ciclo", "SOH estimado (%)", "Duración de la descarga (s)", "Duración de la carga (s)",
           "Temperatura media (°C)", "Temperatura máxima (°C)"]]
        st.dataframe(tabla.round(2), width="stretch", hide_index=True)


def subir_lecturas(conn, usuario: dict, matricula: str, tipo: str):
    st.subheader("Subir lecturas del BMS")
    with st.expander("Formato del CSV"):
        st.write(
            "Una fila por cada muestra de una descarga, con las columnas: "
            + ", ".join(f"`{c}`" for c in COLUMNAS)
            + ". El tiempo de carga y la temperatura ambiente se repiten en todas las filas del ciclo."
        )
    archivo = st.file_uploader(
        "CSV con las lecturas", type="csv", key=f"csv_{st.session_state.get('n_subidas', 0)}"
    )
    if archivo is None:
        return

    archivo.seek(0)
    resultado = procesar_csv(archivo)
    for error in resultado.errores:
        st.error(error)
    for aviso in resultado.avisos:
        st.warning(aviso)
    if resultado.errores:
        return

    ciclos = resultado.ciclos
    st.success(
        f"El CSV tiene {len(ciclos)} ciclos válidos (del {ciclos['ciclo'].min()} "
        f"al {ciclos['ciclo'].max()})."
    )
    repetidos = servicio.ciclos_repetidos(conn, matricula, ciclos)
    if repetidos:
        st.info(f"{len(repetidos)} de esos ciclos ya estaban guardados y se sustituirán.")

    if st.button("Guardar lecturas y calcular"):
        try:
            guardados = servicio.guardar_lecturas(conn, usuario, matricula, tipo, ciclos)
        except (ValueError, PermissionError) as exc:
            st.error(str(exc))
            return
        st.session_state["mensaje"] = f"Se han guardado {guardados} ciclos."
        st.session_state["n_subidas"] = st.session_state.get("n_subidas", 0) + 1
        st.rerun()


def vista(conn):
    usuario = st.session_state["usuario"]
    st.title("Mi vehículo")
    mensaje = st.session_state.pop("mensaje", None)
    if mensaje:
        st.success(mensaje)

    matricula = pedir_matricula()
    if matricula is None:
        return
    try:
        vehiculo = servicio.vehiculo_accesible(conn, usuario, matricula)
    except PermissionError as exc:
        st.error(str(exc))
        return

    if vehiculo is None:
        st.info(
            f"La matrícula {matricula} no está registrada. Elige el tipo de vehículo y sube un "
            f"CSV con al menos {N_REF_CYCLES} ciclos para registrarla."
        )
        tipo = st.selectbox("Tipo de vehículo", list(CARS))
        mostrar_coche(tipo)
    else:
        tipo = vehiculo["tipo"] if vehiculo["tipo"] in CARS else "Urbano"
        st.caption(f"Matrícula {matricula} · {tipo}")
        mostrar_coche(tipo)
        mostrar_historial(conn, matricula, tipo)
    subir_lecturas(conn, usuario, matricula, tipo)


conexion = db.conectar()
try:
    vista(conexion)
finally:
    conexion.close()