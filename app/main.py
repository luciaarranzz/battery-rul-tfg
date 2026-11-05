"""Punto de entrada de la app: inicio de sesión y navegación según el rol.

Uso:  streamlit run app/main.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # para poder importar src

from src import db, seed_demo  # noqa: E402

st.set_page_config(page_title="Monitorización de baterías", layout="wide")


def pagina_login():
    st.title("Monitorización de la batería de tu vehículo")
    st.caption("Aplicación de demostración del TFG.")
    with st.form("login"):
        nombre = st.text_input("Usuario")
        contrasena = st.text_input("Contraseña", type="password")
        enviado = st.form_submit_button("Entrar")

    if enviado:
        conn = db.conectar()
        try:
            usuario = db.verificar_usuario(conn, nombre, contrasena)
        finally:
            conn.close()
        if usuario is None:
            st.error("Usuario o contraseña incorrectos.")
        else:
            st.session_state["usuario"] = usuario
            st.rerun()

    with st.expander("Cuentas de demostración"):
        for nombre, contrasena, rol in (seed_demo.USUARIO_DEMO, seed_demo.ADMIN_DEMO):
            st.markdown(f"**{rol.capitalize()}:** `{nombre}` / `{contrasena}`")
        st.markdown(f"Matrícula de ejemplo con historial: `{seed_demo.MATRICULA_CON_HISTORIAL}`")


def main():
    conn = db.conectar()
    try:
        seed_demo.sembrar(conn)  # crea las cuentas de demostración si la base está vacía
    finally:
        conn.close()

    usuario = st.session_state.get("usuario")
    if usuario is None:
        navegacion = st.navigation(
            [st.Page(pagina_login, title="Iniciar sesión", icon=":material/login:")]
        )
    else:
        if usuario["rol"] == "administrador":
            paginas = [
                st.Page("dashboard.py", title="Panel de administración", icon=":material/dashboard:")
            ]
        else:
            paginas = [
                st.Page("vista_usuario.py", title="Mi vehículo", icon=":material/directions_car:")
            ]
        navegacion = st.navigation(paginas)
        with st.sidebar:
            st.write(f"Sesión iniciada como **{usuario['nombre']}** ({usuario['rol']})")
            if st.button("Cerrar sesión"):
                del st.session_state["usuario"]
                st.rerun()
    navegacion.run()


main()