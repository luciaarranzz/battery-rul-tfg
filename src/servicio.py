"""Lógica de la vista de usuario: acceso a vehículos, guardado de lecturas e historial con SOH.

Está separada de la interfaz para poder probarla sin Streamlit.
"""
import pandas as pd

from src import db
from src.features import N_REF_CYCLES
from src.inference import indicadores_por_ciclo, predecir_soh


def vehiculo_accesible(conn, usuario: dict, matricula: str) -> dict | None:
    """Devuelve el vehículo si existe y el usuario puede verlo, o None si la matrícula es nueva.

    Lanza PermissionError si es de otro usuario (el administrador puede ver todos).
    """
    vehiculo = db.obtener_vehiculo(conn, matricula)
    if (
        vehiculo is not None
        and usuario["rol"] != "administrador"
        and vehiculo["usuario_id"] != usuario["id"]
    ):
        raise PermissionError("Esta matrícula ya está registrada por otro usuario.")
    return vehiculo


def ciclos_repetidos(conn, matricula: str, ciclos: pd.DataFrame) -> list[int]:
    """Números de ciclo del CSV que ya están guardados (al guardar se sustituyen)."""
    guardados = set(db.cargar_ciclos(conn, matricula)["ciclo"])
    return sorted(guardados.intersection(int(c) for c in ciclos["ciclo"]))


def guardar_lecturas(conn, usuario: dict, matricula: str, tipo: str, ciclos: pd.DataFrame) -> int:
    """Guarda los ciclos de un CSV en el vehículo; si es nuevo, lo registra a nombre del usuario.

    ``ciclos`` es el resultado de ``procesar_csv``. Devuelve cuántos ciclos guarda.
    """
    vehiculo = vehiculo_accesible(conn, usuario, matricula)
    if vehiculo is None:
        if len(ciclos) < N_REF_CYCLES:
            raise ValueError(
                f"Un vehículo nuevo necesita al menos {N_REF_CYCLES} ciclos para fijar su "
                f"referencia y el CSV tiene {len(ciclos)}."
            )
        db.crear_vehiculo(conn, matricula, usuario["id"], tipo)
    return db.guardar_ciclos(conn, matricula, indicadores_por_ciclo(ciclos))


def historial_con_soh(conn, matricula: str, modelo, columnas) -> pd.DataFrame:
    """Historial del vehículo con el SOH predicho de cada ciclo (vacío si faltan ciclos)."""
    indicadores = db.cargar_ciclos(conn, matricula)
    if len(indicadores) < N_REF_CYCLES:
        return pd.DataFrame(columns=[*indicadores.columns, "soh_predicho"])
    return indicadores.merge(predecir_soh(indicadores, modelo, columnas), on="ciclo")


def estado_bateria(soh: float) -> str:
    """Estado según el SOH: fin de vida por debajo del 80 %, aviso por debajo del 90 %."""
    if soh < 80:
        return "Fin de vida"
    if soh < 90:
        return "Aviso"
    return "OK"