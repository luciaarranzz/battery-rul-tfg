"""Crea las cuentas y el vehículo de demostración de la app.

Son datos ficticios para que cualquiera pueda probar la app sin preparar
nada: un usuario, un administrador y un vehículo con 60 ciclos de historial
(los de ``demo_vehiculo_nuevo.csv``, de una batería de test). Solo se crean
si la base de datos está vacía; para volver al estado inicial basta con
borrar ``data/app.db``.

Uso:  python -m src.seed_demo
"""
from pathlib import Path

from src import db
from src.data.bms_csv import procesar_csv
from src.inference import indicadores_por_ciclo
from src.utils import get_logger

log = get_logger(__name__)

# (nombre, contraseña, rol). Cuentas de demostración, documentadas en la memoria.
USUARIO_DEMO = ("usuario_demo", "demo-usuario-2026", "usuario")
ADMIN_DEMO = ("admin_demo", "demo-admin-2026", "administrador")
MATRICULA_CON_HISTORIAL = "1234BCD"
TIPO_VEHICULO = "Urbano"
CSV_HISTORIAL = Path("data/demo/demo_vehiculo_nuevo.csv")


def sembrar(conn, csv_historial: Path = CSV_HISTORIAL) -> bool:
    """Crea los datos de demostración si no hay ningún usuario. Devuelve si los ha creado."""
    if conn.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0] > 0:
        return False

    # Se lee el CSV antes de tocar la base, para no dejarla a medias si falla
    resultado = procesar_csv(csv_historial)
    if resultado.errores:
        raise ValueError(f"El CSV de historial no es válido: {'; '.join(resultado.errores)}")

    usuario_id = db.crear_usuario(conn, *USUARIO_DEMO)
    db.crear_usuario(conn, *ADMIN_DEMO)
    db.crear_vehiculo(conn, MATRICULA_CON_HISTORIAL, usuario_id, TIPO_VEHICULO)
    n_ciclos = db.guardar_ciclos(conn, MATRICULA_CON_HISTORIAL, indicadores_por_ciclo(resultado.ciclos))
    log.info(
        "Datos de demostración creados: 2 usuarios y el vehículo %s con %d ciclos.",
        MATRICULA_CON_HISTORIAL,
        n_ciclos,
    )
    return True


def main():
    conn = db.conectar()
    try:
        if not sembrar(conn):
            log.info("La base de datos ya tiene usuarios; no se ha cambiado nada.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()