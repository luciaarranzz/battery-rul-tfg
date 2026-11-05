"""Base de datos SQLite de la app: usuarios, vehículos e indicadores por ciclo.

De cada ciclo se guardan solo los indicadores de salud (los que calcula
``extract_health_indicators``), no las muestras de la descarga. El SOH y la
RUL no se guardan: se calculan al consultar, con el historial completo.
"""
import hashlib
import hmac
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS

DB_PATH = Path("data/app.db")
ROLES = ("usuario", "administrador")
ITERACIONES = 600_000  # iteraciones de PBKDF2; las pruebas lo bajan para ir más rápido

_COLUMNAS_SQL = ", ".join(FEATURE_COLUMNS)
_MARCAS_SQL = ", ".join("?" * (len(FEATURE_COLUMNS) + 2))
_ESQUEMA = f"""
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL UNIQUE,
    contrasena TEXT NOT NULL,
    rol TEXT NOT NULL CHECK (rol IN ('usuario', 'administrador'))
);
CREATE TABLE IF NOT EXISTS vehiculos (
    matricula TEXT PRIMARY KEY,
    usuario_id INTEGER NOT NULL REFERENCES usuarios (id),
    tipo TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ciclos (
    matricula TEXT NOT NULL REFERENCES vehiculos (matricula),
    ciclo INTEGER NOT NULL,
    {", ".join(f"{c} REAL NOT NULL" for c in FEATURE_COLUMNS)},
    PRIMARY KEY (matricula, ciclo)
);
"""


def conectar(ruta=DB_PATH) -> sqlite3.Connection:
    """Abre la base de datos (la crea si no existe) con las tablas listas."""
    if str(ruta) != ":memory:":
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ruta)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_ESQUEMA)
    return conn


# --- Contraseñas: nunca se guardan en claro, solo un hash con sal ---------


def _hash(contrasena: str, sal: bytes, iteraciones: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", contrasena.encode(), sal, iteraciones).hex()


def hashear_contrasena(contrasena: str) -> str:
    """Devuelve ``pbkdf2_sha256$iteraciones$sal$hash`` con una sal aleatoria."""
    sal = os.urandom(16)
    return f"pbkdf2_sha256${ITERACIONES}${sal.hex()}${_hash(contrasena, sal, ITERACIONES)}"


def contrasena_correcta(contrasena: str, guardada: str) -> bool:
    """Comprueba una contraseña contra el texto guardado por ``hashear_contrasena``."""
    try:
        _, iteraciones, sal, esperado = guardada.split("$")
        calculado = _hash(contrasena, bytes.fromhex(sal), int(iteraciones))
    except ValueError:
        return False
    return hmac.compare_digest(calculado, esperado)


# --- Usuarios ---------------------------------------------------------------


def crear_usuario(conn, nombre: str, contrasena: str, rol: str = "usuario") -> int:
    """Crea un usuario y devuelve su id."""
    if rol not in ROLES:
        raise ValueError(f"El rol debe ser uno de {ROLES}.")
    if not nombre.strip() or not contrasena:
        raise ValueError("El nombre y la contraseña no pueden estar vacíos.")
    try:
        with conn:
            cursor = conn.execute(
                "INSERT INTO usuarios (nombre, contrasena, rol) VALUES (?, ?, ?)",
                (nombre.strip(), hashear_contrasena(contrasena), rol),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"El usuario '{nombre.strip()}' ya existe.") from exc
    return cursor.lastrowid


def verificar_usuario(conn, nombre: str, contrasena: str) -> dict | None:
    """Devuelve ``id``, ``nombre`` y ``rol`` si las credenciales son correctas, y None si no."""
    fila = conn.execute(
        "SELECT id, nombre, contrasena, rol FROM usuarios WHERE nombre = ?", (nombre.strip(),)
    ).fetchone()
    if fila is None or not contrasena_correcta(contrasena, fila["contrasena"]):
        return None
    return {"id": fila["id"], "nombre": fila["nombre"], "rol": fila["rol"]}


# --- Vehículos --------------------------------------------------------------


def normalizar_matricula(matricula: str) -> str:
    """Pasa la matrícula a mayúsculas y sin espacios ni guiones (``1234 abc`` -> ``1234ABC``)."""
    limpia = re.sub(r"[\s-]", "", matricula).upper()
    if not re.fullmatch(r"[A-Z0-9]{4,10}", limpia):
        raise ValueError("La matrícula debe tener entre 4 y 10 letras o números.")
    return limpia


def obtener_vehiculo(conn, matricula: str) -> dict | None:
    """Devuelve los datos del vehículo, o None si no está registrado."""
    fila = conn.execute(
        "SELECT matricula, usuario_id, tipo FROM vehiculos WHERE matricula = ?",
        (normalizar_matricula(matricula),),
    ).fetchone()
    return dict(fila) if fila else None


def crear_vehiculo(conn, matricula: str, usuario_id: int, tipo: str) -> str:
    """Registra un vehículo de un usuario y devuelve su matrícula normalizada."""
    if obtener_vehiculo(conn, matricula) is not None:
        raise ValueError(f"La matrícula {normalizar_matricula(matricula)} ya está registrada.")
    matricula = normalizar_matricula(matricula)
    with conn:
        conn.execute(
            "INSERT INTO vehiculos (matricula, usuario_id, tipo) VALUES (?, ?, ?)",
            (matricula, usuario_id, tipo),
        )
    return matricula


def listar_vehiculos(conn, usuario_id: int | None = None) -> list[dict]:
    """Devuelve los vehículos de un usuario, o todos si no se indica usuario."""
    if usuario_id is None:
        filas = conn.execute(
            "SELECT matricula, usuario_id, tipo FROM vehiculos ORDER BY matricula"
        ).fetchall()
    else:
        filas = conn.execute(
            "SELECT matricula, usuario_id, tipo FROM vehiculos WHERE usuario_id = ? ORDER BY matricula",
            (usuario_id,),
        ).fetchall()
    return [dict(fila) for fila in filas]


# --- Ciclos -----------------------------------------------------------------


def guardar_ciclos(conn, matricula: str, indicadores: pd.DataFrame) -> int:
    """Guarda los indicadores de ciclos de un vehículo (si el ciclo ya existe, lo sustituye).

    ``indicadores`` tiene la columna ``ciclo`` y las de ``FEATURE_COLUMNS``,
    como las que devuelve ``indicadores_por_ciclo``. Devuelve cuántos ciclos guarda.
    """
    matricula = normalizar_matricula(matricula)
    filas = [
        (matricula, int(fila["ciclo"]), *(float(fila[c]) for c in FEATURE_COLUMNS))
        for _, fila in indicadores.iterrows()
    ]
    with conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO ciclos (matricula, ciclo, {_COLUMNAS_SQL}) VALUES ({_MARCAS_SQL})",
            filas,
        )
    return len(filas)


def cargar_ciclos(conn, matricula: str) -> pd.DataFrame:
    """Devuelve el historial de indicadores del vehículo, ordenado por ciclo."""
    return pd.read_sql_query(
        f"SELECT ciclo, {_COLUMNAS_SQL} FROM ciclos WHERE matricula = ? ORDER BY ciclo",
        conn,
        params=(normalizar_matricula(matricula),),
    )