"""Pruebas de la base de datos de la app."""
import sqlite3

import pandas as pd
import pytest

from src import db
from src.features import FEATURE_COLUMNS


@pytest.fixture(autouse=True)
def hash_rapido(monkeypatch):
    monkeypatch.setattr(db, "ITERACIONES", 1000)


@pytest.fixture
def conn():
    conexion = db.conectar(":memory:")
    yield conexion
    conexion.close()


def _indicadores(ciclos):
    ciclos = list(ciclos)
    return pd.DataFrame({"ciclo": ciclos, **{c: [float(i) for i in ciclos] for c in FEATURE_COLUMNS}})


def test_usuario_con_credenciales_correctas_entra(conn):
    db.crear_usuario(conn, "ana", "clave123", "usuario")
    usuario = db.verificar_usuario(conn, "ana", "clave123")
    assert usuario["nombre"] == "ana"
    assert usuario["rol"] == "usuario"


def test_contrasena_incorrecta_o_usuario_inexistente_no_entran(conn):
    db.crear_usuario(conn, "ana", "clave123")
    assert db.verificar_usuario(conn, "ana", "otra") is None
    assert db.verificar_usuario(conn, "pepe", "clave123") is None


def test_la_contrasena_no_se_guarda_en_claro(conn):
    db.crear_usuario(conn, "ana", "clave123")
    guardada = conn.execute("SELECT contrasena FROM usuarios").fetchone()[0]
    assert "clave123" not in guardada
    assert guardada.startswith("pbkdf2_sha256$")


def test_dos_usuarios_con_la_misma_contrasena_tienen_hashes_distintos(conn):
    db.crear_usuario(conn, "ana", "clave123")
    db.crear_usuario(conn, "luis", "clave123")
    hashes = [fila[0] for fila in conn.execute("SELECT contrasena FROM usuarios")]
    assert hashes[0] != hashes[1]


def test_usuario_repetido_o_rol_invalido_dan_error(conn):
    db.crear_usuario(conn, "ana", "clave123")
    with pytest.raises(ValueError, match="ya existe"):
        db.crear_usuario(conn, "ana", "otra")
    with pytest.raises(ValueError, match="rol"):
        db.crear_usuario(conn, "luis", "clave123", "jefe")


def test_un_intento_de_inyeccion_sql_no_entra(conn):
    db.crear_usuario(conn, "ana", "clave123")
    assert db.verificar_usuario(conn, "x' OR '1'='1", "lo que sea") is None


def test_la_matricula_se_normaliza():
    assert db.normalizar_matricula(" 1234-abc ") == "1234ABC"
    assert db.normalizar_matricula("1234 ABC") == "1234ABC"
    with pytest.raises(ValueError):
        db.normalizar_matricula("ab")


def test_vehiculo_se_encuentra_con_cualquier_escritura_de_la_matricula(conn):
    usuario_id = db.crear_usuario(conn, "ana", "clave123")
    db.crear_vehiculo(conn, "1234 abc", usuario_id, "Utilitario")
    assert db.obtener_vehiculo(conn, "1234ABC")["tipo"] == "Utilitario"
    assert db.obtener_vehiculo(conn, "9999ZZZ") is None
    with pytest.raises(ValueError, match="ya está registrada"):
        db.crear_vehiculo(conn, "1234-ABC", usuario_id, "Utilitario")


def test_cada_usuario_ve_sus_vehiculos_y_el_administrador_todos(conn):
    ana = db.crear_usuario(conn, "ana", "clave123")
    luis = db.crear_usuario(conn, "luis", "clave123")
    db.crear_vehiculo(conn, "1111AAA", ana, "Utilitario")
    db.crear_vehiculo(conn, "2222BBB", luis, "Furgoneta")
    assert [v["matricula"] for v in db.listar_vehiculos(conn, ana)] == ["1111AAA"]
    assert [v["matricula"] for v in db.listar_vehiculos(conn)] == ["1111AAA", "2222BBB"]


def test_los_ciclos_guardados_se_recuperan_ordenados(conn):
    usuario_id = db.crear_usuario(conn, "ana", "clave123")
    db.crear_vehiculo(conn, "1234ABC", usuario_id, "Utilitario")
    assert db.guardar_ciclos(conn, "1234abc", _indicadores([3, 1, 2])) == 3
    cargados = db.cargar_ciclos(conn, "1234ABC")
    assert list(cargados["ciclo"]) == [1, 2, 3]
    assert list(cargados.columns) == ["ciclo", *FEATURE_COLUMNS]
    assert cargados["t_mean"].tolist() == [1.0, 2.0, 3.0]


def test_guardar_otra_vez_los_mismos_ciclos_no_los_duplica(conn):
    usuario_id = db.crear_usuario(conn, "ana", "clave123")
    db.crear_vehiculo(conn, "1234ABC", usuario_id, "Utilitario")
    db.guardar_ciclos(conn, "1234ABC", _indicadores(range(1, 6)))
    db.guardar_ciclos(conn, "1234ABC", _indicadores(range(4, 9)))
    assert list(db.cargar_ciclos(conn, "1234ABC")["ciclo"]) == list(range(1, 9))


def test_vehiculo_sin_datos_devuelve_tabla_vacia_con_las_columnas(conn):
    usuario_id = db.crear_usuario(conn, "ana", "clave123")
    db.crear_vehiculo(conn, "1234ABC", usuario_id, "Utilitario")
    cargados = db.cargar_ciclos(conn, "1234ABC")
    assert cargados.empty
    assert list(cargados.columns) == ["ciclo", *FEATURE_COLUMNS]


def test_no_se_pueden_guardar_ciclos_de_un_vehiculo_que_no_existe(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.guardar_ciclos(conn, "0000XXX", _indicadores([1]))