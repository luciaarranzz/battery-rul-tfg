"""Pruebas de la lógica de la vista de usuario."""
import numpy as np
import pandas as pd
import pytest

from src import db, servicio


class ModeloFalso:
    def predict(self, X):
        return (X["discharge_duration_rel"] * 100).to_numpy()


COLUMNAS = ["discharge_duration_rel", "t_mean_delta", "discharge_duration"]


@pytest.fixture(autouse=True)
def hash_rapido(monkeypatch):
    monkeypatch.setattr(db, "ITERACIONES", 1000)


@pytest.fixture
def conn():
    conexion = db.conectar(":memory:")
    yield conexion
    conexion.close()


@pytest.fixture
def usuarios(conn):
    ana = {"id": db.crear_usuario(conn, "ana", "clave123"), "nombre": "ana", "rol": "usuario"}
    luis = {"id": db.crear_usuario(conn, "luis", "clave123"), "nombre": "luis", "rol": "usuario"}
    admin = {
        "id": db.crear_usuario(conn, "admin", "clave123", "administrador"),
        "nombre": "admin",
        "rol": "administrador",
    }
    return ana, luis, admin


def _ciclos(primero, ultimo):
    """Ciclos con descargas cada vez más cortas, en el formato de ``procesar_csv``."""
    filas = []
    for ciclo in range(primero, ultimo + 1):
        muestras = int(100 * 0.98**ciclo) + 10
        filas.append(
            {
                "ciclo": ciclo,
                "time_arr": np.arange(muestras) * 20.0,
                "voltage_arr": np.linspace(4.2, 2.7, muestras),
                "temp_arr": np.linspace(25.0, 35.0, muestras),
                "charge_time": 3600.0,
                "ambient_temp": 24.0,
            }
        )
    return pd.DataFrame(filas)


def test_vehiculo_nuevo_se_registra_a_nombre_del_usuario(conn, usuarios):
    ana, _, _ = usuarios
    assert servicio.vehiculo_accesible(conn, ana, "1234ABC") is None
    assert servicio.guardar_lecturas(conn, ana, "1234abc", "Urbano", _ciclos(1, 5)) == 5
    vehiculo = db.obtener_vehiculo(conn, "1234ABC")
    assert vehiculo["usuario_id"] == ana["id"]
    assert vehiculo["tipo"] == "Urbano"


def test_vehiculo_nuevo_con_pocos_ciclos_no_se_registra(conn, usuarios):
    ana, _, _ = usuarios
    with pytest.raises(ValueError, match="al menos"):
        servicio.guardar_lecturas(conn, ana, "1234ABC", "Urbano", _ciclos(1, 2))
    assert db.obtener_vehiculo(conn, "1234ABC") is None


def test_otro_usuario_no_puede_ver_ni_guardar_pero_el_administrador_si(conn, usuarios):
    ana, luis, admin = usuarios
    servicio.guardar_lecturas(conn, ana, "1234ABC", "Urbano", _ciclos(1, 5))
    with pytest.raises(PermissionError):
        servicio.vehiculo_accesible(conn, luis, "1234ABC")
    with pytest.raises(PermissionError):
        servicio.guardar_lecturas(conn, luis, "1234ABC", "Urbano", _ciclos(6, 8))
    assert servicio.vehiculo_accesible(conn, admin, "1234ABC")["usuario_id"] == ana["id"]


def test_las_lecturas_nuevas_se_anaden_al_historial(conn, usuarios):
    ana, _, _ = usuarios
    servicio.guardar_lecturas(conn, ana, "1234ABC", "Urbano", _ciclos(1, 5))
    servicio.guardar_lecturas(conn, ana, "1234ABC", "Urbano", _ciclos(6, 8))
    assert list(db.cargar_ciclos(conn, "1234ABC")["ciclo"]) == list(range(1, 9))


def test_ciclos_repetidos_detecta_los_que_ya_estaban(conn, usuarios):
    ana, _, _ = usuarios
    servicio.guardar_lecturas(conn, ana, "1234ABC", "Urbano", _ciclos(1, 5))
    assert servicio.ciclos_repetidos(conn, "1234ABC", _ciclos(4, 8)) == [4, 5]
    assert servicio.ciclos_repetidos(conn, "1234ABC", _ciclos(6, 8)) == []
    assert servicio.ciclos_repetidos(conn, "9999ZZZ", _ciclos(1, 3)) == []


def test_historial_con_soh_tiene_un_soh_por_ciclo(conn, usuarios):
    ana, _, _ = usuarios
    servicio.guardar_lecturas(conn, ana, "1234ABC", "Urbano", _ciclos(1, 8))
    historial = servicio.historial_con_soh(conn, "1234ABC", ModeloFalso(), COLUMNAS)
    assert list(historial["ciclo"]) == list(range(1, 9))
    assert "soh_predicho" in historial.columns
    assert historial["soh_predicho"].is_monotonic_decreasing


def test_historial_de_un_vehiculo_sin_ciclos_esta_vacio(conn, usuarios):
    ana, _, _ = usuarios
    db.crear_vehiculo(conn, "1234ABC", ana["id"], "Urbano")
    historial = servicio.historial_con_soh(conn, "1234ABC", ModeloFalso(), COLUMNAS)
    assert historial.empty
    assert "soh_predicho" in historial.columns


@pytest.mark.parametrize(
    ("soh", "estado"),
    [(95.0, "OK"), (90.0, "OK"), (89.9, "Aviso"), (80.0, "Aviso"), (79.9, "Fin de vida")],
)
def test_estado_bateria(soh, estado):
    assert servicio.estado_bateria(soh) == estado