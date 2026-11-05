"""Pruebas de la creación de los datos de demostración."""
import numpy as np
import pandas as pd
import pytest

from src import db, seed_demo


@pytest.fixture(autouse=True)
def hash_rapido(monkeypatch):
    monkeypatch.setattr(db, "ITERACIONES", 1000)


@pytest.fixture
def conn():
    conexion = db.conectar(":memory:")
    yield conexion
    conexion.close()


def _escribir_csv(ruta, n_ciclos=5, voltaje_inicial=4.2):
    ciclos = [
        pd.DataFrame(
            {
                "ciclo": n,
                "tiempo_s": np.arange(50) * 20.0,
                "voltaje_v": np.linspace(voltaje_inicial, 2.7, 50),
                "temperatura_c": np.linspace(25.0, 35.0, 50),
                "tiempo_carga_s": 3600.0,
                "temp_ambiente_c": 24.0,
            }
        )
        for n in range(1, n_ciclos + 1)
    ]
    pd.concat(ciclos).to_csv(ruta, index=False)
    return ruta


def test_sembrar_crea_las_cuentas_el_vehiculo_y_su_historial(conn, tmp_path):
    csv = _escribir_csv(tmp_path / "historial.csv")
    assert seed_demo.sembrar(conn, csv) is True

    nombre, contrasena, _ = seed_demo.USUARIO_DEMO
    assert db.verificar_usuario(conn, nombre, contrasena)["rol"] == "usuario"
    nombre, contrasena, _ = seed_demo.ADMIN_DEMO
    assert db.verificar_usuario(conn, nombre, contrasena)["rol"] == "administrador"

    vehiculo = db.obtener_vehiculo(conn, seed_demo.MATRICULA_CON_HISTORIAL)
    assert vehiculo is not None
    assert len(db.cargar_ciclos(conn, seed_demo.MATRICULA_CON_HISTORIAL)) == 5


def test_sembrar_dos_veces_no_duplica_nada(conn, tmp_path):
    csv = _escribir_csv(tmp_path / "historial.csv")
    assert seed_demo.sembrar(conn, csv) is True
    assert seed_demo.sembrar(conn, csv) is False
    assert conn.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0] == 2


def test_si_el_csv_no_es_valido_no_se_toca_la_base(conn, tmp_path):
    csv = _escribir_csv(tmp_path / "malo.csv", voltaje_inicial=9.0)  # todos los ciclos se descartan
    with pytest.raises(ValueError, match="no es válido"):
        seed_demo.sembrar(conn, csv)
    assert conn.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0] == 0
    