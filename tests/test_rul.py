"""Pruebas de la estimación de la RUL."""
import numpy as np
import pytest

from src.rul import MIN_PUNTOS, estimar_rul


def test_degradacion_lineal_da_los_ciclos_exactos():
    ciclos = np.arange(1, 51)
    soh = 100 - 0.2 * ciclos  # en el ciclo 50 vale 90, y baja 0,2 por ciclo
    resultado = estimar_rul(ciclos, soh)
    assert resultado.ciclos_restantes == pytest.approx(50.0)
    assert resultado.pendiente == pytest.approx(-0.2)


def test_si_ya_esta_por_debajo_del_umbral_la_rul_es_cero():
    ciclos = np.arange(1, 51)
    resultado = estimar_rul(ciclos, 85 - 0.5 * ciclos)
    assert resultado.ciclos_restantes == 0.0
    assert "fin de vida" in resultado.mensaje


def test_sin_degradacion_no_se_estima():
    ciclos = np.arange(1, 51)
    resultado = estimar_rul(ciclos, np.full(50, 95.0))
    assert resultado.ciclos_restantes is None
    assert "tendencia de degradación clara" in resultado.mensaje


def test_con_pocos_ciclos_no_se_estima():
    ciclos = np.arange(1, MIN_PUNTOS)
    resultado = estimar_rul(ciclos, 100 - 0.2 * ciclos)
    assert resultado.ciclos_restantes is None
    assert resultado.pendiente is None


def test_solo_cuenta_la_ventana_final():
    ciclos = np.arange(1, 71)
    soh = np.where(ciclos <= 40, 100.0, 100.0 - 0.3 * (ciclos - 40))  # estable y luego baja
    resultado = estimar_rul(ciclos, soh, ventana=30)
    assert resultado.pendiente == pytest.approx(-0.3)
    assert resultado.ciclos_restantes == pytest.approx(11.0 / 0.3)  # en el ciclo 70 vale 91


def test_el_orden_de_los_datos_no_importa():
    ciclos = np.arange(1, 51)
    soh = 100 - 0.2 * ciclos
    desordenados = np.random.default_rng(0).permutation(50)
    resultado = estimar_rul(ciclos[desordenados], soh[desordenados])
    assert resultado.ciclos_restantes == pytest.approx(50.0)


def test_si_la_pendiente_se_pierde_en_el_ruido_no_se_estima():
    ciclos = np.arange(1, 61)
    ruido = np.random.default_rng(1).normal(0, 1.5, 60)
    resultado = estimar_rul(ciclos, 98.5 - 0.01 * ciclos + ruido)  # casi plano y con mucho ruido
    assert resultado.ciclos_restantes is None
    assert "tendencia de degradación clara" in resultado.mensaje


def test_una_degradacion_clara_se_estima_aunque_haya_ruido():
    ciclos = np.arange(1, 61)
    ruido = np.random.default_rng(2).normal(0, 0.5, 60)
    resultado = estimar_rul(ciclos, 100 - 0.3 * ciclos + ruido)  # en el ciclo 60 vale unos 82
    assert resultado.ciclos_restantes is not None
    assert resultado.ciclos_restantes == pytest.approx(7.0, abs=5.0)