"""Estimación de la vida útil restante (RUL) a partir del SOH predicho.

Método indirecto: se ajusta una recta a los últimos ciclos del SOH predicho
y se calcula cuántos ciclos faltan para que llegue al umbral de fin de vida.
Es una aproximación lineal: no anticipa codos en la curva de degradación.
Solo se estima si la pendiente es claramente distinta de cero frente al ruido
de los datos; si no, el resultado dependería del azar más que de la degradación.
"""
from dataclasses import dataclass

import numpy as np

EOL_SOH = 80.0  # % de SOH que marca el fin de vida
VENTANA = 60  # últimos ciclos que se usan para calcular la tendencia
MIN_PUNTOS = 10  # mínimo de ciclos para fiarse de la tendencia
MIN_PENDIENTE = 0.001  # puntos de SOH por ciclo; por debajo no hay degradación apreciable
N_ERRORES = 2  # la pendiente tiene que superar N_ERRORES veces su error típico para fiarse de ella


@dataclass
class EstimacionRUL:
    """Resultado de la estimación; ``ciclos_restantes`` es None si no se puede estimar."""

    ciclos_restantes: float | None
    pendiente: float | None  # puntos de SOH por ciclo (negativa si la batería se degrada)
    mensaje: str


def estimar_rul(ciclos, soh, umbral: float = EOL_SOH, ventana: int = VENTANA) -> EstimacionRUL:
    """Estima cuántos ciclos faltan para que el SOH llegue al umbral."""
    ciclos = np.asarray(ciclos, dtype=float)
    soh = np.asarray(soh, dtype=float)
    orden = np.argsort(ciclos)
    ciclos, soh = ciclos[orden][-ventana:], soh[orden][-ventana:]

    if len(ciclos) < MIN_PUNTOS:
        return EstimacionRUL(
            None,
            None,
            f"Hacen falta al menos {MIN_PUNTOS} ciclos para estimar la tendencia "
            f"y hay {len(ciclos)}.",
        )

    pendiente, ordenada = np.polyfit(ciclos, soh, 1)
    ajustado = pendiente * ciclos + ordenada
    soh_actual = ajustado[-1]  # sobre la recta, menos ruidoso que el último dato
    if soh_actual <= umbral:
        return EstimacionRUL(0.0, float(pendiente), "La batería ya ha alcanzado el fin de vida.")

    # Error típico de la pendiente: cuánto puede variar solo por el ruido de los datos
    varianza_residuos = np.sum((soh - ajustado) ** 2) / (len(ciclos) - 2)
    error_pendiente = np.sqrt(varianza_residuos / np.sum((ciclos - ciclos.mean()) ** 2))
    if pendiente > -MIN_PENDIENTE or abs(pendiente) < N_ERRORES * error_pendiente:
        return EstimacionRUL(
            None,
            float(pendiente),
            "No se observa una tendencia de degradación clara en los últimos ciclos, "
            "así que no se puede estimar la RUL.",
        )

    restantes = (soh_actual - umbral) / -pendiente
    return EstimacionRUL(
        float(restantes),
        float(pendiente),
        f"Se estima que faltan unos {restantes:.0f} ciclos para llegar al {umbral:.0f} % de SOH.",
    )