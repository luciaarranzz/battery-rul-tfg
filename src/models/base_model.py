"""Interfaz común para todos los modelos del pipeline (patrón Strategy).

Cualquier modelo nuevo que se añada al TFG (por ejemplo, un SVR o un
Gaussian Process) solo necesita heredar de BaseModel e implementar fit() y
predict(); train.py y evaluate.py no necesitan cambiar."""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


class BaseModel(ABC):
    name: str = "base_model"

    @abstractmethod
    def fit(self, X_train, y_train) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X_test) -> np.ndarray:
        ...

    def evaluate(self, y_true, y_pred, mape_min_true: float = 1.0) -> dict:
        """Métricas estándar del TFG: RMSE, MAE, MAPE y R².

        Nota sobre MAPE: cuando el objetivo es RUL, los valores reales se
        acercan a 0 al final de la vida de la batería, y el MAPE (que divide
        por y_true) se dispara o queda indefinido ahí. Por eso el MAPE se
        calcula solo sobre las muestras con |y_true| >= `mape_min_true`;
        para SOH (que nunca llega a 0 en este umbral de EOL) esto no
        excluye prácticamente ninguna muestra. Documéntalo así en la
        memoria si usas MAPE con RUL."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        mask = np.abs(y_true) >= mape_min_true
        if mask.any():
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        else:
            mape = float("nan")

        return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}