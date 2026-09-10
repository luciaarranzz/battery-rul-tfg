"""Modelo secuencial (LSTM) sobre secuencias de vectores de features por
batería. Es la ampliación opcional del TFG una vez los modelos clásicos
(src/models/classical_models.py) funcionan bien.

PyTorch es una dependencia opcional del proyecto: si no está instalado, se
lanza un error solo al intentar USAR este modelo (no al importar el
paquete), para que el resto del pipeline funcione sin torch instalado.
"""
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - se comprueba en tiempo de ejecución
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "El modelo LSTM necesita PyTorch. Instálalo con: pip install torch "
            "(o quita 'lstm' de la lista de modelos en src/train.py)."
        )


def _build_sequences(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seq_len: int):
    """A partir de features tabulares por ciclo, construye secuencias de
    longitud `seq_len` por batería (ventana deslizante), para que el LSTM
    vea la evolución temporal, no solo el ciclo actual."""
    X_seqs, y_vals = [], []
    for battery_id in groups.unique():
        mask = (groups == battery_id).to_numpy()
        Xb = X.to_numpy()[mask]
        yb = y.to_numpy()[mask]
        for i in range(seq_len, len(Xb) + 1):
            X_seqs.append(Xb[i - seq_len:i])
            y_vals.append(yb[i - 1])
    return np.array(X_seqs, dtype=np.float32), np.array(y_vals, dtype=np.float32)


def align_targets(y: pd.Series, groups: pd.Series, seq_len: int) -> np.ndarray:
    """Devuelve el vector `y` recortado a las mismas posiciones que generan
    las secuencias de `_build_sequences`, para poder evaluar las
    predicciones del LSTM contra el valor real correcto de cada secuencia."""
    y_vals = []
    y_arr = pd.Series(y).to_numpy()
    for battery_id in groups.unique():
        mask = (groups == battery_id).to_numpy()
        yb = y_arr[mask]
        for i in range(seq_len, len(yb) + 1):
            y_vals.append(yb[i - 1])
    return np.array(y_vals, dtype=np.float32)


class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, hidden_size: int = 32, n_layers: int = 1, sequence_length: int = 10,
                 epochs: int = 15, lr: float = 1e-3):
        _require_torch()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.net = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _normalize_x(self, X, fit=False):
        if fit:
            self._x_mean = X.mean(axis=(0, 1), keepdims=True)
            self._x_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - self._x_mean) / self._x_std

    def fit(self, X_train, y_train, groups=None):
        if groups is None:
            raise ValueError("LSTMModel.fit necesita `groups` (battery_id por fila).")
        X_seq, y_seq = _build_sequences(X_train, y_train, groups, self.sequence_length)
        X_seq = self._normalize_x(X_seq, fit=True)

        # Normalizar también el objetivo ayuda mucho a que Adam converja en
        # pocas épocas cuando y está en una escala grande (p. ej. SOH en %).
        self._y_mean = float(y_seq.mean())
        self._y_std = float(y_seq.std() + 1e-8)
        y_seq_norm = (y_seq - self._y_mean) / self._y_std

        self.net = _LSTMNet(X_train.shape[1], self.hidden_size, self.n_layers)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(X_seq)
        y_t = torch.from_numpy(y_seq_norm)
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.net(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X_test, groups=None):
        if groups is None:
            raise ValueError("LSTMModel.predict necesita `groups` (battery_id por fila).")
        X_seq, _ = _build_sequences(X_test, pd.Series(np.zeros(len(X_test))), groups, self.sequence_length)
        X_seq = self._normalize_x(X_seq, fit=False)
        self.net.eval()
        with torch.no_grad():
            preds_norm = self.net(torch.from_numpy(X_seq)).numpy()
        return preds_norm * self._y_std + self._y_mean

    def predict_with_uncertainty(self, X_test, groups=None, n_samples: int = 30):
        """Estimación sencilla de incertidumbre vía Monte Carlo Dropout-like
        ensembling (aquí, reentrenos ligeros); para el TFG puede sustituirse
        por MC-Dropout real activando dropout en inferencia. Se deja como
        ejemplo mínimo de cómo abordar la sección 3.6 de la propuesta."""
        raise NotImplementedError(
            "Placeholder: implementar cuantificación de incertidumbre "
            "(ver sección 3.6 de la propuesta) como ampliación del TFG."
        )
    