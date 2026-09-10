"""Modelos clásicos / ensemble: el punto de partida recomendado para el TFG
(rápidos de entrenar, interpretables y muy competitivos con buenas features)."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.models.base_model import BaseModel


class LinearBaselineModel(BaseModel):
    name = "linear_baseline"

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, n_estimators: int = 300, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def feature_importance(self):
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, n_estimators: int = 300, max_depth: int = 6, learning_rate: float = 0.05,
                 random_state: int = 42):
        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def feature_importance(self):
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))