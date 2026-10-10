
import numpy as np
import joblib

from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


class ConstrainedSectionFusion:
    """Non-negative section weights constrained to sum to one."""

    def __init__(self):
        self.weights = None
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("X and y have incompatible shapes.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")
        if X.shape[1] == 0:
            raise ValueError("At least one feature is required.")
        X = np.clip(X, 0.0, 1.0)
        if np.any(y < 0) or np.any(y > 4):
            raise ValueError("Target labels must be in [0, 4].")

        self.feature_names = (
            list(feature_names) if feature_names is not None else None
        )

        n_features = X.shape[1]

        def objective(weights):
            predictions = 4.0 * (X @ weights)
            return np.mean((predictions - y) ** 2)

        result = minimize(
            objective,
            x0=np.full(n_features, 1.0 / n_features),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_features,
            constraints={
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1.0,
            },
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"SLSQP optimization failed: {result.message}"
            )

        weights = result.x

        if (
            not np.isfinite(weights).all()
            or np.any(weights < -1e-7)
            or not np.isclose(weights.sum(), 1.0, atol=1e-6)
        ):
            raise RuntimeError("Optimizer returned invalid weights.")

        # Remove tiny numerical errors and renormalize.
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()

        self.weights = weights
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("X has the wrong number of features.")

        return np.clip(4.0 * (X @ self.weights), 0.0, 4.0)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)


class SectionFusionRegressor:
    """Regression model over section-wise similarity features."""

    def __init__(self, model_type="ridge"):
        if model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model_type = model_type

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0.0, 4.0)

    def save(self, filepath):
        joblib.dump(
            {
                "model_type": self.model_type,
                "model": self.model,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath):
        artifact = joblib.load(filepath)

        # Backward compatibility with older files that saved only
        # the underlying sklearn estimator.
        if isinstance(artifact, dict):
            instance = cls(model_type=artifact["model_type"])
            instance.model = artifact["model"]
        else:
            instance = cls.__new__(cls)
            instance.model = artifact
            if isinstance(artifact, Ridge):
                instance.model_type = "ridge"
            elif isinstance(artifact, RandomForestRegressor):
                instance.model_type = "random_forest"
            else:
                raise ValueError("Unrecognized saved estimator.")

        return instance
