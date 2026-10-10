import sys
from pathlib import Path

import numpy as np
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.features.section_tfidf import FEATURE_NAMES
from src.models.fusion_model import (
    ConstrainedSectionFusion,
    SectionFusionRegressor,
)

# Small dummy dataset with 8 section-similarity features
X = np.array([
    [0.10, 0.20, 0.15, 0.10, 0.30, 0.05, 0.10, 0.20],
    [0.80, 0.75, 0.90, 0.85, 0.20, 0.70, 0.80, 0.30],
    [0.30, 0.40, 0.35, 0.30, 0.25, 0.20, 0.30, 0.25],
    [0.90, 0.85, 0.95, 0.90, 0.15, 0.90, 0.85, 0.35],
    [0.15, 0.25, 0.20, 0.15, 0.35, 0.10, 0.15, 0.20],
    [0.65, 0.70, 0.75, 0.70, 0.20, 0.60, 0.70, 0.30],
], dtype=float)

y = np.array([0.5, 3.5, 1.0, 4.0, 0.7, 3.0])

assert X.shape[1] == len(FEATURE_NAMES)

# 1. Test constrained fusion
fusion = ConstrainedSectionFusion()
fusion.fit(X, y, feature_names=FEATURE_NAMES)

predictions = fusion.predict(X)

assert predictions.shape == y.shape
assert np.isfinite(predictions).all()
assert np.all((predictions >= 0) & (predictions <= 4))
assert np.all(fusion.weights >= -1e-8)
assert np.isclose(fusion.weights.sum(), 1.0, atol=1e-5)

print("Constrained fusion: PASS")
print("Weights sum:", fusion.weights.sum())
print("Predictions:", np.round(predictions, 3))

# 2. Test Ridge training and save/load
ridge = SectionFusionRegressor(model_type="ridge")
ridge.fit(X, y)

test_path = Path("models") / "_smoke_test_ridge.joblib"
test_path.parent.mkdir(parents=True, exist_ok=True)

ridge.save(test_path)
loaded_ridge = SectionFusionRegressor.load(test_path)
loaded_predictions = loaded_ridge.predict(X)

assert np.allclose(ridge.predict(X), loaded_predictions)

print("Ridge training and save/load: PASS")

# Remove the temporary test checkpoint
test_path.unlink(missing_ok=True)

print("\nAll smoke tests passed.")
