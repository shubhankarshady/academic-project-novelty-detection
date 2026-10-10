
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.hybrid_fusion import HybridSectionFusionModel

MODEL_PATH = ROOT_DIR / "models" / "hybrid_fusion_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Check the filename in your models/ folder."
    )

# Load the saved HybridSectionFusionModel checkpoint.
model = HybridSectionFusionModel.load(str(MODEL_PATH))

# Show the raw model coefficients/importances.
df = model.get_feature_importances()

print("\nFEATURE COEFFICIENTS / IMPORTANCES")
print("=" * 72)
print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# For Ridge, calculate a descriptive channel comparison using
# absolute coefficient magnitudes. These are NOT percentages of model output.
if hasattr(model.model, "coef_"):
    coefficients = np.asarray(model.model.coef_).ravel()

    if len(coefficients) != 16:
        raise ValueError(f"Expected 16 coefficients, got {len(coefficients)}")

    lexical_magnitude = np.abs(coefficients[:8]).sum()
    semantic_magnitude = np.abs(coefficients[8:]).sum()
    total_magnitude = lexical_magnitude + semantic_magnitude

    print("\nRIDGE CHANNEL MAGNITUDE COMPARISON")
    print("=" * 45)

    if total_magnitude > 0:
        print(
            f"Lexical |coefficient| share:  "
            f"{100 * lexical_magnitude / total_magnitude:.2f}%"
        )
        print(
            f"Semantic |coefficient| share: "
            f"{100 * semantic_magnitude / total_magnitude:.2f}%"
        )
    else:
        print("All coefficients are zero.")

    print("\nNote: These shares describe coefficient magnitudes,")
    print("not causal contribution or performance improvement.")

elif hasattr(model.model, "feature_importances_"):
    importances = np.asarray(model.model.feature_importances_).ravel()

    if len(importances) != 16:
        raise ValueError(f"Expected 16 importances, got {len(importances)}")

    lexical_importance = importances[:8].sum()
    semantic_importance = importances[8:].sum()

    print("\nTREE MODEL CHANNEL IMPORTANCE")
    print("=" * 45)
    print(f"Lexical importance:  {100 * lexical_importance:.2f}%")
    print(f"Semantic importance: {100 * semantic_importance:.2f}%")
    print(f"Total importance:    {100 * importances.sum():.2f}%")

else:
    print("\nThis model does not expose coefficients or feature importances.")
