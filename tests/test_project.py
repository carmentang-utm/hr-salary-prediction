import sys
from pathlib import Path
import pandas as pd
import joblib

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validate_data import validate_dataset

BASE_DIR = Path(__file__).resolve().parents[1]


def test_data_validation():
    """Cleaned dataset must pass all automated quality checks."""
    assert validate_dataset(BASE_DIR / "data" / "cleaned_employee_dataset.csv"), \
        "Data validation failed - dataset contains quality issues"


def test_model_loads():
    """Trained model artefact must exist and load correctly."""
    model = joblib.load(BASE_DIR / "models" / "salary_model.pkl")
    assert model is not None, "Model failed to load"


def test_model_prediction():
    """Model must produce a valid salary prediction for a sample employee."""
    model = joblib.load(BASE_DIR / "models" / "salary_model.pkl")
    sample = pd.DataFrame({
        "Age": [30],
        "Department": ["DevOps"],
        "Region": ["California"],
        "Status": ["Active"],
        "Performance_Level": [3],
        "Remote_Work": [1],
        "Tenure_Years": [3.5],
    })
    prediction = model.predict(sample)[0]
    assert 20000 <= prediction <= 200000, \
        f"Prediction {prediction} outside valid salary range"
