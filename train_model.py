import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["Age", "Department", "Region", "Status",
            "Performance_Level", "Remote_Work", "Tenure_Years"]

df = pd.read_csv("data/cleaned_employee_dataset.csv")
X, y = df[FEATURES], df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    [("onehot", OneHotEncoder(handle_unknown="ignore"),
      ["Department", "Region", "Status"])],
    remainder="passthrough")

# Best hyperparameters selected by GridSearchCV in Sprint 3
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.03, random_state=42))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):,.2f}")
print(f"R2:   {r2_score(y_test, pred):.4f}")

joblib.dump(model, "models/salary_model.pkl")
print("Model saved to models/salary_model.pkl")
