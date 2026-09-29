"""Explore the advertising data, train Linear Regression, and save its outputs."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "advertising.csv"
MODEL_PATH = PROJECT_DIR / "models" / "sales_prediction_model.pkl"
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"


def load_and_inspect_data() -> pd.DataFrame:
    """Read the CSV, validate its values, and print beginner-friendly summaries."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place advertising.csv in the project's data/ folder."
        )

    try:
        data = pd.read_csv(DATA_PATH)
    except pd.errors.ParserError as error:
        raise ValueError(f"Could not parse the dataset CSV: {error}") from error

    # The commonly shared source file includes a row-number column; it is not a feature.
    data = data.loc[:, ~data.columns.astype(str).str.match(r"^Unnamed:")]
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(missing_columns)}. "
            f"Required columns are: {', '.join(required_columns)}."
        )

    data = data[required_columns].copy()
    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_counts = data.isna().sum()
    if missing_counts.any():
        columns_with_missing = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            "The dataset contains missing or non-numeric values: "
            f"{columns_with_missing}. Correct these values and try again."
        )
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("The dataset contains infinite values; replace them with finite numbers.")
    if (data[FEATURES] < 0).any().any():
        raise ValueError("Advertising budgets cannot be negative.")
    if data.empty:
        raise ValueError("The dataset contains no data rows.")

    print("First five rows:\n", data.head(), "\n")
    print(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print("Columns:", data.columns.tolist())
    print("\nData types:\n", data.dtypes)
    print("\nMissing values:\n", data.isna().sum())
    print(f"\nDuplicate rows: {data.duplicated().sum()}")
    print("\nBasic statistics:\n", data.describe())
    print("\nCorrelation matrix:\n", data.corr(numeric_only=True))
    return data


def save_exploratory_plots(data: pd.DataFrame) -> None:
    """Save the advertising scatter plots, sales distribution, and correlation heatmap."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, feature in zip(axes, FEATURES):
        sns.scatterplot(data=data, x=feature, y=TARGET, ax=axis, alpha=0.8)
        axis.set_title(f"{feature} Advertising vs Sales")
        axis.set_xlabel(f"{feature} advertising budget")
        axis.set_ylabel("Sales")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "advertising_vs_sales.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=axis)
    axis.set_title("Correlation Between Advertising and Sales")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x=TARGET, kde=True, ax=axis)
    axis.set_title("Distribution of Sales")
    axis.set_xlabel("Sales")
    axis.set_ylabel("Number of records")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "sales_distribution.png", dpi=150)
    plt.close(figure)


def train_and_evaluate(data: pd.DataFrame) -> tuple[LinearRegression, dict[str, float]]:
    """Split the data, fit Linear Regression, evaluate it, and save its files."""
    # X holds the input clues; y holds the outcome the model learns to predict.
    features = data[FEATURES]
    target = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    training_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("\nModel intercept:", round(float(model.intercept_), 4))
    print("Model coefficients (effect per one-unit increase, holding other inputs fixed):")
    for feature, coefficient in zip(FEATURES, model.coef_):
        print(f"  {feature}: {coefficient:.4f}")
    print("A positive coefficient means the predicted sales tend to rise as that budget rises;")
    print("a negative coefficient means they tend to fall, with other budgets held constant.\n")
    print("First 10 training predictions:")
    print(pd.DataFrame({"Actual Sales": y_train, "Predicted Sales": training_predictions}).head(10).to_string(index=False))
    print("\nTesting predictions:")
    print(pd.DataFrame({"Actual Sales": y_test, "Predicted Sales": test_predictions}).to_string(index=False))

    metrics = {
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "mse": float(mean_squared_error(y_test, test_predictions)),
        "rmse": float(mean_squared_error(y_test, test_predictions) ** 0.5),
        "r2": float(r2_score(y_test, test_predictions)),
    }
    print("\nTest-set evaluation:")
    print(f"  MAE  (average absolute error): {metrics['mae']:.4f}")
    print(f"  MSE  (average squared error): {metrics['mse']:.4f}")
    print(f"  RMSE (error in sales units):  {metrics['rmse']:.4f}")
    print(f"  R²   (variance explained):    {metrics['r2']:.4f}")
    print("These scores describe this test split; a strong R² does not guarantee perfect predictions.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print(f"Saved test metrics to: {METRICS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(y_test, test_predictions, alpha=0.8, label="Test observations")
    lower = min(float(y_test.min()), float(test_predictions.min()))
    upper = max(float(y_test.max()), float(test_predictions.max()))
    axis.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="Perfect prediction")
    axis.set_title("Actual Sales vs Predicted Sales")
    axis.set_xlabel("Actual Sales")
    axis.set_ylabel("Predicted Sales")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(figure)

    residuals = y_test - test_predictions
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(test_predictions, residuals, alpha=0.8)
    axis.axhline(0, color="red", linestyle="--", label="Zero error")
    axis.set_title("Prediction Residuals")
    axis.set_xlabel("Predicted Sales")
    axis.set_ylabel("Actual Sales - Predicted Sales")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "residuals.png", dpi=150)
    plt.close(figure)

    return model, metrics


def main() -> int:
    """Run the complete training workflow and report actionable input errors."""
    try:
        dataset = load_and_inspect_data()
        save_exploratory_plots(dataset)
        train_and_evaluate(dataset)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())