"""Load the saved model and predict sales from three advertising budgets."""

import math
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "sales_prediction_model.pkl"
FEATURES = ["TV", "Radio", "Newspaper"]


def read_non_negative_number(prompt: str) -> float:
    """Prompt until the user enters a finite, non-negative number."""
    while True:
        raw_value = input(prompt).strip()
        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number, such as 150 or 25.5.")
            continue
        if not math.isfinite(value):
            print("Please enter a finite number.")
        elif value < 0:
            print("Advertising budgets cannot be negative.")
        else:
            return value


def predict_sales(tv: float, radio: float, newspaper: float) -> float:
    """Load the persisted model and return a sales prediction."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run `python src/train_model.py` from the project folder first."
        )
    values = [tv, radio, newspaper]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("All advertising budgets must be finite numbers.")
    if any(value < 0 for value in values):
        raise ValueError("Advertising budgets cannot be negative.")

    model = joblib.load(MODEL_PATH)
    input_data = pd.DataFrame([values], columns=FEATURES)
    return float(model.predict(input_data)[0])


def main() -> int:
    """Collect validated budgets, predict sales, and handle expected errors."""
    try:
        print("Enter advertising budgets using the same units as the training data.")
        tv = read_non_negative_number("TV advertising budget: ")
        radio = read_non_negative_number("Radio advertising budget: ")
        newspaper = read_non_negative_number("Newspaper advertising budget: ")
        prediction = predict_sales(tv, radio, newspaper)
    except (FileNotFoundError, ValueError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"\nPredicted Sales: {prediction:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())