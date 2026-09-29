# SALES PREDICTION USING PYTHON

A beginner-friendly machine learning project that estimates product sales from TV, radio, and newspaper advertising budgets. It includes data inspection, exploratory plots, a Linear Regression model, command-line predictions, and an optional Streamlit interface.

## Objective

Learn the relationship between advertising expenditure and sales, then use the trained model to estimate sales for new budgets. The model is educational and should not be treated as a guarantee of business outcomes.

## Dataset

Place the file at `data/advertising.csv`. It must be a comma-separated file with these exact column names and numeric values:

```csv
TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9
```

The project includes a 200-row copy of the widely used Advertising dataset. If it is absent, obtain the public CSV from [the scikit-learn-videos data repository](https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv), then save it as `data/advertising.csv`. That source includes an extra unnamed row-number column; the training script safely ignores it. Alternatively, create your own CSV with the four-column format above. The example rows alone are only a tiny demonstration and are not a good basis for meaningful evaluation.

If your file has a different filename, rename it to `advertising.csv` or update `DATA_PATH` near the top of `src/train_model.py`. If your columns use different names, rename them in the CSV to `TV`, `Radio`, `Newspaper`, and `Sales`, or update `FEATURES` and `TARGET` in both Python scripts, `app.py`, and the notebook so the names stay consistent. Columns are case-sensitive.

`TV`, `Radio`, and `Newspaper` are the input features; `Sales` is the target. Budgets must be numeric and non-negative. Rows with missing or non-numeric values stop training with a message; fix or remove those rows before retrying. Repeated rows are reported, not silently removed.

## Technologies

Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, and Streamlit. No paid service or runtime network access is needed.

## Project structure

```text
Sales_Prediction/
├── data/
│   └── advertising.csv
├── assets/
│   └── advertising-analytics.jpg
├── models/
│   ├── sales_prediction_model.pkl
│   └── metrics.json
├── notebooks/
│   └── sales_prediction.ipynb
├── outputs/
│   ├── advertising_vs_sales.png
│   ├── correlation_heatmap.png
│   ├── sales_distribution.png
│   ├── actual_vs_predicted.png
│   └── residuals.png
├── src/
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Generated model and plot files are produced by training and can be recreated at any time.
The Streamlit hero background is bundled locally at `assets/advertising-analytics.jpg`; it was downloaded from [Unsplash](https://images.unsplash.com/photo-1460925895917-afdab827c52f) and is not fetched when the app runs.

## VS Code setup

1. Open the `Sales_Prediction` folder in VS Code. If starting from an empty folder, create the folders and files shown above, or use this completed project folder.
2. Open **Terminal > New Terminal**. Confirm the terminal is in the `Sales_Prediction` project folder, the one containing `requirements.txt`.
3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate it using the command for your terminal:

   **Windows PowerShell:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use the Windows CMD command below, or allow scripts for the current user according to your organization's policy.

   **Windows CMD:**
   ```cmd
   venv\Scripts\activate.bat
   ```

   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

5. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. In VS Code, open the Command Palette with `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS), choose **Python: Select Interpreter**, and select the interpreter inside this project's `venv` folder. In notebooks, select the same environment as the notebook kernel.

## Run order

Run these commands from the `Sales_Prediction` project folder. Make sure `data/advertising.csv` exists first.

1. Install dependencies with `pip install -r requirements.txt` as described above. Expected: packages install into the active virtual environment.
2. Train and evaluate the model:

   ```bash
   python src/train_model.py
   ```

   Expected: the script prints the first five records, dataset shape, columns, types, missing-value and duplicate counts, summary statistics, correlations, coefficients, sample training predictions, all test predictions, and test metrics. It saves `models/sales_prediction_model.pkl`, `models/metrics.json`, and the EDA/evaluation PNGs in `outputs/`.
3. Verify `models/sales_prediction_model.pkl` exists, then run the command-line predictor:

   ```bash
   python src/predict.py
   ```

   Enter non-negative numeric TV, radio, and newspaper budgets when prompted. For example, enter `150`, `25`, and `20`; expected output is a line like `Predicted Sales: 15.XX`. The exact value comes from the trained model and is not hard-coded. Invalid or negative inputs are rejected and prompted again.
4. Start the optional web app:

   ```bash
   streamlit run app.py
   ```

   Expected: Streamlit prints a local URL. Open it in a browser, enter budgets, and select **Predict Sales**. The app loads the saved model and displays the real prediction and test-set metrics. Stop it with `Ctrl+C` in the terminal.

You can also open `notebooks/sales_prediction.ipynb` and run its cells from top to bottom for an interactive version of the same workflow.

## Model and evaluation

The main algorithm is `sklearn.linear_model.LinearRegression`. The features and target are kept separate (`X = df[['TV', 'Radio', 'Newspaper']]` and `y = df['Sales']`) so the model learns from the advertising inputs to predict the outcome. `train_test_split` reserves 20% for testing and uses `random_state=42` for reproducibility.

- **MAE:** average absolute prediction error, in sales units; lower is better.
- **MSE:** average squared error; larger mistakes count more heavily; lower is better.
- **RMSE:** square root of MSE, expressed in sales units; lower is better.
- **R²:** how much of the target variation is explained on the test split; closer to 1 can indicate a better fit, but does not mean predictions are perfect or establish causation.

The model's coefficients estimate the change in predicted sales for a one-unit increase in one advertising budget while the other two budgets are held constant. A coefficient is an association in this dataset, not proof that changing a budget causes that change in sales.

## Example prediction

For TV `150`, Radio `25`, and Newspaper `20`, `src/predict.py` prints the estimate produced by the saved model. Since model values depend on the training data, use the program's output rather than a copied or hard-coded number.

## Future improvements

Compare Linear Regression with alternatives such as `RandomForestRegressor` using the same train/test split and MAE, RMSE, and R² metrics. Keep Linear Regression as the baseline, compare on held-out data (or cross-validation), and avoid choosing a model using test data repeatedly. Other useful extensions include cross-validation, input-range checks based on the training data, and monitoring performance on newer data.

## Conclusion

This project demonstrates a complete, reproducible regression workflow: inspect data, visualize relationships, train and evaluate a model, save it with Joblib, and reuse it for command-line or web predictions.