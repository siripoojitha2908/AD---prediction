# Ad Click Prediction System

A full-stack machine learning web application that predicts whether a user will click on an online advertisement.

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Pandas, NumPy, Scikit-learn (Logistic Regression)
- **Frontend:** HTML, CSS, Bootstrap
- **Visualization:** Matplotlib, Seaborn

## Project Structure

```text
ad-click-prediction/
│
├── dataset/
│   └── advertising.csv
│
├── model/
│   └── ad_click_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Features

1. **Dataset Handling**
   - Loads `dataset/advertising.csv` with required advertising columns.
   - Handles missing values using median imputation for feature columns.
   - Selects only the required model features and target column.

2. **Exploratory Data Analysis (EDA)**
   - Saves visualizations to `model/eda/`:
     - Correlation heatmap
     - Time spent vs internet usage scatter plot
     - Age distribution by click class

3. **Model Development**
   - Splits data into train/test sets.
   - Trains a baseline Logistic Regression model.
   - Evaluates model with:
     - Accuracy
     - Confusion Matrix
     - Classification Report

4. **Model Persistence**
   - Saves trained model artifact as `model/ad_click_model.pkl`.
   - Saves metrics summary as `model/metrics.json`.

5. **Flask Web App**
   - Exposes `/predict` endpoint for form/API predictions.
   - Validates inputs and handles invalid data with clear error messages.
   - Returns prediction class and click probability.

6. **Responsive Frontend**
   - Bootstrap form captures all required features.
   - Uses JavaScript `fetch` for dynamic prediction updates without page reload.

## Setup and Run

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:

   ```bash
   python train_model.py
   ```

4. **Run the Flask server**:

   ```bash
   python app.py
   ```

5. Open your browser at:

   ```
   http://127.0.0.1:5000
   ```

## API Usage

`POST /predict`

Form fields:
- `daily_time_spent`
- `age`
- `area_income`
- `daily_internet_usage`
- `male` (Male = 1, Female = 0)

Sample JSON success response:

```json
{
  "success": true,
  "prediction": 1,
  "probability": 0.87,
  "message": "User is likely to click on the advertisement."
}
```