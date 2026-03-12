from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
TARGET_COLUMN = "Clicked on Ad"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_OUTPUT_PATH = Path("model/ad_click_model.pkl")


def resolve_dataset_path() -> Path:
    """Resolve the dataset location, supporting common project layouts."""
    candidate_paths = [Path("advertising.csv"), Path("dataset/advertising.csv")]
    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find 'advertising.csv'. Place it in the project root or in 'dataset/advertising.csv'."
    )


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load dataset and validate that all required columns are present."""
    data_frame = pd.read_csv(dataset_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data_frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return data_frame[required_columns].copy()


def preprocess_data(data_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector for model training."""
    # Convert feature columns to numeric and coerce invalid values to NaN.
    for column in FEATURE_COLUMNS:
        data_frame[column] = pd.to_numeric(data_frame[column], errors="coerce")

    # Convert target to numeric and remove rows with invalid target values.
    data_frame[TARGET_COLUMN] = pd.to_numeric(data_frame[TARGET_COLUMN], errors="coerce")
    data_frame = data_frame.dropna(subset=[TARGET_COLUMN]).copy()

    # Ensure binary target values (0/1) and integer type.
    data_frame[TARGET_COLUMN] = data_frame[TARGET_COLUMN].clip(0, 1).astype(int)

    # Fill missing feature values with column medians.
    for column in FEATURE_COLUMNS:
        median_value = data_frame[column].median()
        if pd.isna(median_value):
            raise ValueError(f"Unable to compute median for feature column: {column}")
        data_frame[column] = data_frame[column].fillna(median_value)

    target = data_frame[TARGET_COLUMN]
    if target.nunique() < 2:
        raise ValueError("Training data must contain at least two classes in 'Clicked on Ad'.")

    features = data_frame[FEATURE_COLUMNS]
    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets with stratification."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a Logistic Regression model using a scaling + classifier pipeline."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, list[list[int]]]:
    """Evaluate model performance using accuracy and confusion matrix."""
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    matrix = confusion_matrix(y_test, predictions).tolist()
    return accuracy, matrix


def save_model(model: Pipeline, output_path: Path = MODEL_OUTPUT_PATH) -> None:
    """Persist the trained model to disk using pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as model_file:
        pickle.dump(model, model_file)


def main() -> None:
    """Execute the full training workflow for ad-click prediction."""
    dataset_path = resolve_dataset_path()
    raw_data = load_dataset(dataset_path)

    features, target = preprocess_data(raw_data)
    x_train, x_test, y_train, y_test = split_data(features, target)

    model = train_model(x_train, y_train)
    accuracy, matrix = evaluate_model(model, x_test, y_test)
    save_model(model)

    print("Model training completed successfully.")
    print(f"Dataset: {dataset_path}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    for row in matrix:
        print(row)


if __name__ == "__main__":
    main()