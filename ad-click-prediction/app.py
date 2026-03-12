from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

FEATURE_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]

FORM_FIELD_ORDER = [
    "daily_time_spent",
    "age",
    "area_income",
    "daily_internet_usage",
    "male",
]

MODEL_PATH = Path("model/ad_click_model.pkl")

app = Flask(__name__)

MODEL: Any | None = None
MODEL_LOAD_ERROR: str | None = None


@dataclass(frozen=True)
class InputRule:
    label: str
    min_value: float
    max_value: float


INPUT_RULES: dict[str, InputRule] = {
    "daily_time_spent": InputRule("Daily Time Spent on Site", 0, 300),
    "age": InputRule("Age", 10, 120),
    "area_income": InputRule("Area Income", 0, 1_000_000),
    "daily_internet_usage": InputRule("Daily Internet Usage", 0, 500),
    "male": InputRule("Male", 0, 1),
}


def load_model(model_path: Path) -> Any:
    """Load a trained ad-click model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run train_model.py before starting the app."
        )

    with model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]

    return artifact


def ensure_model_ready() -> tuple[bool, str | None]:
    """Load model lazily and cache model loading errors."""
    global MODEL, MODEL_LOAD_ERROR

    if MODEL is not None:
        return True, None

    if MODEL_LOAD_ERROR:
        return False, MODEL_LOAD_ERROR

    try:
        MODEL = load_model(MODEL_PATH)
        return True, None
    except Exception as error:
        MODEL_LOAD_ERROR = str(error)
        return False, MODEL_LOAD_ERROR


def validate_and_build_input(form_values: dict[str, str]) -> tuple[np.ndarray | None, list[str]]:
    """Validate user form values and build the NumPy input array for inference."""
    numeric_values: list[float] = []
    validation_errors: list[str] = []

    for field_name in FORM_FIELD_ORDER:
        rule = INPUT_RULES[field_name]
        raw_value = str(form_values.get(field_name, "")).strip()

        if not raw_value:
            validation_errors.append(f"{rule.label} is required.")
            continue

        try:
            numeric_value = float(raw_value)
        except ValueError:
            validation_errors.append(f"{rule.label} must be a numeric value.")
            continue

        if not rule.min_value <= numeric_value <= rule.max_value:
            validation_errors.append(
                f"{rule.label} must be between {rule.min_value:g} and {rule.max_value:g}."
            )
            continue

        if field_name == "male" and numeric_value not in (0, 1):
            validation_errors.append("Gender value must be 0 (Female) or 1 (Male).")
            continue

        numeric_values.append(float(int(numeric_value)) if field_name == "male" else numeric_value)

    if validation_errors:
        return None, validation_errors

    input_array = np.array([numeric_values], dtype=float)
    return input_array, []


def render_home(
    *,
    model_ready: bool,
    model_error: str | None,
    form_data: dict[str, str] | None = None,
    prediction_text: str | None = None,
    prediction_value: int | None = None,
    errors: list[str] | None = None,
) -> str:
    """Render the homepage with optional prediction, form data, and errors."""
    return render_template(
        "index.html",
        model_ready=model_ready,
        model_error=model_error,
        form_data=form_data or {},
        prediction_text=prediction_text,
        prediction_value=prediction_value,
        errors=errors or [],
    )


def wants_json_response() -> bool:
    """Detect whether the request expects a JSON response for async UI updates."""
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def build_prediction_payload(prediction: int) -> dict[str, Any]:
    """Create a consistent response payload for ad-click predictions."""
    prediction_text = (
        "User will likely click on the advertisement."
        if prediction == 1
        else "User will likely not click on the advertisement."
    )
    return {
        "success": True,
        "prediction": prediction,
        "message": prediction_text,
        "result_tone": "success" if prediction == 1 else "secondary",
    }


@app.route("/", methods=["GET"])
def index() -> str:
    """Render the prediction form UI."""
    model_ready, model_error = ensure_model_ready()
    return render_home(model_ready=model_ready, model_error=model_error)


@app.route("/predict", methods=["POST"])
def predict() -> tuple[str, int] | str:
    """Handle prediction requests from the form and display the result."""
    model_ready, model_error = ensure_model_ready()
    form_data = request.form.to_dict(flat=True)
    is_async_request = wants_json_response()

    if not model_ready:
        if is_async_request:
            return (
                jsonify(
                    {
                        "success": False,
                        "errors": [
                            model_error
                            or "Model is unavailable. Run train_model.py and restart the server."
                        ],
                    }
                ),
                500,
            )

        return render_home(
            model_ready=False,
            model_error=model_error,
            form_data=form_data,
            errors=[
                model_error or "Model is unavailable. Run train_model.py and restart the server."
            ],
        ), 500

    input_array, validation_errors = validate_and_build_input(form_data)

    if validation_errors:
        if is_async_request:
            return jsonify({"success": False, "errors": validation_errors}), 400

        return render_home(
            model_ready=True,
            model_error=None,
            form_data=form_data,
            errors=validation_errors,
        ), 400

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            prediction = int(MODEL.predict(input_array)[0])

        response_payload = build_prediction_payload(prediction)

        if is_async_request:
            return jsonify(response_payload)

        return render_home(
            model_ready=True,
            model_error=None,
            form_data=form_data,
            prediction_text=response_payload["message"],
            prediction_value=response_payload["prediction"],
        )
    except Exception as error:
        if is_async_request:
            return jsonify({"success": False, "errors": [f"Prediction failed: {error}"]}), 500

        return render_home(
            model_ready=True,
            model_error=None,
            form_data=form_data,
            errors=[f"Prediction failed: {error}"],
        ), 500


if __name__ == "__main__":
    app.run(debug=True)