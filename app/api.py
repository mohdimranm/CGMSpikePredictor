"""
FastAPI service for CGM spike prediction.

Provides /predict endpoint for glucose spike risk assessment.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="CGM Spike Predictor API",
    description="Predict glucose spikes from continuous glucose monitor data",
    version="1.0.0"
)

# Global variables for model and feature names
model = None
feature_names = None
feature_engineer = None


# Pydantic models for request/response
class CGMDataPoint(BaseModel):
    """Single CGM data point with all OhioT1DM fields."""
    timestamp: str = Field(..., description="ISO format timestamp")
    glucose: float = Field(..., description="Glucose level in mg/dL")
    carbs: Optional[float] = Field(0, description="Carbohydrate intake in grams")
    bolus_dose: Optional[float] = Field(0, description="Insulin bolus in units")
    meal_type: Optional[str] = Field(None, description="Type of meal")

    # Insulin pump data
    finger_stick_glucose: Optional[float] = Field(0, description="Manual finger stick reading in mg/dL")
    basal_rate: Optional[float] = Field(0, description="Basal insulin rate in u/hr")
    temp_basal_rate: Optional[float] = Field(0, description="Temporary basal rate in u/hr")

    # Event markers
    hypo_event: Optional[int] = Field(0, description="Hypoglycemia event (0 or 1)")
    stress_event: Optional[int] = Field(0, description="Stress event (0 or 1)")

    # Exercise data
    exercise_duration: Optional[float] = Field(0, description="Exercise duration in minutes")
    exercise_intensity: Optional[float] = Field(0, description="Exercise intensity 1-10")

    # Wearable sensor data
    heart_rate: Optional[float] = Field(0, description="Heart rate in bpm")
    gsr: Optional[float] = Field(0, description="Galvanic skin response")
    skin_temp_f: Optional[float] = Field(0, description="Skin temperature in Fahrenheit")
    air_temp_f: Optional[float] = Field(0, description="Air temperature in Fahrenheit")
    steps: Optional[float] = Field(0, description="Step count per 5-min window")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:30:00",
                "glucose": 145.0,
                "carbs": 60.0,
                "bolus_dose": 5.0,
                "meal_type": "Lunch",
                "finger_stick_glucose": 0,
                "basal_rate": 1.25,
                "temp_basal_rate": 1.25,
                "hypo_event": 0,
                "stress_event": 0,
                "exercise_duration": 0,
                "exercise_intensity": 0,
                "heart_rate": 75,
                "gsr": 0.000058,
                "skin_temp_f": 87.2,
                "air_temp_f": 85.1,
                "steps": 50
            }
        }


class PredictionRequest(BaseModel):
    """Request body for spike prediction."""
    user_id: str = Field(..., description="User identifier")
    recent_data: List[CGMDataPoint] = Field(
        ...,
        description="Recent CGM readings (at least 24 readings = 2 hours of data)",
        min_length=24
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123",
                "recent_data": [
                    {
                        "timestamp": "2024-01-15T14:00:00",
                        "glucose": 120.0,
                        "carbs": 0,
                        "bolus_dose": 0
                    },
                    {
                        "timestamp": "2024-01-15T14:05:00",
                        "glucose": 125.0,
                        "carbs": 0,
                        "bolus_dose": 0
                    }
                    # ... more readings
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response body for spike prediction."""
    user_id: str
    will_spike: bool = Field(..., description="Prediction: will glucose spike?")
    risk_score: float = Field(..., description="Probability of spike (0-1)")
    explanation: str = Field(..., description="Human-readable explanation")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123",
                "will_spike": True,
                "risk_score": 0.78,
                "explanation": "Risk: HIGH (78% probability of glucose spike above 180 mg/dL in next 2 hours). Current glucose is 165 mg/dL with rapidly rising velocity of 2.5 mg/dL/min (up 35 mg/dL in last 30 minutes). Recent significant carb intake of 60g consumed 15 minutes ago with glucose already rising."
            }
        }


def load_models():
    """Load trained model and feature names."""
    global model, feature_names, feature_engineer

    models_dir = Path("models")

    # Load model
    model_path = models_dir / "spike_predictor.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)

    # Load feature names
    feature_names_path = models_dir / "feature_names.txt"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    # Try to load feature engineer (if available)
    fe_path = models_dir / "feature_engineer.pkl"
    if fe_path.exists():
        feature_engineer = joblib.load(fe_path)

    print(f"✓ Loaded model: {type(model).__name__}")
    print(f"✓ Loaded {len(feature_names)} features")


def prepare_dataframe(recent_data: List[CGMDataPoint], user_id: str) -> pd.DataFrame:
    """Convert request data to DataFrame."""
    # Convert to list of dicts
    data_dicts = [point.model_dump() for point in recent_data]

    # Create DataFrame
    df = pd.DataFrame(data_dicts)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add patient_id
    df['patient_id'] = user_id

    # Fill missing values for all fields
    df['carbs'] = df['carbs'].fillna(0)
    df['bolus_dose'] = df['bolus_dose'].fillna(0)
    df['exercise_duration'] = df['exercise_duration'].fillna(0)
    df['exercise_intensity'] = df['exercise_intensity'].fillna(0)

    # Insulin pump fields
    df['finger_stick_glucose'] = df['finger_stick_glucose'].fillna(0)
    df['basal_rate'] = df['basal_rate'].fillna(0)
    df['temp_basal_rate'] = df['temp_basal_rate'].fillna(0)

    # Event markers
    df['hypo_event'] = df['hypo_event'].fillna(0)
    df['stress_event'] = df['stress_event'].fillna(0)

    # Physiological data - fill with mean if available, otherwise 0
    if 'heart_rate' in df.columns:
        mean_hr = df['heart_rate'].mean()
        df['heart_rate'] = df['heart_rate'].fillna(mean_hr if not pd.isna(mean_hr) else 0)
    else:
        df['heart_rate'] = 0

    df['gsr'] = df['gsr'].fillna(0)

    if 'skin_temp_f' in df.columns:
        mean_skin = df['skin_temp_f'].mean()
        df['skin_temp_f'] = df['skin_temp_f'].fillna(mean_skin if not pd.isna(mean_skin) else 0)
    else:
        df['skin_temp_f'] = 0

    if 'air_temp_f' in df.columns:
        mean_air = df['air_temp_f'].mean()
        df['air_temp_f'] = df['air_temp_f'].fillna(mean_air if not pd.isna(mean_air) else 0)
    else:
        df['air_temp_f'] = 0

    df['steps'] = df['steps'].fillna(0)

    return df


def engineer_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from raw data (simplified version).

    This is a lightweight version for API use.
    """
    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Glucose trends
    df['glucose_delta_5min'] = df['glucose'].diff(1)
    df['glucose_delta_15min'] = df['glucose'].diff(3)
    df['glucose_delta_30min'] = df['glucose'].diff(6)
    df['glucose_delta_60min'] = df['glucose'].diff(12)
    df['glucose_delta_120min'] = df['glucose'].diff(24)

    # Velocity
    df['glucose_velocity_5min'] = df['glucose_delta_5min'] / 5
    df['glucose_velocity_15min'] = df['glucose_delta_15min'] / 15
    df['glucose_velocity_30min'] = df['glucose_delta_30min'] / 30
    df['glucose_velocity_60min'] = df['glucose_delta_60min'] / 60

    # Acceleration
    df['glucose_accel_15min'] = df['glucose_velocity_15min'].diff(3)
    df['glucose_accel_30min'] = df['glucose_velocity_30min'].diff(6)

    # Direction
    df['glucose_rising'] = (df['glucose_delta_15min'] > 5).astype(int)
    df['glucose_falling'] = (df['glucose_delta_15min'] < -5).astype(int)
    df['glucose_stable'] = ((df['glucose_delta_15min'].abs()) <= 5).astype(int)
    df['glucose_rising_fast'] = (df['glucose_velocity_15min'] > 2).astype(int)

    # Rolling stats
    df['glucose_mean_30min'] = df['glucose'].rolling(window=6, min_periods=1).mean()
    df['glucose_mean_60min'] = df['glucose'].rolling(window=12, min_periods=1).mean()
    df['glucose_mean_120min'] = df['glucose'].rolling(window=24, min_periods=1).mean()
    df['glucose_std_30min'] = df['glucose'].rolling(window=6, min_periods=1).std()
    df['glucose_std_60min'] = df['glucose'].rolling(window=12, min_periods=1).std()
    df['glucose_range_30min'] = (
        df['glucose'].rolling(window=6, min_periods=1).max() -
        df['glucose'].rolling(window=6, min_periods=1).min()
    )
    df['glucose_range_60min'] = (
        df['glucose'].rolling(window=12, min_periods=1).max() -
        df['glucose'].rolling(window=12, min_periods=1).min()
    )
    df['glucose_baseline'] = df['glucose_mean_120min']

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Meal features
    df['minutes_since_meal'] = 999.0
    df['carbs_from_last_meal'] = 0.0

    meal_indices = df[df['carbs'] > 0].index
    for idx in df.index:
        past_meals = meal_indices[meal_indices <= idx]
        if len(past_meals) > 0:
            last_meal_idx = past_meals[-1]
            time_diff = (df.loc[idx, 'timestamp'] - df.loc[last_meal_idx, 'timestamp']).total_seconds() / 60
            df.loc[idx, 'minutes_since_meal'] = time_diff
            df.loc[idx, 'carbs_from_last_meal'] = df.loc[last_meal_idx, 'carbs']

    df['had_meal_recently'] = (df['minutes_since_meal'] < 60).astype(int)
    df['had_meal_very_recently'] = (df['minutes_since_meal'] < 30).astype(int)
    df['high_carb_meal'] = (df['carbs_from_last_meal'] > 50).astype(int)

    df['carbs_last_30min'] = df['carbs'].rolling(window=6, min_periods=1).sum()
    df['carbs_last_60min'] = df['carbs'].rolling(window=12, min_periods=1).sum()
    df['carbs_last_120min'] = df['carbs'].rolling(window=24, min_periods=1).sum()

    # Insulin features
    df['minutes_since_bolus'] = 999.0
    df['dose_from_last_bolus'] = 0.0

    bolus_indices = df[df['bolus_dose'] > 0].index
    for idx in df.index:
        past_bolus = bolus_indices[bolus_indices <= idx]
        if len(past_bolus) > 0:
            last_bolus_idx = past_bolus[-1]
            time_diff = (df.loc[idx, 'timestamp'] - df.loc[last_bolus_idx, 'timestamp']).total_seconds() / 60
            df.loc[idx, 'minutes_since_bolus'] = time_diff
            df.loc[idx, 'dose_from_last_bolus'] = df.loc[last_bolus_idx, 'bolus_dose']

    df['bolus_last_60min'] = df['bolus_dose'].rolling(window=12, min_periods=1).sum()
    df['bolus_last_120min'] = df['bolus_dose'].rolling(window=24, min_periods=1).sum()

    # Interaction features
    df['meal_no_insulin'] = (
        (df['minutes_since_meal'] < 60) &
        (df['minutes_since_bolus'] > 60)
    ).astype(int)
    df['high_carb_and_rising'] = (
        (df['carbs_from_last_meal'] > 50) &
        (df['glucose_rising'] == 1)
    ).astype(int)
    df['fast_rise_after_meal'] = (
        (df['glucose_velocity_15min'] > 2) &
        (df['minutes_since_meal'] < 90)
    ).astype(int)
    df['accelerating_rise'] = (
        (df['glucose_accel_15min'] > 0.1) &
        (df['glucose_rising'] == 1)
    ).astype(int)

    # Exercise features
    df['is_exercising'] = (df['exercise_duration'] > 0).astype(int)
    df['exercise_last_60min'] = df['exercise_duration'].rolling(window=12, min_periods=1).max()
    df['exercise_last_120min'] = df['exercise_duration'].rolling(window=24, min_periods=1).max()

    # Physiological features (if available)
    if 'heart_rate' not in df.columns:
        df['heart_rate'] = 0
    if 'steps' not in df.columns:
        df['steps'] = 0

    return df


def generate_explanation(probability: float, features: np.ndarray, feature_names: List[str]) -> str:
    """Generate human-readable explanation."""
    feat = dict(zip(feature_names, features))

    # Risk level
    if probability >= 0.80:
        risk = "VERY HIGH"
    elif probability >= 0.60:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MODERATE"
    elif probability >= 0.20:
        risk = "LOW"
    else:
        risk = "VERY LOW"

    explanation = f"Risk: {risk} ({probability:.0%} probability of glucose spike above 180 mg/dL in next 2 hours). "

    reasons = []

    # Glucose trends
    glucose_parts = []

    if 'glucose_mean_60min' in feat and feat['glucose_mean_60min'] > 0:
        glucose_parts.append(f"Recent glucose average is {feat['glucose_mean_60min']:.0f} mg/dL")
    elif 'glucose_baseline' in feat and feat['glucose_baseline'] > 0:
        glucose_parts.append(f"Baseline glucose is {feat['glucose_baseline']:.0f} mg/dL")

    if 'glucose_velocity_15min' in feat:
        velocity = feat['glucose_velocity_15min']
        if velocity > 2.0:
            glucose_parts.append(f"with rapidly rising velocity of {velocity:.1f} mg/dL/min")
        elif velocity > 0.8:
            glucose_parts.append(f"with rising velocity of {velocity:.1f} mg/dL/min")
        elif velocity > 0.3:
            glucose_parts.append(f"with slowly rising velocity of {velocity:.1f} mg/dL/min")
        elif velocity < -2.0:
            glucose_parts.append(f"with rapidly falling velocity of {abs(velocity):.1f} mg/dL/min")
        elif velocity < -0.8:
            glucose_parts.append(f"with falling velocity of {abs(velocity):.1f} mg/dL/min")

    if 'glucose_delta_30min' in feat and abs(feat['glucose_delta_30min']) > 10:
        delta = feat['glucose_delta_30min']
        if delta > 0:
            glucose_parts.append(f"(up {delta:.0f} mg/dL in last 30 minutes)")
        else:
            glucose_parts.append(f"(down {abs(delta):.0f} mg/dL in last 30 minutes)")

    if glucose_parts:
        reasons.append(" ".join(glucose_parts))

    # Meal context
    if 'minutes_since_meal' in feat and 'carbs_from_last_meal' in feat:
        minutes = feat['minutes_since_meal']
        carbs = feat['carbs_from_last_meal']

        if minutes < 120 and carbs > 0:
            if minutes < 60:
                time_str = f"{int(minutes)} minutes ago"
            else:
                time_str = f"{minutes/60:.1f} hours ago"

            if carbs > 60:
                carb_desc = f"large carb intake of {carbs:.0f}g"
            elif carbs > 40:
                carb_desc = f"significant carb intake of {carbs:.0f}g"
            else:
                carb_desc = f"moderate carb intake of {carbs:.0f}g"

            meal_str = f"Recent {carb_desc} consumed {time_str}"

            if feat.get('high_carb_and_rising', 0) == 1:
                meal_str += " with glucose already rising"

            reasons.append(meal_str)

    # Insulin context
    if 'minutes_since_bolus' in feat and 'dose_from_last_bolus' in feat:
        minutes = feat['minutes_since_bolus']
        dose = feat['dose_from_last_bolus']

        if minutes < 120 and dose > 0:
            if minutes < 60:
                time_str = f"{int(minutes)} minutes ago"
            else:
                time_str = f"{minutes/60:.1f} hours ago"
            reasons.append(f"Bolus insulin of {dose:.1f}u given {time_str}")
        elif feat.get('meal_no_insulin', 0) == 1:
            reasons.append("No recent insulin coverage for meal")

    if reasons:
        explanation += ". ".join(reasons) + "."

    return explanation


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        load_models()
        print("✓ API started successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "CGM Spike Predictor API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_names) if feature_names else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict glucose spike risk from recent CGM data.

    Requires at least 24 readings (2 hours) for feature engineering.
    """
    try:
        # Validate we have enough data
        if len(request.recent_data) < 24:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 24 readings (2 hours). Received {len(request.recent_data)}"
            )

        # Convert to DataFrame
        df = prepare_dataframe(request.recent_data, request.user_id)

        # Engineer features
        features_df = engineer_features_simple(df)

        # Get the most recent reading's features
        latest_idx = len(features_df) - 1

        # Extract features in the correct order and create DataFrame
        feature_values = []
        for fname in feature_names:
            if fname in features_df.columns:
                value = features_df.loc[latest_idx, fname]
                # Handle NaN
                if pd.isna(value):
                    value = 0
                feature_values.append(value)
            else:
                # Feature not available, use 0
                feature_values.append(0)

        # Convert to DataFrame with feature names to avoid sklearn warning
        X = pd.DataFrame([feature_values], columns=feature_names)
        print(X.head())
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]

        # Generate explanation
        explanation = generate_explanation(probability, feature_values, feature_names)

        # Format response
        return PredictionResponse(
            user_id=request.user_id,
            will_spike=bool(prediction),
            risk_score=round(float(probability), 2),
            explanation=explanation
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)