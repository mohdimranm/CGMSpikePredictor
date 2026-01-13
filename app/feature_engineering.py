"""
Feature engineering for CGM spike prediction.

Creates time-series features from glucose, meal, insulin, and exercise data
to predict whether glucose will spike in the next 2 hours.
"""
from typing import Tuple

import numpy as np
import pandas as pd

# Feature engineering parameters
SPIKE_THRESHOLD = 180.0  # Glucose > 180 mg/dL = spike
PREDICTION_HORIZON_MINUTES = 120  # Predict 2 hours ahead


def _coerce_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_cols = [
        col for col in df.columns if col == "timestamp" or col.endswith("_timestamp")
    ]
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def engineer_features_for_patient(
    df_patient: pd.DataFrame,
    spike_threshold: float = SPIKE_THRESHOLD,
    prediction_horizon_minutes: int = PREDICTION_HORIZON_MINUTES,
) -> pd.DataFrame:
    """
    Engineer temporal features for spike prediction.

    Focus on glucose dynamics and recent events.
    """
    df = df_patient.copy()
    df = _coerce_timestamps(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ========================================
    # 1. GLUCOSE DYNAMICS (PRIMARY FEATURES)
    # ========================================
    # Deltas - change over different time windows
    df["glucose_delta_5min"] = df["glucose"].diff(1)
    df["glucose_delta_15min"] = df["glucose"].diff(3)
    df["glucose_delta_30min"] = df["glucose"].diff(6)
    df["glucose_delta_60min"] = df["glucose"].diff(12)
    df["glucose_delta_120min"] = df["glucose"].diff(24)

    # Velocity - rate of change per minute
    df["glucose_velocity_5min"] = df["glucose_delta_5min"] / 5
    df["glucose_velocity_15min"] = df["glucose_delta_15min"] / 15
    df["glucose_velocity_30min"] = df["glucose_delta_30min"] / 30
    df["glucose_velocity_60min"] = df["glucose_delta_60min"] / 60

    # Acceleration - change in velocity
    df["glucose_accel_15min"] = df["glucose_velocity_15min"].diff(3)
    df["glucose_accel_30min"] = df["glucose_velocity_30min"].diff(6)

    # Directional indicators
    df["glucose_rising"] = (df["glucose_delta_15min"] > 5).astype(int)
    df["glucose_falling"] = (df["glucose_delta_15min"] < -5).astype(int)
    df["glucose_stable"] = (df["glucose_delta_15min"].abs() <= 5).astype(int)
    df["glucose_rising_fast"] = (df["glucose_velocity_15min"] > 2).astype(int)

    # ========================================
    # 2. GLUCOSE VARIABILITY (TRENDS)
    # ========================================
    # Standard deviation - measure of variability
    df["glucose_std_30min"] = df["glucose"].rolling(window=6, min_periods=1).std()
    df["glucose_std_60min"] = df["glucose"].rolling(window=12, min_periods=1).std()

    # Range - max - min
    df["glucose_range_30min"] = (
        df["glucose"].rolling(window=6, min_periods=1).max()
        - df["glucose"].rolling(window=6, min_periods=1).min()
    )
    df["glucose_range_60min"] = (
        df["glucose"].rolling(window=12, min_periods=1).max()
        - df["glucose"].rolling(window=12, min_periods=1).min()
    )

    # Recent average (context)
    df["glucose_mean_60min"] = df["glucose"].rolling(window=12, min_periods=1).mean()
    df["glucose_mean_120min"] = df["glucose"].rolling(window=24, min_periods=1).mean()

    # ========================================
    # 3. TIME FEATURES
    # ========================================
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ========================================
    # 4. MEAL FEATURES
    # ========================================
    df["minutes_since_meal"] = np.nan
    df["carbs_from_last_meal"] = 0.0

    meal_rows = df[df["carbs"] > 0].copy()

    for idx in df.index:
        current_time = df.loc[idx, "timestamp"]
        past_meals = meal_rows[meal_rows["timestamp"] <= current_time]

        if len(past_meals) > 0:
            last_meal = past_meals.iloc[-1]
            actual_meal_time = last_meal["meal_timestamp"]
            time_diff = (current_time - actual_meal_time).total_seconds() / 60

            df.loc[idx, "minutes_since_meal"] = time_diff
            df.loc[idx, "carbs_from_last_meal"] = last_meal["carbs"]

    df["minutes_since_meal"] = df["minutes_since_meal"].fillna(999)
    df["had_meal_recently"] = (df["minutes_since_meal"] < 60).astype(int)
    df["had_meal_very_recently"] = (df["minutes_since_meal"] < 30).astype(int)

    # Rolling carb windows
    df["carbs_last_30min"] = df["carbs"].rolling(window=6, min_periods=1).sum()
    df["carbs_last_60min"] = df["carbs"].rolling(window=12, min_periods=1).sum()
    df["carbs_last_120min"] = df["carbs"].rolling(window=24, min_periods=1).sum()

    # High carb meal indicator
    df["high_carb_meal"] = (df["carbs_from_last_meal"] > 50).astype(int)

    # ========================================
    # 5. INSULIN FEATURES
    # ========================================
    df["minutes_since_bolus"] = np.nan
    df["dose_from_last_bolus"] = 0.0

    bolus_rows = df[df["bolus_dose"] > 0].copy()

    for idx in df.index:
        current_time = df.loc[idx, "timestamp"]
        past_bolus = bolus_rows[bolus_rows["timestamp"] <= current_time]

        if len(past_bolus) > 0:
            last_bolus = past_bolus.iloc[-1]
            actual_bolus_time = last_bolus["bolus_timestamp"]
            time_diff = (current_time - actual_bolus_time).total_seconds() / 60

            df.loc[idx, "minutes_since_bolus"] = time_diff
            df.loc[idx, "dose_from_last_bolus"] = last_bolus["bolus_dose"]

    df["minutes_since_bolus"] = df["minutes_since_bolus"].fillna(999)

    # Rolling bolus doses
    df["bolus_last_60min"] = df["bolus_dose"].rolling(window=12, min_periods=1).sum()
    df["bolus_last_120min"] = df["bolus_dose"].rolling(window=24, min_periods=1).sum()

    # ========================================
    # 6. INTERACTION FEATURES
    # ========================================
    df["meal_no_insulin"] = (
        (df["minutes_since_meal"] < 60) & (df["minutes_since_bolus"] > 60)
    ).astype(int)

    df["high_carb_and_rising"] = (
        (df["carbs_from_last_meal"] > 50) & (df["glucose_rising"] == 1)
    ).astype(int)

    df["fast_rise_after_meal"] = (
        (df["glucose_velocity_15min"] > 2) & (df["minutes_since_meal"] < 90)
    ).astype(int)

    df["accelerating_rise"] = (
        (df["glucose_accel_15min"] > 0.1) & (df["glucose_rising"] == 1)
    ).astype(int)

    # ========================================
    # 7. EXERCISE FEATURES
    # ========================================
    df["is_exercising"] = (df["exercise_duration"] > 0).astype(int)
    df["exercise_last_60min"] = df["exercise_duration"].rolling(
        window=12, min_periods=1
    ).max()
    df["exercise_last_120min"] = df["exercise_duration"].rolling(
        window=24, min_periods=1
    ).max()

    # ========================================
    # 8. BASELINE CONTEXT
    # ========================================
    df["glucose_baseline"] = df["glucose_mean_120min"]

    # ========================================
    # 9. CREATE TARGET VARIABLE
    # ========================================
    horizon_rows = prediction_horizon_minutes // 5

    df["future_glucose_max"] = (
        df["glucose"].shift(-horizon_rows).rolling(window=horizon_rows, min_periods=1).max()
    )
    df["will_spike"] = (df["future_glucose_max"] > spike_threshold).astype(int)
    df["future_glucose_rise"] = df["future_glucose_max"] - df["glucose"]

    return df


class CGMFeatureEngineer:
    """Feature engineering for CGM spike prediction."""

    def __init__(
        self,
        spike_threshold: float = SPIKE_THRESHOLD,
        prediction_horizon_minutes: int = PREDICTION_HORIZON_MINUTES,
    ):
        self.spike_threshold = spike_threshold
        self.prediction_horizon_minutes = prediction_horizon_minutes

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw CGM data."""
        print("Creating features...")

        data = df.copy()
        data = _coerce_timestamps(data)

        data = data.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        features_list = []
        for patient_id in data["patient_id"].unique():
            print(f"  Processing patient {patient_id}...")
            patient_data = data[data["patient_id"] == patient_id].copy()
            patient_features = engineer_features_for_patient(
                patient_data,
                spike_threshold=self.spike_threshold,
                prediction_horizon_minutes=self.prediction_horizon_minutes,
            )
            features_list.append(patient_features)

        features = pd.concat(features_list, ignore_index=True)
        print(f"Created {len(features.columns)} features for {len(features)} samples")
        return features

    def prepare_for_training(
        self, df: pd.DataFrame, drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.

        Args:
            df: DataFrame with engineered features
            drop_na: Whether to drop rows with NaN values in target

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        exclude_cols = [
            "patient_id",
            "timestamp",
            "glucose",
            "meal_type",
            "carbs",
            "bolus_dose",
            "exercise_duration",
            "exercise_intensity",
            "will_spike",
            "future_glucose_max",
            "future_glucose_rise",
            "meal_timestamp",
            "finger_stick_timestamp",
            "bolus_timestamp",
            "exercise_timestamp",
            "hypo_timestamp",
            "stress_timestamp",
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df["will_spike"].copy()

        X = X.fillna(0)

        if drop_na:
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]

        print(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        print(f"  Spike class distribution: {y.value_counts().to_dict()}")

        return X, y


def build_feature_dataframe(
    df: pd.DataFrame,
    spike_threshold: float = SPIKE_THRESHOLD,
    prediction_horizon_minutes: int = PREDICTION_HORIZON_MINUTES,
) -> pd.DataFrame:
    """Convenience wrapper to create engineered features from raw CGM data."""
    fe = CGMFeatureEngineer(
        spike_threshold=spike_threshold,
        prediction_horizon_minutes=prediction_horizon_minutes,
    )
    return fe.create_features(df)


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 70)

    train_df = pd.read_csv("data/processed/train_data.csv")

    fe = CGMFeatureEngineer()

    features_df = fe.create_features(train_df)

    X, y = fe.prepare_for_training(features_df)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nFeature statistics:")
    print(X.describe())

    print("\nSample features:")
    print(X.head(10))

    print("\nTarget distribution:")
    print(f"No spike (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"Will spike (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
