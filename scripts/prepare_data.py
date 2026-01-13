"""
Script to load OhioT1DM dataset and save processed data for training.
This creates clean CSV files that can be loaded quickly for model training.
"""
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ohio_data_loader import OhioT1DMLoader
import pandas as pd


def prepare_and_save_data():
    """Load OhioT1DM data and save as CSV files."""

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize loader
    loader = OhioT1DMLoader(data_dir="data/archive")

    print("=" * 70)
    print("LOADING AND PROCESSING OHIO T1DM DATASET")
    print("=" * 70)

    # Load and save training data
    print("\n1. Processing TRAINING data...")
    print("-" * 70)
    train_df = loader.create_unified_dataset('training')
    train_output = output_dir / "train_data.csv"
    train_df.to_csv(train_output, index=False)
    print(f"✓ Saved training data: {train_output}")
    print(f"  - Shape: {train_df.shape}")
    print(f"  - Columns: {list(train_df.columns)}")
    print(f"  - Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")

    # Load and save testing data
    print("\n2. Processing TESTING data...")
    print("-" * 70)
    test_df = loader.create_unified_dataset('testing')
    test_output = output_dir / "test_data.csv"
    test_df.to_csv(test_output, index=False)
    print(f"✓ Saved testing data: {test_output}")
    print(f"  - Shape: {test_df.shape}")
    print(f"  - Columns: {list(test_df.columns)}")
    print(f"  - Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print("\nTraining Data Statistics:")
    print(train_df.describe())

    print("\nGlucose Distribution:")
    print(f"  Mean: {train_df['glucose'].mean():.2f} mg/dL")
    print(f"  Std:  {train_df['glucose'].std():.2f} mg/dL")
    print(f"  Min:  {train_df['glucose'].min():.2f} mg/dL")
    print(f"  Max:  {train_df['glucose'].max():.2f} mg/dL")

    print("\nCarbohydrate Intake:")
    meals_with_carbs = train_df[train_df['carbs'] > 0]
    print(f"  Total meals logged: {len(meals_with_carbs)}")
    print(f"  Average carbs per meal: {meals_with_carbs['carbs'].mean():.2f}g")
    print(f"  Max carbs in a meal: {meals_with_carbs['carbs'].max():.2f}g")

    print("\nInsulin Dosing:")
    bolus_events = train_df[train_df['bolus_dose'] > 0]
    print(f"  Total bolus events: {len(bolus_events)}")
    print(f"  Average bolus dose: {bolus_events['bolus_dose'].mean():.2f}u")
    print(f"  Max bolus dose: {bolus_events['bolus_dose'].max():.2f}u")

    print("\nExercise Events:")
    exercise_events = train_df[train_df['exercise_duration'] > 0]
    print(f"  Total exercise sessions: {len(exercise_events)}")
    if len(exercise_events) > 0:
        print(f"  Average duration: {exercise_events['exercise_duration'].mean():.2f} min")
        print(f"  Average intensity: {exercise_events['exercise_intensity'].mean():.2f}/10")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nProcessed files saved to: {output_dir.absolute()}")
    print("  - train_data.csv")
    print("  - test_data.csv")

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = prepare_and_save_data()