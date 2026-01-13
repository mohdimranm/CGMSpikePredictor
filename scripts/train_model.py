"""
Train CGM spike prediction model using OhioT1DM dataset.

This script:
1. Loads processed data
2. Creates features
3. Trains baseline and tree-based models
4. Evaluates performance
5. Saves the best model
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import joblib

from app.feature_engineering import CGMFeatureEngineer


def train_and_evaluate_models():
    """Train and evaluate spike prediction models."""

    print("=" * 80)
    print("CGM SPIKE PREDICTION MODEL TRAINING")
    print("=" * 80)

    # 1. Load processed data
    print("\n1. Loading processed data...")
    print("-" * 80)
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    print(f"✓ Training data: {len(train_df)} records")
    print(f"✓ Testing data: {len(test_df)} records")

    # 2. Create features
    print("\n2. Creating features...")
    print("-" * 80)
    fe = CGMFeatureEngineer(
        spike_threshold=180.0,  # Glucose > 180 mg/dL is considered a spike
        prediction_horizon_minutes=120  # Predict 2 hours ahead
    )

    # Engineer features for training data
    train_features = fe.create_features(train_df)
    X_train_full, y_train_full = fe.prepare_for_training(train_features)

    # Engineer features for test data
    test_features = fe.create_features(test_df)
    X_test, y_test = fe.prepare_for_training(test_features)

    # Split training data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Features: {len(X_train.columns)}")

    # 3. Train models
    print("\n3. Training models...")
    print("-" * 80)

    models = {}

    # Baseline: Logistic Regression
    print("\nTraining Logistic Regression (baseline)...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    models['logistic_regression'] = lr_model
    print("✓ Logistic Regression trained")

    # Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    print("✓ Random Forest trained")

    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models['gradient_boosting'] = gb_model
    print("✓ Gradient Boosting trained")

    # 4. Evaluate models
    print("\n4. Evaluating models...")
    print("-" * 80)

    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 40}")
        print(f"{name.upper().replace('_', ' ')}")
        print(f"{'=' * 40}")

        # Validation set performance
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        val_pr_auc = average_precision_score(y_val, y_val_proba)

        print(f"\nValidation Performance:")
        print(f"  ROC-AUC: {val_roc_auc:.4f}")
        print(f"  PR-AUC:  {val_pr_auc:.4f}")
        print(f"\nClassification Report (Validation):")
        print(classification_report(y_val, y_val_pred, target_names=['No Spike', 'Spike']))

        # Test set performance
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        test_pr_auc = average_precision_score(y_test, y_test_proba)

        print(f"\nTest Performance:")
        print(f"  ROC-AUC: {test_roc_auc:.4f}")
        print(f"  PR-AUC:  {test_pr_auc:.4f}")
        print(f"\nClassification Report (Test):")
        print(classification_report(y_test, y_test_pred, target_names=['No Spike', 'Spike']))

        print(f"\nConfusion Matrix (Test):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")

        # Store results
        results[name] = {
            'model': model,
            'val_roc_auc': val_roc_auc,
            'val_pr_auc': val_pr_auc,
            'test_roc_auc': test_roc_auc,
            'test_pr_auc': test_pr_auc
        }

    # 5. Select best model
    print("\n5. Model Selection...")
    print("-" * 80)

    best_model_name = max(results.keys(), key=lambda k: results[k]['test_roc_auc'])
    best_model = results[best_model_name]['model']

    print(f"\nBest model: {best_model_name.upper().replace('_', ' ')}")
    print(f"  Test ROC-AUC: {results[best_model_name]['test_roc_auc']:.4f}")
    print(f"  Test PR-AUC: {results[best_model_name]['test_pr_auc']:.4f}")

    # 6. Feature importance (if tree-based model)
    if hasattr(best_model, 'feature_importances_'):
        print("\nTop 15 Most Important Features:")
        print("-" * 80)
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in importances.head(15).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # 7. Save models
    print("\n6. Saving models...")
    print("-" * 80)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save best model
    best_model_path = models_dir / "spike_predictor.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"✓ Best model saved: {best_model_path}")

    # Save feature engineer
    fe_path = models_dir / "feature_engineer.pkl"
    joblib.dump(fe, fe_path)
    print(f"✓ Feature engineer saved: {fe_path}")

    # Save feature names
    feature_names_path = models_dir / "feature_names.txt"
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X_train.columns))
    print(f"✓ Feature names saved: {feature_names_path}")

    # Save model metadata
    metadata = {
        'model_type': best_model_name,
        'spike_threshold': fe.spike_threshold,
        'prediction_horizon_minutes': fe.prediction_horizon_minutes,
        'n_features': len(X_train.columns),
        'test_roc_auc': results[best_model_name]['test_roc_auc'],
        'test_pr_auc': results[best_model_name]['test_pr_auc'],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    metadata_path = models_dir / "model_metadata.txt"
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Metadata saved: {metadata_path}")

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModels saved to: {models_dir.absolute()}")
    print("  - spike_predictor.pkl (trained model)")
    print("  - feature_engineer.pkl (feature engineering pipeline)")
    print("  - feature_names.txt (list of features)")
    print("  - model_metadata.txt (model information)")

    return best_model, fe, results


if __name__ == "__main__":
    model, fe, results = train_and_evaluate_models()