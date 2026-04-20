"""Submission-ready weed detection demo using Random Forest.

This script is designed for academic submission when only a synthetic dataset
is available. The synthetic data is generated to reflect agricultural feature
names more honestly than renaming generic `make_classification` columns.

What the script does:
1. Generates a reproducible, domain-inspired synthetic dataset.
2. Uses a stratified train/test split to preserve class balance.
3. Tunes a Random Forest using 5-fold cross-validation on the training set.
4. Evaluates the best model on a held-out test set.
5. Saves the trained model and feature importance table.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


RANDOM_STATE = 42
N_SAMPLES = 1000
TEST_SIZE = 0.20

FEATURE_COLUMNS = ["GreenIndex", "Texture", "Moisture", "Height", "NDVI"]
TARGET_COLUMN = "WeedPresent"

MODEL_PATH = Path("weed_rf_model.pkl")
IMPORTANCE_PATH = Path("feature_importance.csv")
DATA_PATH = Path("synthetic_weed_dataset.csv")


def generate_synthetic_weed_dataset(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a reproducible synthetic dataset with agriculturally named features.

    Class meanings:
    - 0: Crop
    - 1: Weed

    The feature distributions intentionally overlap so the task remains realistic
    instead of being perfectly separable.
    """

    rng = np.random.default_rng(random_state)

    # Rough field ratio: 60% crop, 40% weed.
    y = rng.choice([0, 1], size=n_samples, p=[0.60, 0.40])

    green_index = np.where(
        y == 0,
        rng.normal(loc=0.68, scale=0.11, size=n_samples),
        rng.normal(loc=0.40, scale=0.14, size=n_samples),
    )
    texture = np.where(
        y == 0,
        rng.normal(loc=0.95, scale=0.28, size=n_samples),
        rng.normal(loc=1.50, scale=0.35, size=n_samples),
    )
    moisture = np.where(
        y == 0,
        rng.normal(loc=0.58, scale=0.11, size=n_samples),
        rng.normal(loc=0.47, scale=0.14, size=n_samples),
    )
    height = np.where(
        y == 0,
        rng.normal(loc=29.0, scale=5.0, size=n_samples),
        rng.normal(loc=18.5, scale=5.5, size=n_samples),
    )
    ndvi = np.where(
        y == 0,
        rng.normal(loc=0.76, scale=0.10, size=n_samples),
        rng.normal(loc=0.32, scale=0.16, size=n_samples),
    )

    # Introduce mild correlations and realistic measurement noise.
    green_index = green_index + 0.10 * ndvi + rng.normal(0, 0.02, n_samples)
    moisture = moisture + 0.03 * green_index + rng.normal(0, 0.02, n_samples)
    height = height + 3.0 * ndvi + rng.normal(0, 1.2, n_samples)

    # Add a small amount of label noise so evaluation remains realistic.
    flip_mask = rng.random(n_samples) < 0.05
    y[flip_mask] = 1 - y[flip_mask]

    df = pd.DataFrame(
        {
            "GreenIndex": np.clip(green_index, 0.0, 1.0),
            "Texture": np.clip(texture, 0.2, 3.0),
            "Moisture": np.clip(moisture, 0.0, 1.0),
            "Height": np.clip(height, 5.0, 45.0),
            "NDVI": np.clip(ndvi, -0.2, 1.0),
            TARGET_COLUMN: y,
        }
    )

    return df


def build_model(random_state: int = RANDOM_STATE) -> GridSearchCV:
    """Configure cross-validated hyperparameter tuning for Random Forest."""

    estimator = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 8, 12],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def print_metrics(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Print the evaluation metrics used in the report."""

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    print("=== TEST SET PERFORMANCE ===")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-Score : {f1:.3f}\n")

    print("Confusion Matrix:")
    print(matrix)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Crop", "Weed"], digits=3))


def save_artifacts(model: RandomForestClassifier, feature_table: pd.DataFrame, df: pd.DataFrame) -> None:
    """Persist outputs for reuse in a notebook, demo, or viva."""

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_COLUMNS,
            "target_name": TARGET_COLUMN,
            "random_state": RANDOM_STATE,
        },
        MODEL_PATH,
    )
    feature_table.to_csv(IMPORTANCE_PATH, index=False)
    df.to_csv(DATA_PATH, index=False)


def main() -> None:
    print("=== WEED DETECTION WITH RANDOM FOREST ===\n")

    df = generate_synthetic_weed_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"Dataset shape: {df.shape}")
    print("Class distribution:", y.value_counts().sort_index().to_dict())
    print("\nNote: This is a synthetic, domain-inspired dataset for demonstration.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}\n")

    search = build_model()
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("Best hyperparameters:")
    print(search.best_params_)
    print(f"\nBest 5-Fold CV F1-Score: {search.best_score_:.3f}\n")

    y_pred = best_model.predict(X_test)
    print_metrics(y_test, y_pred)

    feature_table = (
        pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": best_model.feature_importances_,
            }
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    print("Feature Importances:")
    print(feature_table.to_string(index=False))

    save_artifacts(best_model, feature_table, df)
    print(
        f"\nSaved model to '{MODEL_PATH}', feature importances to '{IMPORTANCE_PATH}', "
        f"and dataset to '{DATA_PATH}'."
    )


if __name__ == "__main__":
    main()
