"""Image-based weed detection prototype built around Random Forest.

This module upgrades the earlier tabular demo into an RGB image workflow:
1. Accept a crop-field image.
2. Extract vegetation and texture cues from the image.
3. Feed the image-derived features into a Random Forest classifier.
4. Return a weed/no-weed decision, confidence score, and visual overlays.

Important:
- This is a prototype for project/demo use.
- The uploaded image is standard RGB, so true NDVI is not available.
- The classifier is trained on synthetic image-inspired features unless the
  user later replaces it with a real labeled dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "vegetation_coverage",
    "excess_green_mean",
    "excess_green_std",
    "texture_variation",
    "component_density",
    "row_alignment",
]
TARGET_COLUMN = "weed_present"

MODEL_BUNDLE_PATH = Path("weed_image_rf_model.joblib")
TRAINING_DATA_PATH = Path("synthetic_image_feature_data.csv")


@dataclass
class AnalysisResult:
    label: int
    label_text: str
    confidence: float
    feature_values: dict[str, float]
    feature_table: pd.DataFrame
    reasoning: list[str]
    resized_image: Image.Image
    vegetation_mask_image: Image.Image
    overlay_image: Image.Image


def _ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def resize_for_analysis(image: Image.Image, max_side: int = 640) -> Image.Image:
    image = _ensure_rgb(image)
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def compute_excess_green(rgb: np.ndarray) -> np.ndarray:
    """Compute ExG from normalized RGB values."""

    channel_sum = rgb.sum(axis=2, keepdims=True)
    channel_sum = np.where(channel_sum == 0, 1e-6, channel_sum)
    normalized = rgb / channel_sum

    red = normalized[:, :, 0]
    green = normalized[:, :, 1]
    blue = normalized[:, :, 2]
    return (2.0 * green) - red - blue


def build_vegetation_mask(rgb: np.ndarray, exg_threshold: float = 0.08) -> np.ndarray:
    """Segment vegetation using Excess Green and green dominance."""

    exg = compute_excess_green(rgb)
    green_dominance = (rgb[:, :, 1] > rgb[:, :, 0] * 0.95) & (rgb[:, :, 1] > rgb[:, :, 2] * 1.05)
    mask = (exg > exg_threshold) & green_dominance

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MedianFilter(size=3))
    return np.asarray(mask_image) > 0


def _downsample_mask(mask: np.ndarray, max_side: int = 128) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    width, height = mask_image.size
    scale = min(max_side / max(width, height), 1.0)
    resized = mask_image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.NEAREST,
    )
    return np.asarray(resized) > 0


def count_connected_components(mask: np.ndarray) -> int:
    """Count connected vegetation clusters on a small binary mask."""

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    component_count = 0

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue

            component_count += 1
            stack = [(row, col)]
            visited[row, col] = True

            while stack:
                current_row, current_col = stack.pop()
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue

                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                            continue
                        if visited[next_row, next_col] or not mask[next_row, next_col]:
                            continue

                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

    return component_count


def compute_row_alignment(mask: np.ndarray) -> float:
    """Estimate how strongly vegetation aligns along crop rows."""

    row_profile = mask.mean(axis=1)
    col_profile = mask.mean(axis=0)
    smoothing_window = np.ones(15, dtype=np.float32) / 15.0

    smoothed_rows = np.convolve(row_profile, smoothing_window, mode="same")
    smoothed_cols = np.convolve(col_profile, smoothing_window, mode="same")

    alignment = max(smoothed_rows.std(), smoothed_cols.std())
    return float(np.clip(alignment / 0.22, 0.0, 1.0))


def create_mask_preview(mask: np.ndarray) -> Image.Image:
    preview = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    preview[:, :, :] = np.array([238, 245, 240], dtype=np.uint8)
    preview[mask] = np.array([35, 132, 82], dtype=np.uint8)
    return Image.fromarray(preview, mode="RGB")


def create_overlay_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    highlight = np.array([255.0, 166.0, 77.0], dtype=np.float32)
    alpha = np.zeros_like(base, dtype=np.float32)
    alpha[mask] = 0.42
    blended = (base * (1.0 - alpha)) + (highlight * alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def extract_image_features(image: Image.Image) -> tuple[dict[str, float], np.ndarray, Image.Image]:
    """Extract engineered features from an RGB crop-field image."""

    resized = resize_for_analysis(image)
    rgb = np.asarray(resized).astype(np.float32) / 255.0
    mask = build_vegetation_mask(rgb)
    exg = compute_excess_green(rgb)

    grayscale = (0.299 * rgb[:, :, 0]) + (0.587 * rgb[:, :, 1]) + (0.114 * rgb[:, :, 2])
    vertical_diff = np.abs(np.diff(grayscale, axis=0))
    horizontal_diff = np.abs(np.diff(grayscale, axis=1))
    texture_variation = float((vertical_diff.mean() + horizontal_diff.mean()) / 2.0)

    vegetation_coverage = float(mask.mean())
    if mask.any():
        excess_green_mean = float(exg[mask].mean())
        excess_green_std = float(exg[mask].std())
    else:
        excess_green_mean = float(exg.mean())
        excess_green_std = float(exg.std())

    small_mask = _downsample_mask(mask)
    area_scale = max(1.0, small_mask.size / 10_000.0)
    component_density = float(count_connected_components(small_mask) / area_scale)
    row_alignment = compute_row_alignment(mask)

    features = {
        "vegetation_coverage": round(vegetation_coverage, 4),
        "excess_green_mean": round(excess_green_mean, 4),
        "excess_green_std": round(excess_green_std, 4),
        "texture_variation": round(texture_variation, 4),
        "component_density": round(component_density, 4),
        "row_alignment": round(row_alignment, 4),
    }
    return features, mask, resized


def generate_synthetic_image_feature_data(
    n_samples: int = 1800,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate synthetic image-feature training data for the Random Forest."""

    rng = np.random.default_rng(random_state)

    weed_pressure = rng.beta(2.2, 1.8, n_samples)
    vegetation_coverage = np.clip(rng.normal(0.12 + (0.42 * weed_pressure), 0.06, n_samples), 0.01, 0.95)
    excess_green_mean = np.clip(rng.normal(0.28 - (0.16 * weed_pressure), 0.04, n_samples), 0.01, 0.60)
    excess_green_std = np.clip(rng.normal(0.06 + (0.10 * weed_pressure), 0.02, n_samples), 0.01, 0.40)
    texture_variation = np.clip(rng.normal(0.04 + (0.18 * weed_pressure), 0.03, n_samples), 0.01, 0.50)
    component_density = np.clip(rng.normal(0.8 + (6.5 * weed_pressure), 1.0, n_samples), 0.0, 20.0)
    row_alignment = np.clip(rng.normal(0.82 - (0.62 * weed_pressure), 0.08, n_samples), 0.0, 1.0)

    noise = rng.normal(0.0, 0.09, n_samples)
    labels = ((weed_pressure + noise) > 0.52).astype(int)

    # Flip a few labels to avoid an overly clean boundary.
    flip_mask = rng.random(n_samples) < 0.04
    labels[flip_mask] = 1 - labels[flip_mask]

    return pd.DataFrame(
        {
            "vegetation_coverage": vegetation_coverage,
            "excess_green_mean": excess_green_mean,
            "excess_green_std": excess_green_std,
            "texture_variation": texture_variation,
            "component_density": component_density,
            "row_alignment": row_alignment,
            TARGET_COLUMN: labels,
        }
    ).round(4)


def train_image_classifier(
    save_bundle: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train a Random Forest on synthetic image-derived features."""

    dataset = generate_synthetic_image_feature_data(random_state=random_state)
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced",
        ),
        param_grid={
            "n_estimators": [120, 180],
            "max_depth": [8, 10, None],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", None],
        },
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    feature_importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": best_model.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    bundle = {
        "model": best_model,
        "feature_names": FEATURE_COLUMNS,
        "best_params": search.best_params_,
        "cv_f1": round(float(search.best_score_), 4),
        "holdout_metrics": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions)), 4),
            "recall": round(float(recall_score(y_test, predictions)), 4),
            "f1": round(float(f1_score(y_test, predictions)), 4),
        },
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, target_names=["No Weed", "Weed"], digits=3),
        "feature_importance": feature_importance,
        "training_data": dataset,
        "prediction_mean": float(np.mean(probabilities)),
    }

    if save_bundle:
        joblib.dump(bundle, MODEL_BUNDLE_PATH)
        dataset.to_csv(TRAINING_DATA_PATH, index=False)

    return bundle


def load_or_train_classifier() -> dict[str, Any]:
    if MODEL_BUNDLE_PATH.exists():
        return joblib.load(MODEL_BUNDLE_PATH)
    return train_image_classifier(save_bundle=True)


def explain_prediction(features: dict[str, float], label: int) -> list[str]:
    reasons: list[str] = []

    if features["component_density"] > 4.0:
        reasons.append("The vegetation mask is fragmented into many small patches, which often suggests weed growth between crop rows.")
    if features["row_alignment"] < 0.45:
        reasons.append("Vegetation is weakly aligned, so the scene looks less like orderly crop rows and more like scattered growth.")
    if features["vegetation_coverage"] > 0.30:
        reasons.append("Vegetation coverage is relatively high in this field view, which raises weed likelihood.")
    if features["texture_variation"] > 0.12:
        reasons.append("Texture variation is elevated, indicating a visually busy field surface rather than a uniform crop pattern.")

    if label == 0:
        if features["row_alignment"] >= 0.55:
            reasons.append("Vegetation appears more aligned with consistent crop-row structure, which lowers weed likelihood.")
        if features["component_density"] < 3.0:
            reasons.append("The vegetation pattern is not highly fragmented, which is more consistent with cleaner plantation rows.")

    if not reasons:
        reasons.append("The model prediction is based on the overall vegetation pattern extracted from the uploaded RGB image.")

    return reasons[:3]


def analyze_image(image: Image.Image, model_bundle: dict[str, Any] | None = None) -> AnalysisResult:
    """Predict weed presence from an uploaded image."""

    bundle = model_bundle or load_or_train_classifier()
    model: RandomForestClassifier = bundle["model"]

    features, mask, resized = extract_image_features(image)
    feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    probabilities = model.predict_proba(feature_frame)[0]
    label = int(model.predict(feature_frame)[0])
    confidence = float(probabilities[label])

    label_text = "Weeds likely present" if label == 1 else "No obvious weeds detected"
    reasoning = explain_prediction(features, label)

    feature_table = pd.DataFrame(
        {
            "Feature": [name.replace("_", " ").title() for name in FEATURE_COLUMNS],
            "Value": [features[name] for name in FEATURE_COLUMNS],
        }
    )

    return AnalysisResult(
        label=label,
        label_text=label_text,
        confidence=confidence,
        feature_values=features,
        feature_table=feature_table,
        reasoning=reasoning,
        resized_image=resized,
        vegetation_mask_image=create_mask_preview(mask),
        overlay_image=create_overlay_image(resized, mask),
    )
