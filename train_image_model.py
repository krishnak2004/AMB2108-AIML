"""Train the image-based weed detection prototype model."""

from __future__ import annotations

from image_model import MODEL_BUNDLE_PATH, TRAINING_DATA_PATH, train_image_classifier


def main() -> None:
    bundle = train_image_classifier(save_bundle=True)

    print("=== IMAGE-BASED WEED DETECTION MODEL ===\n")
    print("Saved model bundle to:", MODEL_BUNDLE_PATH)
    print("Saved synthetic feature data to:", TRAINING_DATA_PATH)
    print("\nBest hyperparameters:")
    print(bundle["best_params"])
    print(f"\nBest CV F1-score: {bundle['cv_f1']:.4f}")
    print("\nHoldout metrics:")
    for metric_name, metric_value in bundle["holdout_metrics"].items():
        print(f"{metric_name.title():>10}: {metric_value:.4f}")
    print("\nConfusion matrix:")
    print(bundle["confusion_matrix"])
    print("\nClassification report:")
    print(bundle["classification_report"])
    print("\nFeature importance:")
    print(bundle["feature_importance"].to_string(index=False))


if __name__ == "__main__":
    main()
