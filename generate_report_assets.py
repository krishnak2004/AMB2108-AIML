"""Generate report charts for the final project submission."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "report_assets"
ASSET_DIR.mkdir(exist_ok=True)

BUNDLE_PATH = ROOT / "weed_image_rf_model.joblib"


def save_feature_importance(bundle: dict) -> None:
    feature_table = bundle["feature_importance"]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(feature_table["feature"], feature_table["importance"], color=["#1b6b49", "#2f7f5f", "#4c9d75", "#6fb38d", "#8fc8a9", "#b7dbc6"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance for Image-Based Weed Detection")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(ASSET_DIR / "feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(bundle: dict) -> None:
    matrix = np.array(bundle["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    heatmap = ax.imshow(matrix, cmap="Greens")
    plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=["Predicted No Weed", "Predicted Weed"])
    ax.set_yticks([0, 1], labels=["Actual No Weed", "Actual Weed"])
    ax.set_title("Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="#102a1d", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(ASSET_DIR / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metric_chart(bundle: dict) -> None:
    metrics = bundle["holdout_metrics"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(labels, values, color=["#355f4a", "#4c8a67", "#63a17d", "#7bb896"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance on Holdout Set")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + (bar.get_width() / 2), value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(ASSET_DIR / "model_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bundle = joblib.load(BUNDLE_PATH)
    save_feature_importance(bundle)
    save_confusion_matrix(bundle)
    save_metric_chart(bundle)

    print("Generated assets:")
    for asset_path in sorted(ASSET_DIR.glob("*")):
        print("-", asset_path.name)


if __name__ == "__main__":
    main()
