"""Pretrained weed detection using a remote YOLO model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image

from image_model import compute_excess_green


MODELS_DIR = Path("models")
DEFAULT_MODEL_URL = os.getenv(
    "WEED_MODEL_URL",
    "https://huggingface.co/NvMayMay/weedblaster-vision-yolov8s/resolve/main/best.pt",
)
DEFAULT_MODEL_PATH = Path(os.getenv("WEED_MODEL_PATH", MODELS_DIR / "weedblaster-vision-yolov8s.pt"))
DEFAULT_CONFIDENCE = float(os.getenv("WEED_MODEL_CONFIDENCE", "0.25"))
DEFAULT_IMAGE_SIZE = int(os.getenv("WEED_MODEL_IMAGE_SIZE", "960"))
ABSOLUTE_MIN_DIMENSION = int(os.getenv("WEED_MODEL_ABSOLUTE_MIN_DIMENSION", "128"))
PREFERRED_MIN_DIMENSION = int(os.getenv("WEED_MODEL_PREFERRED_MIN_DIMENSION", "224"))


@dataclass
class FieldImageValidation:
    is_valid: bool
    issues: list[str]
    notes: list[str]
    prepared_image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    auto_upscaled: bool


@dataclass
class PretrainedDetectionResult:
    status: str
    weed_present: bool | None
    headline: str
    confidence: float
    weed_count: int
    crop_count: int
    total_detections: int
    annotated_image: Image.Image | None
    detections_table: pd.DataFrame
    notes: list[str]
    model_path: str
    model_source: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    auto_upscaled: bool


def get_pretrained_model_status() -> dict[str, Any]:
    model_path = DEFAULT_MODEL_PATH
    return {
        "model_path": str(model_path.resolve()),
        "local_exists": model_path.exists(),
        "model_source": DEFAULT_MODEL_URL,
        "absolute_min_dimension": ABSOLUTE_MIN_DIMENSION,
        "preferred_min_dimension": PREFERRED_MIN_DIMENSION,
    }


def _configure_yolo_environment() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd()))


def _prepare_image_rgb(
    image: Image.Image,
    max_side: int = 1280,
    min_side: int | None = None,
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = 1.0
    if min_side is not None and min(width, height) < min_side:
        scale = max(scale, min_side / min(width, height))
    if max(width, height) * scale > max_side:
        scale = max_side / max(width, height)
    if abs(scale - 1.0) < 1e-6:
        return image
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def validate_field_image(image: Image.Image) -> FieldImageValidation:
    original = image.convert("RGB")
    original_size = original.size
    original_width, original_height = original_size
    auto_upscaled = False
    notes: list[str] = []
    issues: list[str] = []

    if min(original_width, original_height) < ABSOLUTE_MIN_DIMENSION:
        issues.append(
            f"Image resolution is too low. Use a field image whose shorter side is at least {ABSOLUTE_MIN_DIMENSION} pixels."
        )
        prepared = _prepare_image_rgb(original, max_side=640)
        return FieldImageValidation(
            is_valid=False,
            issues=issues,
            notes=notes,
            prepared_image=prepared,
            original_size=original_size,
            processed_size=prepared.size,
            auto_upscaled=False,
        )

    prepared = _prepare_image_rgb(original, max_side=640, min_side=PREFERRED_MIN_DIMENSION)
    width, height = prepared.size
    aspect_ratio = width / max(height, 1)

    auto_upscaled = prepared.size != original_size and min(original_width, original_height) < PREFERRED_MIN_DIMENSION
    if auto_upscaled:
        notes.append(
            "Image was auto-upscaled from "
            f"{original_width} x {original_height} to {prepared.size[0]} x {prepared.size[1]} before analysis."
        )

    if aspect_ratio < 0.45 or aspect_ratio > 2.8:
        issues.append("Image shape is unusual. Use a standard portrait or landscape crop-field image.")

    rgb = np.asarray(prepared).astype(np.float32) / 255.0
    brightness = float(rgb.mean())
    if brightness < 0.16:
        issues.append("The image is too dark. Use a brighter daylight field image.")
    if brightness > 0.94:
        issues.append("The image is overexposed. Use a photo with more balanced lighting.")

    exg = compute_excess_green(rgb)
    vegetation_ratio = float((exg > 0.08).mean())
    if vegetation_ratio < 0.03:
        issues.append("Very little vegetation is visible. Upload a crop-field image that clearly shows plants.")
    if vegetation_ratio > 0.97:
        issues.append("The image is too close to the leaves. Upload a wider field view that shows crop rows.")

    grayscale = (0.299 * rgb[:, :, 0]) + (0.587 * rgb[:, :, 1]) + (0.114 * rgb[:, :, 2])
    if grayscale.shape[0] > 1 and grayscale.shape[1] > 1:
        texture = float(
            (
                np.abs(np.diff(grayscale, axis=0)).mean()
                + np.abs(np.diff(grayscale, axis=1)).mean()
            )
            / 2.0
        )
        if texture < 0.02:
            issues.append("The image appears too blurred or flat. Use a clearer field photo.")

    return FieldImageValidation(
        is_valid=not issues,
        issues=issues,
        notes=notes,
        prepared_image=prepared,
        original_size=original_size,
        processed_size=prepared.size,
        auto_upscaled=auto_upscaled,
    )


def ensure_pretrained_model() -> Path:
    model_path = DEFAULT_MODEL_PATH
    if model_path.exists():
        return model_path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(model_path.suffix + ".part")

    with requests.get(DEFAULT_MODEL_URL, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with temp_path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

    temp_path.replace(model_path)
    return model_path


@lru_cache(maxsize=1)
def load_pretrained_model():
    _configure_yolo_environment()
    from ultralytics import YOLO

    model_path = ensure_pretrained_model()
    return YOLO(str(model_path))


def _boxes_to_table(result) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    boxes = result.boxes
    names = result.names

    if boxes is None:
        return pd.DataFrame(columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])

    xyxy_values = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
    confidence_values = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
    class_values = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []

    for box, confidence, class_id in zip(xyxy_values, confidence_values, class_values):
        label = str(names.get(int(class_id), class_id))
        records.append(
            {
                "Class": label,
                "Confidence": round(float(confidence), 3),
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
            }
        )

    return pd.DataFrame(records)


def _summarize_detections(detections_table: pd.DataFrame) -> tuple[int, int]:
    if detections_table.empty:
        return 0, 0

    weed_mask = detections_table["Class"].str.lower().str.contains("weed", na=False)
    weed_count = int(weed_mask.sum())
    crop_count = int((~weed_mask).sum())
    return weed_count, crop_count


def detect_weeds_with_pretrained_model(
    image: Image.Image,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> PretrainedDetectionResult:
    validation = validate_field_image(image)
    if not validation.is_valid:
        return PretrainedDetectionResult(
            status="invalid",
            weed_present=None,
            headline="Image not accepted for reliable analysis",
            confidence=0.0,
            weed_count=0,
            crop_count=0,
            total_detections=0,
            annotated_image=None,
            detections_table=pd.DataFrame(columns=["Class", "Confidence", "x1", "y1", "x2", "y2"]),
            notes=[*validation.notes, *validation.issues],
            model_path=str(DEFAULT_MODEL_PATH.resolve()),
            model_source=DEFAULT_MODEL_URL,
            original_size=validation.original_size,
            processed_size=validation.processed_size,
            auto_upscaled=validation.auto_upscaled,
        )

    try:
        model = load_pretrained_model()
        prepared_image = _prepare_image_rgb(image, max_side=max(image_size, 640), min_side=PREFERRED_MIN_DIMENSION)
        result = model.predict(
            source=np.asarray(prepared_image),
            conf=confidence_threshold,
            imgsz=image_size,
            device="cpu",
            verbose=False,
        )[0]
    except Exception as exc:
        return PretrainedDetectionResult(
            status="error",
            weed_present=None,
            headline="Pretrained detector unavailable",
            confidence=0.0,
            weed_count=0,
            crop_count=0,
            total_detections=0,
            annotated_image=None,
            detections_table=pd.DataFrame(columns=["Class", "Confidence", "x1", "y1", "x2", "y2"]),
            notes=[f"YOLO model could not run: {exc}"],
            model_path=str(DEFAULT_MODEL_PATH.resolve()),
            model_source=DEFAULT_MODEL_URL,
            original_size=validation.original_size,
            processed_size=prepared_image.size,
            auto_upscaled=validation.auto_upscaled,
        )

    detections_table = _boxes_to_table(result)
    weed_count, crop_count = _summarize_detections(detections_table)
    total_detections = len(detections_table)

    plotted = result.plot()
    annotated_image = Image.fromarray(plotted[:, :, ::-1]) if plotted is not None else None

    if weed_count > 0:
        headline = "Weeds detected in the uploaded field image"
        top_confidence = float(
            detections_table[detections_table["Class"].str.lower().str.contains("weed", na=False)]["Confidence"].max()
        )
        notes = [
            *validation.notes,
            f"{weed_count} weed detection(s) were found by the pretrained model.",
            "Bounding boxes highlight the plant regions predicted as weeds.",
        ]
        return PretrainedDetectionResult(
            status="ok",
            weed_present=True,
            headline=headline,
            confidence=top_confidence,
            weed_count=weed_count,
            crop_count=crop_count,
            total_detections=total_detections,
            annotated_image=annotated_image,
            detections_table=detections_table,
            notes=notes,
            model_path=str(DEFAULT_MODEL_PATH.resolve()),
            model_source=DEFAULT_MODEL_URL,
            original_size=validation.original_size,
            processed_size=prepared_image.size,
            auto_upscaled=validation.auto_upscaled,
        )

    if crop_count > 0:
        top_confidence = float(detections_table["Confidence"].max())
        return PretrainedDetectionResult(
            status="ok",
            weed_present=False,
            headline="No weed boxes were detected in this field image",
            confidence=top_confidence,
            weed_count=0,
            crop_count=crop_count,
            total_detections=total_detections,
            annotated_image=annotated_image,
            detections_table=detections_table,
            notes=[
                *validation.notes,
                f"{crop_count} crop detection(s) were found and no weeds crossed the confidence threshold.",
                "If this result seems wrong, retake the photo with a wider top-down field view.",
            ],
            model_path=str(DEFAULT_MODEL_PATH.resolve()),
            model_source=DEFAULT_MODEL_URL,
            original_size=validation.original_size,
            processed_size=prepared_image.size,
            auto_upscaled=validation.auto_upscaled,
        )

    return PretrainedDetectionResult(
        status="ok",
        weed_present=None,
        headline="No confident plant detections were found",
        confidence=0.0,
        weed_count=0,
        crop_count=0,
        total_detections=0,
        annotated_image=annotated_image,
        detections_table=detections_table,
        notes=[
            *validation.notes,
            "The model did not find confident crop or weed boxes in this image.",
            "Use a clearer, wider field image that shows visible plants and row structure.",
        ],
        model_path=str(DEFAULT_MODEL_PATH.resolve()),
        model_source=DEFAULT_MODEL_URL,
        original_size=validation.original_size,
        processed_size=prepared_image.size,
        auto_upscaled=validation.auto_upscaled,
    )
