"""Run end-to-end verification scenarios and save report assets."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from image_model import create_mask_preview, create_overlay_image, extract_image_features
from yolo_weed_detector import detect_weeds_with_pretrained_model


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "report_assets"
ASSET_DIR.mkdir(exist_ok=True)
SOURCE_IMAGE_PATH = ROOT / "OIP.jpeg"
RESULTS_PATH = ASSET_DIR / "verification_results.json"
SUMMARY_PATH = ASSET_DIR / "verification_summary.txt"


def _load_source_image() -> Image.Image:
    return Image.open(SOURCE_IMAGE_PATH).convert("RGB")


def _result_to_dict(case_name: str, result) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "status": result.status,
        "headline": result.headline,
        "confidence": round(float(result.confidence), 3),
        "weed_count": int(result.weed_count),
        "crop_count": int(result.crop_count),
        "total_detections": int(result.total_detections),
        "notes": list(result.notes),
        "model_path": result.model_path,
        "model_source": result.model_source,
        "original_size": list(result.original_size),
        "processed_size": list(result.processed_size),
        "auto_upscaled": bool(result.auto_upscaled),
        "detections": result.detections_table.to_dict("records"),
    }


def _fit_for_panel(image: Image.Image, panel_size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), panel_size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", panel_size, "#f4f6f7")
    offset_x = (panel_size[0] - fitted.width) // 2
    offset_y = (panel_size[1] - fitted.height) // 2
    panel.paste(fitted, (offset_x, offset_y))
    return panel


def _draw_labeled_panel(
    canvas: Image.Image,
    origin: tuple[int, int],
    label: str,
    image: Image.Image,
    panel_size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = origin
    draw.rounded_rectangle((x, y, x + panel_size[0], y + panel_size[1] + 34), radius=20, fill="#ffffff", outline="#d0d7de")
    draw.text((x + 14, y + 10), label, fill="#1b4332")
    panel_image = _fit_for_panel(image, panel_size)
    canvas.paste(panel_image, (x, y + 34))


def _compose_output_panel(source_image: Image.Image, detection_result, target_path: Path) -> None:
    feature_values, vegetation_mask, resized_image = extract_image_features(source_image)
    mask_image = create_mask_preview(vegetation_mask)
    overlay_image = create_overlay_image(resized_image, vegetation_mask)
    annotated_image = detection_result.annotated_image or resized_image

    canvas = Image.new("RGB", (1500, 1180), "#f5f7f6")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((40, 32, 1460, 165), radius=28, fill="#133c2c")
    draw.text((72, 62), "Program Output After Running the Final Weed Detector", fill="#ffffff")
    draw.text(
        (72, 100),
        (
            f"Status: {detection_result.status}    Headline: {detection_result.headline}    "
            f"Confidence: {detection_result.confidence * 100:.1f}%"
        ),
        fill="#d8efe2",
    )
    draw.text(
        (72, 130),
        (
            f"Original size: {detection_result.original_size[0]} x {detection_result.original_size[1]} px    "
            f"Processed size: {detection_result.processed_size[0]} x {detection_result.processed_size[1]} px    "
            f"Auto-upscaled: {'Yes' if detection_result.auto_upscaled else 'No'}"
        ),
        fill="#d8efe2",
    )

    panel_size = (640, 320)
    _draw_labeled_panel(canvas, (60, 210), "Original uploaded image", source_image, panel_size)
    _draw_labeled_panel(canvas, (800, 210), "YOLO detection output", annotated_image, panel_size)
    _draw_labeled_panel(canvas, (60, 610), "Vegetation mask", mask_image, panel_size)
    _draw_labeled_panel(canvas, (800, 610), "Analytics overlay", overlay_image, panel_size)

    draw.rounded_rectangle((60, 980, 1460, 1120), radius=22, fill="#ffffff", outline="#d0d7de")
    summary_lines = [
        f"Weed detections: {detection_result.weed_count}",
        f"Crop detections: {detection_result.crop_count}",
        f"Total detections: {detection_result.total_detections}",
        "Notes:",
    ]
    summary_lines.extend(f"- {note}" for note in detection_result.notes[:3])
    summary_lines.append(
        "Analytics: "
        f"vegetation_coverage={feature_values['vegetation_coverage']}, "
        f"texture_variation={feature_values['texture_variation']}, "
        f"row_alignment={feature_values['row_alignment']}"
    )
    draw.multiline_text((88, 1012), "\n".join(summary_lines), fill="#22313f", spacing=8)
    canvas.save(target_path, quality=95)


def _save_image(image: Image.Image, target_path: Path) -> None:
    image.save(target_path, quality=95)


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

    source_image = _load_source_image()
    larger_positive_image = source_image.resize((738, 440), Image.Resampling.LANCZOS)
    small_accepted_image = source_image.copy()
    invalid_dark_image = ImageEnhance.Brightness(source_image).enhance(0.08)

    positive_result = detect_weeds_with_pretrained_model(larger_positive_image)
    small_result = detect_weeds_with_pretrained_model(small_accepted_image)
    invalid_result = detect_weeds_with_pretrained_model(invalid_dark_image)

    if positive_result.annotated_image is not None:
        _save_image(positive_result.annotated_image, ASSET_DIR / "runtime_positive_detection.png")
        _save_image(positive_result.annotated_image, ASSET_DIR / "yolo_detection_example.png")
    if small_result.annotated_image is not None:
        _save_image(small_result.annotated_image, ASSET_DIR / "runtime_auto_upscaled_detection.png")
    _save_image(invalid_dark_image, ASSET_DIR / "runtime_invalid_dark_input.png")

    _compose_output_panel(small_accepted_image, small_result, ASSET_DIR / "sample_program_output.png")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "positive_large": _result_to_dict("positive_large", positive_result),
        "small_auto_upscaled": _result_to_dict("small_auto_upscaled", small_result),
        "invalid_dark": _result_to_dict("invalid_dark", invalid_result),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = [
        f"Generated at: {payload['generated_at']}",
        "",
        "Positive large-image run:",
        json.dumps(payload["positive_large"], indent=2),
        "",
        "Small accepted image run:",
        json.dumps(payload["small_auto_upscaled"], indent=2),
        "",
        "Invalid dark-image run:",
        json.dumps(payload["invalid_dark"], indent=2),
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Saved verification assets:")
    for asset_name in [
        "runtime_positive_detection.png",
        "runtime_auto_upscaled_detection.png",
        "runtime_invalid_dark_input.png",
        "sample_program_output.png",
        "verification_results.json",
        "verification_summary.txt",
    ]:
        print(f"- {asset_name}")


if __name__ == "__main__":
    main()
