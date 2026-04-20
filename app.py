"""Streamlit UI for pretrained weed detection plus analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from image_model import (
    analyze_image,
    create_mask_preview,
    create_overlay_image,
    extract_image_features,
    load_or_train_classifier,
)
from yolo_weed_detector import detect_weeds_with_pretrained_model, get_pretrained_model_status


st.set_page_config(page_title="Crop Field Weed Detector", layout="wide")


ARTICLE_LINKS = [
    ("DeepWeeds dataset and field-image baseline", "https://www.nature.com/articles/s41598-018-38343-3"),
    ("CropAndWeed dataset (WACV 2023)", "https://github.com/cropandweed/cropandweed-dataset"),
    ("CWFID crop/weed field image dataset", "https://github.com/cwfid/dataset"),
    ("Systematic review of weed detection with deep learning", "https://www.mdpi.com/1424-8220/23/7/3670"),
]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(204, 236, 219, 0.55), transparent 28%),
              radial-gradient(circle at bottom right, rgba(244, 227, 199, 0.45), transparent 32%),
              #f7f5ef;
          }
          .hero {
            background: linear-gradient(135deg, #153d2f, #2f6b53 62%, #d8b26a 140%);
            color: #f9fbfa;
            border-radius: 24px;
            padding: 30px 34px;
            margin-bottom: 20px;
            box-shadow: 0 20px 60px rgba(21, 61, 47, 0.18);
          }
          .hero h1 {
            font-size: 2.2rem;
            margin-bottom: 0.35rem;
          }
          .hero p {
            font-size: 1.02rem;
            max-width: 940px;
            margin: 0;
            line-height: 1.65;
          }
          .status-card {
            border-radius: 18px;
            padding: 20px 22px;
            margin-bottom: 14px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
          }
          .status-positive {
            background: linear-gradient(135deg, rgba(255, 244, 229, 0.96), rgba(255, 232, 205, 0.96));
          }
          .status-negative {
            background: linear-gradient(135deg, rgba(234, 246, 239, 0.96), rgba(220, 241, 229, 0.96));
          }
          .status-warning {
            background: linear-gradient(135deg, rgba(255, 248, 229, 0.96), rgba(255, 239, 204, 0.96));
          }
          .status-error {
            background: linear-gradient(135deg, rgba(254, 226, 226, 0.96), rgba(254, 242, 242, 0.96));
          }
          .status-card h2 {
            margin: 0 0 6px;
            font-size: 1.55rem;
          }
          .status-card p {
            margin: 0.25rem 0;
            color: #364152;
          }
          .source-list a {
            color: #1b6b49;
            text-decoration: none;
          }
          .source-list a:hover {
            text-decoration: underline;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    status = get_pretrained_model_status()

    st.sidebar.header("Detection Setup")
    st.sidebar.write(
        "This app uses a pretrained YOLO weed detector first and then shows vegetation-pattern analytics for explanation."
    )

    with st.sidebar.expander("Pretrained model", expanded=True):
        st.write(f"Local weights available: `{'Yes' if status['local_exists'] else 'No - first run downloads them'}`")
        st.caption(f"Model path: `{Path(status['model_path']).name}`")
        st.markdown(f"[Model source]({status['model_source']})")

    with st.sidebar.expander("Accepted image restrictions", expanded=True):
        st.write("- Clear crop-field images only")
        st.write("- PNG, JPG, JPEG, or WEBP")
        st.write("- Daylight images work best")
        st.write(
            f"- Shorter side must be at least {status['absolute_min_dimension']} px"
        )
        st.write(
            f"- Images between {status['absolute_min_dimension']} px and "
            f"{status['preferred_min_dimension'] - 1} px are auto-upscaled before inference"
        )
        st.write("- Avoid extreme close-ups, dark images, screenshots, and heavy blur")

    with st.sidebar.expander("Research links", expanded=False):
        st.markdown('<div class="source-list">', unsafe_allow_html=True)
        for label, url in ARTICLE_LINKS:
            st.markdown(f"- [{label}]({url})")
        st.markdown("</div>", unsafe_allow_html=True)


def render_status_card(title: str, confidence: float, description: str, card_class: str) -> None:
    st.markdown(
        f"""
        <div class="status-card {card_class}">
          <h2>{title}</h2>
          <p><strong>Confidence:</strong> {confidence * 100:.1f}%</p>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_notes(notes: list[str]) -> None:
    if not notes:
        return
    st.subheader("Detection notes")
    for note in notes:
        st.write(f"- {note}")


def render_analytics_panel(feature_values: dict[str, float], feature_table: pd.DataFrame) -> None:
    st.subheader("Vegetation analytics")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.metric("Vegetation coverage", f"{feature_values['vegetation_coverage']:.3f}")
        st.metric("Excess green mean", f"{feature_values['excess_green_mean']:.3f}")
    with metric_col_2:
        st.metric("Texture variation", f"{feature_values['texture_variation']:.3f}")
        st.metric("Component density", f"{feature_values['component_density']:.3f}")
    with metric_col_3:
        st.metric("Row alignment", f"{feature_values['row_alignment']:.3f}")
        st.metric("Excess green std", f"{feature_values['excess_green_std']:.3f}")

    st.dataframe(feature_table, use_container_width=True, hide_index=True)


def main() -> None:
    apply_theme()
    render_sidebar()

    st.markdown(
        """
        <div class="hero">
          <h1>Crop Field Weed Detector</h1>
          <p>
            Upload a clear crop-field image and the app will run a pretrained weed detector with bounding boxes.
            Alongside the final weed / no-weed result, the interface also shows vegetation-mask analytics so you can
            explain the decision in your project demo and report.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a crop-field image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Use a daylight field image with visible crop rows or plantation structure. Smaller usable images are auto-upscaled.",
    )

    if not uploaded_file:
        st.info(
            "No image uploaded yet. The first real run will download the pretrained detector weights and then analyze the image."
        )
        st.markdown(
            """
            **What this app shows**
            - Original image
            - YOLO detection view with bounding boxes
            - Vegetation mask
            - Overlay analytics view
            - Weed / no-weed decision
            - Detection table and image-feature summary
            - Auto-upscale note for smaller accepted images
            """
        )
        return

    image = Image.open(uploaded_file).convert("RGB")
    feature_values, vegetation_mask, resized_image = extract_image_features(image)
    vegetation_mask_image = create_mask_preview(vegetation_mask)
    overlay_image = create_overlay_image(resized_image, vegetation_mask)
    feature_table = pd.DataFrame(
        {
            "Feature": [name.replace("_", " ").title() for name in feature_values.keys()],
            "Value": list(feature_values.values()),
        }
    )

    with st.spinner("Running pretrained weed detector..."):
        detection_result = detect_weeds_with_pretrained_model(image)

    fallback_result = None
    if detection_result.status == "error":
        fallback_bundle = load_or_train_classifier()
        fallback_result = analyze_image(image, model_bundle=fallback_bundle)

    if detection_result.status == "invalid":
        render_status_card(
            detection_result.headline,
            0.0,
            "The uploaded file does not meet the current crop-field image acceptance rules.",
            "status-error",
        )
        render_detection_notes(detection_result.notes)
    elif detection_result.status == "error":
        render_status_card(
            "Pretrained detector not available",
            fallback_result.confidence if fallback_result else 0.0,
            "The app switched to the older vegetation-pattern fallback so you can still continue the demo.",
            "status-warning",
        )
        render_detection_notes(detection_result.notes)
        if fallback_result is not None:
            st.subheader("Fallback result")
            st.write(f"- {fallback_result.label_text}")
            for reason in fallback_result.reasoning:
                st.write(f"- {reason}")
    else:
        if detection_result.weed_present is True:
            render_status_card(
                detection_result.headline,
                detection_result.confidence,
                "At least one weed-class bounding box was found by the pretrained detector.",
                "status-positive",
            )
        elif detection_result.weed_present is False:
            render_status_card(
                detection_result.headline,
                detection_result.confidence,
                "Crop detections were found and no weed boxes crossed the confidence threshold.",
                "status-negative",
            )
        else:
            render_status_card(
                detection_result.headline,
                detection_result.confidence,
                "No reliable crop or weed boxes were found, so the image should be retaken for a stronger result.",
                "status-warning",
            )
        render_detection_notes(detection_result.notes)

    if detection_result.auto_upscaled:
        st.info(
            "This image was auto-upscaled before inference so the shorter side met the model's preferred minimum size. "
            f"Original size: {detection_result.original_size[0]} x {detection_result.original_size[1]} px. "
            f"Processed size: {detection_result.processed_size[0]} x {detection_result.processed_size[1]} px."
        )

    image_col_1, image_col_2 = st.columns(2)
    with image_col_1:
        st.subheader("Original image")
        st.image(resized_image, use_container_width=True)
    with image_col_2:
        st.subheader("YOLO detection view")
        if detection_result.annotated_image is not None:
            st.image(detection_result.annotated_image, use_container_width=True)
        elif fallback_result is not None:
            st.image(fallback_result.resized_image, use_container_width=True)
        else:
            st.info("No detection overlay available for this image.")

    image_col_3, image_col_4 = st.columns(2)
    with image_col_3:
        st.subheader("Vegetation mask")
        st.image(vegetation_mask_image, use_container_width=True)
    with image_col_4:
        st.subheader("Analytics overlay")
        st.image(overlay_image, use_container_width=True)

    st.subheader("Detection summary")
    summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
    with summary_col_1:
        st.metric("Weed detections", detection_result.weed_count)
    with summary_col_2:
        st.metric("Crop detections", detection_result.crop_count)
    with summary_col_3:
        st.metric(
            "Processed size",
            f"{detection_result.processed_size[0]} x {detection_result.processed_size[1]}",
        )
    with summary_col_4:
        st.metric("Top confidence", f"{detection_result.confidence * 100:.1f}%")

    info_col_1, info_col_2 = st.columns(2)
    with info_col_1:
        st.caption(
            f"Original upload size: {detection_result.original_size[0]} x {detection_result.original_size[1]} px"
        )
    with info_col_2:
        st.caption(
            "Auto-upscaled before inference" if detection_result.auto_upscaled else "Original size was already acceptable"
        )

    st.subheader("Detection table")
    if detection_result.detections_table.empty:
        st.info("No confident bounding boxes were produced for this image.")
    else:
        st.dataframe(detection_result.detections_table, use_container_width=True, hide_index=True)

    render_analytics_panel(feature_values, feature_table)

    with st.expander("How to run the project", expanded=False):
        st.write("1. Install dependencies with `pip install -r requirements.txt`.")
        st.write("2. Launch the interface with `streamlit run app.py`.")
        st.write("3. On the first detection run, the app downloads the pretrained model weights automatically.")
        st.write("4. Upload a crop-field image and review the detections plus analytics.")

    with st.expander("Open the detailed literature notes", expanded=False):
        literature_path = Path("LITERATURE_REVIEW.md")
        if literature_path.exists():
            st.markdown(literature_path.read_text(encoding="utf-8"))
        else:
            st.write("Literature notes file not found.")


if __name__ == "__main__":
    main()
