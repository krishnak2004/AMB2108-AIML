"""Build the final submission DOCX directly from the sample template."""

from __future__ import annotations

import html
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "report_assets"
RESULTS_PATH = ASSET_DIR / "verification_results.json"
TEMPLATE_PATH = Path(r"C:\Users\Shreya\Downloads\Sample report format (1).docx")
OUTPUT_PATHS = [
    ROOT / "Weed_Detection_Final_Submission.docx",
    ROOT / "Final_Submission_Report.docx",
]

IMAGE_ASSETS = [
    ("project_workflow.png", "Workflow of the final weed detection system"),
    ("system_architecture.png", "System architecture of the final weed detection system"),
    ("browser_ui_upload.png", "Lightweight browser demo interface"),
    ("sample_program_output.png", "Primary Streamlit interface showing executed output"),
    ("runtime_auto_upscaled_detection.png", "Annotated detector output from the smaller accepted image"),
    ("yolo_detection_example.png", "Primary YOLO output from the larger accepted image"),
    ("runtime_invalid_dark_input.png", "Dark invalid input used to verify rejection behavior"),
    ("model_metrics.png", "Secondary Random Forest analytics metrics"),
    ("confusion_matrix.png", "Secondary Random Forest confusion matrix"),
    ("feature_importance.png", "Secondary Random Forest feature importance"),
]

PYTHON_CODE_FILES = [
    ROOT / "app.py",
    ROOT / "yolo_weed_detector.py",
    ROOT / "image_model.py",
    ROOT / "train_image_model.py",
]

WEB_CODE_FILES = [
    ROOT / "_tmp_frontend_index.html",
    ROOT / "_tmp_frontend_script.js",
    ROOT / "_tmp_frontend_style.css",
]

NS_WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

ET.register_namespace("w", NS_WORD)
ET.register_namespace("r", NS_DOC_REL)


def xml_escape(value: str) -> str:
    return html.escape(value, quote=False)


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def extract_template_sectpr(template_xml: str) -> str:
    match = re.search(r"(<w:sectPr[\s\S]*?</w:sectPr>)", template_xml)
    if match:
        return match.group(1)
    return (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )


def paragraph(
    text: str,
    *,
    style: str | None = None,
    center: bool = False,
    bold: bool = False,
    italic: bool = False,
    font: str | None = None,
    size_half_points: int | None = None,
    after_spacing: int | None = None,
) -> str:
    ppr_parts: list[str] = []
    if style:
        ppr_parts.append(f'<w:pStyle w:val="{style}"/>')
    if center:
        ppr_parts.append('<w:jc w:val="center"/>')
    if after_spacing is not None:
        ppr_parts.append(f'<w:spacing w:after="{after_spacing}"/>')
    ppr = f"<w:pPr>{''.join(ppr_parts)}</w:pPr>" if ppr_parts else ""

    run_props: list[str] = []
    if bold:
        run_props.append("<w:b/>")
    if italic:
        run_props.append("<w:i/>")
    if font:
        run_props.append(f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>')
    if size_half_points is not None:
        run_props.append(f'<w:sz w:val="{size_half_points}"/><w:szCs w:val="{size_half_points}"/>')
    rpr = f"<w:rPr>{''.join(run_props)}</w:rPr>" if run_props else ""

    lines = text.splitlines() or [""]
    run_xml_parts = []
    for index, line in enumerate(lines):
        safe_line = xml_escape(line)
        text_node = f'<w:t xml:space="preserve">{safe_line}</w:t>' if line else '<w:t xml:space="preserve"></w:t>'
        run_xml_parts.append(f"<w:r>{rpr}{text_node}</w:r>")
        if index != len(lines) - 1:
            run_xml_parts.append("<w:r><w:br/></w:r>")
    return f"<w:p>{ppr}{''.join(run_xml_parts)}</w:p>"


def page_break() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def simple_list(items: Iterable[str]) -> list[str]:
    return [paragraph(f"- {item}") for item in items]


def code_block(text: str) -> list[str]:
    paragraphs: list[str] = []
    lines = text.split("\n")
    for line in lines:
        paragraphs.append(
            paragraph(
                line,
                font="Consolas",
                size_half_points=14,
                after_spacing=0,
            )
        )
    if not paragraphs:
        paragraphs.append(paragraph("", font="Consolas", size_half_points=14, after_spacing=0))
    paragraphs.append(paragraph("", after_spacing=80))
    return paragraphs


def add_image_run(
    rel_id: str,
    filename: str,
    width_px: int,
    height_px: int,
    doc_pr_id: int,
    max_width_emu: int = 5_900_000,
) -> str:
    cx = width_px * 9525
    cy = height_px * 9525
    if cx > max_width_emu:
        scale = max_width_emu / cx
        cx = int(cx * scale)
        cy = int(cy * scale)

    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{doc_pr_id}" name="{xml_escape(filename)}"/>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr>
                <pic:cNvPr id="{doc_pr_id}" name="{xml_escape(filename)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def build_document_content(image_rel_map: dict[str, str]) -> str:
    results = load_results()
    positive = results["positive_large"]
    small = results["small_auto_upscaled"]
    invalid = results["invalid_dark"]

    content: list[str] = []

    content.extend(
        [
            paragraph("FINAL PROJECT REPORT", center=True, bold=True, size_half_points=28, after_spacing=120),
            paragraph(
                "Image-Based Weed Detection in Crop Fields Using Pretrained YOLO and Dual User Interfaces",
                center=True,
                bold=True,
                size_half_points=34,
                after_spacing=180,
            ),
            paragraph("A Word-Format Final Submission Report for Academic Evaluation", center=True, italic=True),
            paragraph("", center=True, after_spacing=240),
            paragraph("Student Name: [Your Name]", center=True, bold=True),
            paragraph("USN: [Your USN]", center=True),
            paragraph("Department: [Department Name]", center=True),
            paragraph("Institution: [College / University Name]", center=True),
            paragraph("Guide / Faculty: [Guide Name]", center=True),
            paragraph("Academic Year: 2025-2026", center=True),
            paragraph("Date of Submission: April 16, 2026", center=True),
            paragraph("Replace the bracketed fields before final submission.", center=True, italic=True),
            page_break(),
            paragraph("Certificate", style="Heading1"),
            paragraph(
                'This is to certify that the project entitled "Image-Based Weed Detection in Crop Fields Using '
                'Pretrained YOLO and Dual User Interfaces" is a bonafide work carried out by [Your Name] during the '
                "academic year 2025-2026 under our guidance."
            ),
            paragraph("Guide Signature: ____________________"),
            paragraph("Head of Department: ____________________"),
            page_break(),
            paragraph("Declaration", style="Heading1"),
            paragraph(
                "I hereby declare that this project report is my original work and has not been submitted elsewhere "
                "for any other academic award. All references and supporting sources used in this work have been "
                "properly acknowledged."
            ),
            paragraph("Student Signature: ____________________"),
            page_break(),
            paragraph("Acknowledgement", style="Heading1"),
            paragraph(
                "I express my sincere gratitude to my guide, faculty members, and institution for their support, "
                "encouragement, and technical guidance during the completion of this project. I also thank my friends "
                "and family for their constant motivation. The published research papers, open-source datasets, and "
                "public pretrained model sources referenced in this work were especially useful in improving the "
                "project from a basic feature-based prototype into a more practical image-based weed-detection system."
            ),
            page_break(),
            paragraph("Abstract", style="Heading1"),
            paragraph(
                "Weed infestation reduces agricultural productivity because weeds compete with crops for water, "
                "nutrients, sunlight, and space. Manual inspection is time-consuming, while blanket herbicide "
                "spraying is inefficient and environmentally harmful. This project presents a final image-based weed "
                "detection system that uses a pretrained YOLO model as the primary detector and offers both a "
                "Streamlit application and a lightweight browser demo for user interaction. The system accepts "
                "crop-field RGB images, validates them for practical quality issues, and now uses a relaxed upload "
                "policy: very tiny images are rejected, while smaller usable images are auto-upscaled to the "
                "detector's preferred minimum size before inference. During local verification, the larger accepted "
                f"test image produced {positive['weed_count']} weed detections with a top confidence of "
                f"{positive['confidence'] * 100:.1f}%, while the smaller accepted image was auto-upscaled from "
                f"{small['original_size'][0]} x {small['original_size'][1]} to {small['processed_size'][0]} x "
                f"{small['processed_size'][1]} pixels and still returned {small['weed_count']} weed detections. "
                "The final report includes the full source code, real program output, analytical figures, and the "
                "corrected workflow used for submission."
            ),
            page_break(),
            paragraph("Table of Contents", style="Heading1"),
        ]
    )
    for item in [
        "Certificate",
        "Declaration",
        "Acknowledgement",
        "Abstract",
        "Chapter 1: Introduction",
        "Chapter 2: Literature Survey",
        "Chapter 3: Existing System and Proposed System",
        "Chapter 4: System Design and Requirements",
        "Chapter 5: Methodology",
        "Chapter 6: User Interface and Implementation",
        "Chapter 7: Execution, Output, and Analysis",
        "Chapter 8: Complete Program Code",
        "Chapter 9: Applications",
        "Chapter 10: Conclusion and Future Scope",
        "References",
    ]:
        content.append(paragraph(item))

    content.extend(
        [
            page_break(),
            paragraph("Chapter 1: Introduction", style="Heading1"),
            paragraph(
                "Weeds are unwanted plants that reduce agricultural productivity by competing with cultivated crops "
                "for essential resources. Traditional weed monitoring depends heavily on manual field inspection or "
                "broad chemical spraying. This project upgrades an earlier prototype into a more practical, "
                "image-driven weed-detection pipeline."
            ),
            paragraph(
                "The final implementation allows users to upload crop-field images, validates whether those images are "
                "suitable for analysis, runs a pretrained YOLO detector, and displays the result alongside supporting "
                "vegetation analytics. A lightweight browser demo is also maintained for presentation, while the "
                "Streamlit interface is treated as the primary working system."
            ),
            paragraph("Problem Statement", style="Heading2"),
            paragraph(
                "Build a user-friendly system that accepts crop-field images, validates image quality, detects weeds "
                "using a pretrained object-detection model, and displays the result with visual evidence and "
                "analytical explanation."
            ),
            paragraph("Objectives", style="Heading2"),
        ]
    )
    content.extend(
        simple_list(
            [
                "Provide a primary working weed detector with an upload-based user interface.",
                "Support smaller usable images through automatic upscaling instead of over-restrictive rejection.",
                "Display vegetation masks and feature summaries for explanation.",
                "Verify the program on accepted and rejected scenarios and include the real outputs in the report.",
                "Package the final submission in Word format with the full program code.",
            ]
        )
    )

    content.extend(
        [
            paragraph("Chapter 2: Literature Survey", style="Heading1"),
            paragraph(
                "The literature consistently shows that real field-image analysis and deep learning based detectors are "
                "more suitable for practical weed recognition than purely synthetic tabular experiments. DeepWeeds "
                "(2019), the 2023 review on weed-plant discrimination in agriculture 5.0, and recent systematic "
                "deep-learning surveys were especially important in shaping the final direction of this project."
            ),
            paragraph(
                "Student-provided papers such as IRJET-V11I5168 and IJCRT2203197 were also used as secondary "
                "supporting references while refining the report structure and technical background."
            ),
            paragraph("Chapter 3: Existing System and Proposed System", style="Heading1"),
            paragraph(
                "The earlier idea relied more heavily on feature-only logic and strict image rejection. The final "
                "proposed system uses a pretrained YOLO detector, a clearer upload workflow, supporting vegetation "
                "analytics, and dual interfaces. It also replaces the old rigid 224 x 224 gate with a two-level rule: "
                "very tiny images are rejected, while smaller usable images are auto-upscaled and clearly marked."
            ),
            paragraph("Chapter 4: System Design and Requirements", style="Heading1"),
            paragraph("Software stack: Python 3.13, Streamlit, Pillow, NumPy, Pandas, Requests, Torch, Torchvision, Ultralytics, scikit-learn, and Joblib."),
            paragraph("Primary UI: Streamlit application."),
            paragraph("Secondary UI: lightweight browser demo using HTML, CSS, and JavaScript."),
            paragraph("Primary model file: models/weedblaster-vision-yolov8s.pt."),
        ]
    )

    doc_pr_id = 1
    for filename, caption in IMAGE_ASSETS[:4]:
        rel_id = image_rel_map.get(filename)
        asset_path = ASSET_DIR / filename
        if not rel_id or not asset_path.exists():
            continue
        with Image.open(asset_path) as image:
            width, height = image.size
        content.append(add_image_run(rel_id, filename, width, height, doc_pr_id))
        doc_pr_id += 1
        content.append(paragraph(caption, center=True, italic=True))

    content.extend(
        [
            paragraph("Chapter 5: Methodology", style="Heading1"),
            paragraph(
                "Input validation now accepts PNG, JPG, JPEG, and WEBP crop-field images. Images whose shorter side "
                "is below 128 pixels are rejected as too small. Images whose shorter side lies between 128 and 223 "
                "pixels are accepted and auto-upscaled to the detector's preferred minimum size of 224 pixels. "
                "Brightness, exposure, vegetation visibility, blur, and field suitability checks are also applied."
            ),
            paragraph(
                "The primary deployed detector is a pretrained YOLO model loaded through the ultralytics package and "
                "used for CPU-based inference. In parallel, the application computes RGB vegetation analytics using "
                "Excess Green segmentation. These analytics support explanation and the older Random Forest fallback "
                "bundle."
            ),
            paragraph("Chapter 6: User Interface and Implementation", style="Heading1"),
            paragraph(
                "The Streamlit application is the main interface used for the final detector. A secondary browser demo "
                "mirrors the upload policy and provides a lightweight presentation layer. The app shows the original "
                "image, YOLO detection output, vegetation mask, overlay, notes, and a detection table."
            ),
            paragraph(
                "Key runtime configuration: confidence threshold 0.25, preferred minimum shorter-side input size 224 "
                "pixels, absolute rejection threshold 128 pixels, and a secondary Random Forest fallback retained for "
                "supporting analytics only."
            ),
            paragraph("Chapter 7: Execution, Output, and Analysis", style="Heading1"),
            paragraph(
                f"Larger accepted run: status={positive['status']}, weed_detections={positive['weed_count']}, "
                f"confidence={positive['confidence'] * 100:.1f}%, processed_size={positive['processed_size'][0]} x {positive['processed_size'][1]} px."
            ),
            paragraph(
                f"Smaller accepted run: status={small['status']}, weed_detections={small['weed_count']}, "
                f"confidence={small['confidence'] * 100:.1f}%, original_size={small['original_size'][0]} x {small['original_size'][1]} px, "
                f"processed_size={small['processed_size'][0]} x {small['processed_size'][1]} px, auto_upscaled={'Yes' if small['auto_upscaled'] else 'No'}."
            ),
            paragraph(
                f"Invalid dark-image run: status={invalid['status']}, notes={'; '.join(invalid['notes'])}."
            ),
        ]
    )

    for filename, caption in IMAGE_ASSETS[4:]:
        rel_id = image_rel_map.get(filename)
        asset_path = ASSET_DIR / filename
        if not rel_id or not asset_path.exists():
            continue
        with Image.open(asset_path) as image:
            width, height = image.size
        content.append(add_image_run(rel_id, filename, width, height, doc_pr_id))
        doc_pr_id += 1
        content.append(paragraph(caption, center=True, italic=True))

    content.extend(
        [
            paragraph("Detection Notes from the Verified Runs", style="Heading2"),
            paragraph("Large accepted image:"),
        ]
    )
    content.extend(simple_list(positive["notes"]))
    content.append(paragraph("Small accepted and auto-upscaled image:"))
    content.extend(simple_list(small["notes"]))
    content.append(paragraph("Invalid dark image:"))
    content.extend(simple_list(invalid["notes"]))

    content.extend(
        [
            paragraph("Chapter 8: Complete Program Code", style="Heading1"),
            paragraph(
                "The code listings are separated into Python implementation files and web interface files so the "
                "submission format is cleaner and easier to review. Each file is placed line by line in a monospaced "
                "layout so it stays closer to the original programming format."
            ),
        ]
    )

    content.append(paragraph("8.1 Python Implementation Code", style="Heading2"))
    for code_path in PYTHON_CODE_FILES:
        content.append(paragraph(code_path.name, style="Heading2"))
        content.extend(code_block(code_path.read_text(encoding="utf-8")))

    content.append(paragraph("8.2 Web Interface Code", style="Heading2"))
    for code_path in WEB_CODE_FILES:
        content.append(paragraph(code_path.name, style="Heading2"))
        content.extend(code_block(code_path.read_text(encoding="utf-8")))

    content.extend(
        [
            paragraph("Chapter 9: Applications", style="Heading1"),
        ]
    )
    content.extend(
        simple_list(
            [
                "Site-specific weed monitoring in precision agriculture",
                "Farmer-facing crop-field screening tools with visual evidence",
                "Decision support for targeted herbicide spraying",
                "Future integration with drone or mobile-field imaging pipelines",
                "Educational demonstrations of computer vision in agriculture",
            ]
        )
    )
    content.extend(
        [
            paragraph("Chapter 10: Conclusion and Future Scope", style="Heading1"),
            paragraph(
                "The final project successfully upgrades the earlier prototype into a working image-based weed "
                "detection system with a pretrained YOLO detector, dual user interfaces, real runtime verification, "
                "and report-ready outputs. The most important correction made during finalization was replacing the "
                "over-restrictive image rule with a more practical two-level policy."
            ),
            paragraph(
                "Future improvements should include evaluation on a full labeled crop/weed dataset, fine-tuning of "
                "the detector for specific crops and weed species, and connecting the lightweight browser demo "
                "directly to the same backend used by the Streamlit application."
            ),
            paragraph("References", style="Heading1"),
        ]
    )
    for reference in [
        'A. Olsen et al., "DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning," Scientific Reports, 2019.',
        'F. H. Juwono et al., "Machine learning for weed-plant discrimination in agriculture 5.0: An in-depth review," Artificial Intelligence in Agriculture, 2023.',
        '"Weed Detection Using Deep Learning: A Systematic Literature Review," Sensors, 2023.',
        "CropAndWeed dataset repository, GitHub.",
        "weedblaster-vision-yolov8s pretrained model source, Hugging Face.",
        "IRJET-V11I5168.pdf, supporting implementation-style reference supplied by the student.",
        "IJCRT2203197.pdf, supporting review reference supplied by the student.",
    ]:
        content.append(paragraph(reference))

    return "\n".join(content)


def ensure_content_types(content_types_xml: bytes) -> bytes:
    root = ET.fromstring(content_types_xml)
    default_tags = {child.attrib.get("Extension") for child in root if child.tag.endswith("Default")}
    if "png" not in default_tags:
        ET.SubElement(root, "{http://schemas.openxmlformats.org/package/2006/content-types}Default", Extension="png", ContentType="image/png")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_image_relationships(template_files: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, str]]:
    rels_path = "word/_rels/document.xml.rels"
    rels_root = ET.fromstring(template_files[rels_path])

    next_id = 1
    for rel in rels_root:
        rel_id = rel.attrib.get("Id", "")
        if rel_id.startswith("rId") and rel_id[3:].isdigit():
            next_id = max(next_id, int(rel_id[3:]) + 1)

    image_rel_map: dict[str, str] = {}
    for filename, _caption in IMAGE_ASSETS:
        asset_path = ASSET_DIR / filename
        if not asset_path.exists():
            continue
        rel_id = f"rId{next_id}"
        next_id += 1
        image_rel_map[filename] = rel_id
        ET.SubElement(
            rels_root,
            f"{{{NS_REL}}}Relationship",
            Id=rel_id,
            Type=f"{NS_DOC_REL}/image",
            Target=f"media/{filename}",
        )
        template_files[f"word/media/{filename}"] = asset_path.read_bytes()

    template_files[rels_path] = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)
    return template_files, image_rel_map


def main() -> None:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template DOCX not found: {TEMPLATE_PATH}")
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Verification results not found: {RESULTS_PATH}")

    template_files: dict[str, bytes] = {}
    with zipfile.ZipFile(TEMPLATE_PATH, "r") as template_zip:
        for member in template_zip.namelist():
            template_files[member] = template_zip.read(member)

    template_files["[Content_Types].xml"] = ensure_content_types(template_files["[Content_Types].xml"])
    template_files, image_rel_map = build_image_relationships(template_files)

    template_document_xml = template_files["word/document.xml"].decode("utf-8")
    sect_pr = extract_template_sectpr(template_document_xml)
    body_content = build_document_content(image_rel_map)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {body_content}
    {sect_pr}
  </w:body>
</w:document>
"""
    template_files["word/document.xml"] = document_xml.encode("utf-8")

    for output_path in OUTPUT_PATHS:
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            for member_name, member_bytes in template_files.items():
                output_zip.writestr(member_name, member_bytes)
        print(f"Wrote {output_path.name}")


if __name__ == "__main__":
    main()
