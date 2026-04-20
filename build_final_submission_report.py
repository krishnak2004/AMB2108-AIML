"""Build the final submission report HTML from live project assets."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "report_assets"
RESULTS_PATH = ASSET_DIR / "verification_results.json"

OUTPUT_HTML_PATHS = [
    ROOT / "Final_Submission_Report.html",
    ROOT / "Weed_Detection_Final_Submission.html",
    ROOT / "submission_report.html",
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


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing verification results at {RESULTS_PATH}")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def asset_tag(filename: str, caption: str) -> str:
    asset_path = ASSET_DIR / filename
    if not asset_path.exists():
        return f"<div class='boxed'><p><strong>Asset missing:</strong> {html.escape(filename)}</p></div>"
    return (
        "<div class='figure'>"
        f"<img src='report_assets/{html.escape(filename)}' alt='{html.escape(caption)}'>"
        f"<div class='caption'>{html.escape(caption)}</div>"
        "</div>"
    )


def format_result_rows(result: dict) -> str:
    rows = [
        ("Status", result["status"]),
        ("Headline", result["headline"]),
        ("Original size", f"{result['original_size'][0]} x {result['original_size'][1]} px"),
        ("Processed size", f"{result['processed_size'][0]} x {result['processed_size'][1]} px"),
        ("Auto-upscaled", "Yes" if result["auto_upscaled"] else "No"),
        ("Weed detections", str(result["weed_count"])),
        ("Crop detections", str(result["crop_count"])),
        ("Top confidence", f"{result['confidence'] * 100:.1f}%"),
    ]
    return "\n".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )


def format_detection_table(result: dict) -> str:
    detections = result.get("detections", [])
    if not detections:
        return "<p class='small-note'>No bounding boxes were returned for this case.</p>"

    rows = []
    for record in detections:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('Class', '')))}</td>"
            f"<td>{html.escape(str(record.get('Confidence', '')))}</td>"
            f"<td>{html.escape(str(record.get('x1', '')))}</td>"
            f"<td>{html.escape(str(record.get('y1', '')))}</td>"
            f"<td>{html.escape(str(record.get('x2', '')))}</td>"
            f"<td>{html.escape(str(record.get('y2', '')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Class</th><th>Confidence</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def format_notes(result: dict) -> str:
    notes = result.get("notes", [])
    if not notes:
        return "<p class='small-note'>No additional notes were recorded.</p>"
    return "<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in notes) + "</ul>"


def format_code_sections(title: str, code_paths: list[Path]) -> str:
    sections: list[str] = []
    sections.append(f"<h3>{html.escape(title)}</h3>")
    for code_path in code_paths:
        code_text = code_path.read_text(encoding="utf-8")
        sections.append(
            f"<h4>{html.escape(code_path.name)}</h4>"
            f"<pre class='code-block'>{html.escape(code_text)}</pre>"
        )
    return "\n".join(sections)


def format_requirements() -> str:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return "<ul>" + "".join(f"<li><code>{html.escape(line)}</code></li>" for line in requirements if line.strip()) + "</ul>"


def build_html() -> str:
    results = load_results()
    positive = results["positive_large"]
    small = results["small_auto_upscaled"]
    invalid = results["invalid_dark"]

    browser_ui_figure = (
        asset_tag("browser_ui_upload.png", "Figure 3. Lightweight browser demo interface after the upload-policy update.")
        if (ASSET_DIR / "browser_ui_upload.png").exists()
        else asset_tag("ui_mockup.svg", "Figure 3. Browser interface layout used for the lightweight demo UI.")
    )
    streamlit_ui_figure = asset_tag(
        "sample_program_output.png",
        "Figure 4. Primary Streamlit application view showing the executed weed-detection result.",
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Final Submission Report - Weed Detection</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm;
    }}
    :root {{
      --ink: #1f2933;
      --muted: #52606d;
      --accent: #165c40;
      --accent-soft: #e9f3ed;
      --line: #d9e2ec;
      --paper: #ffffff;
      --soft: #f8fbf9;
      --soft-gold: #f3ead8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: "Georgia", "Times New Roman", serif;
      font-size: 12pt;
      line-height: 1.55;
    }}
    .page {{
      max-width: 860px;
      margin: 0 auto;
      padding: 22px 12px 60px;
    }}
    .page-break {{ page-break-before: always; }}
    .cover {{
      min-height: 1000px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      text-align: center;
      padding: 34px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background:
        radial-gradient(circle at top left, rgba(213, 236, 221, 0.72), transparent 30%),
        radial-gradient(circle at bottom right, rgba(244, 229, 198, 0.72), transparent 32%),
        #fbfcfb;
    }}
    .cover .eyebrow {{
      font-size: 10pt;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 18px;
    }}
    .cover h1 {{
      font-size: 26pt;
      line-height: 1.2;
      margin: 0 0 14px;
    }}
    .cover h2 {{
      font-size: 15pt;
      color: var(--muted);
      margin: 0 0 30px;
      font-weight: normal;
    }}
    .meta-card {{
      max-width: 620px;
      margin: 0 auto;
      text-align: left;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px 26px;
    }}
    .meta-card p {{ margin: 10px 0; }}
    h2.section-title {{
      margin-top: 36px;
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 2px solid var(--line);
      color: var(--accent);
      font-size: 17pt;
    }}
    h3 {{
      margin-top: 24px;
      margin-bottom: 8px;
      font-size: 13.5pt;
      color: var(--ink);
    }}
    p {{
      margin: 10px 0;
      text-align: justify;
    }}
    ul, ol {{
      margin: 8px 0 12px 22px;
    }}
    li {{ margin: 4px 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 20px;
      font-size: 11pt;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{ background: #f8fafc; }}
    .boxed {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      margin: 14px 0 20px;
    }}
    .figure {{
      margin: 20px 0 24px;
      border: 1px solid var(--line);
      background: #fcfdfd;
      border-radius: 16px;
      padding: 14px;
    }}
    .figure img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
    }}
    .caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 10.5pt;
      text-align: center;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 16px 0 24px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: linear-gradient(135deg, #ffffff, #f5faf7);
      text-align: center;
    }}
    .metric-card .label {{
      font-size: 10pt;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .metric-card .value {{
      font-size: 16pt;
      font-weight: bold;
      color: var(--accent);
    }}
    pre.code-block {{
      margin: 14px 0 20px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #f6f8fa;
      overflow-x: auto;
      font-size: 9.8pt;
      line-height: 1.48;
      font-family: "Consolas", "Courier New", monospace;
      white-space: pre-wrap;
    }}
    .small-note {{
      color: var(--muted);
      font-size: 10.5pt;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="cover">
      <div class="eyebrow">Final Project Report</div>
      <h1>Image-Based Weed Detection in Crop Fields Using Pretrained YOLO and Dual User Interfaces</h1>
      <h2>A Word-Format Final Submission Report for Academic Evaluation</h2>
      <div class="meta-card">
        <p><strong>Student Name:</strong> [Your Name]</p>
        <p><strong>USN:</strong> [Your USN]</p>
        <p><strong>Department:</strong> [Department Name]</p>
        <p><strong>Institution:</strong> [College / University Name]</p>
        <p><strong>Guide / Faculty:</strong> [Guide Name]</p>
        <p><strong>Academic Year:</strong> 2025-2026</p>
        <p><strong>Date of Submission:</strong> April 16, 2026</p>
      </div>
      <p class="small-note">Replace the bracketed fields before final submission.</p>
    </section>

    <h2 class="section-title page-break">Certificate</h2>
    <div class="boxed">
      <p>
        This is to certify that the project entitled <strong>"Image-Based Weed Detection in Crop Fields Using
        Pretrained YOLO and Dual User Interfaces"</strong> is a bonafide work carried out by <strong>[Your Name]</strong>
        during the academic year 2025-2026 in partial fulfillment of the academic requirements under our guidance.
      </p>
      <p><strong>Guide Signature:</strong> ____________________</p>
      <p><strong>Head of Department:</strong> ____________________</p>
    </div>

    <h2 class="section-title page-break">Declaration</h2>
    <div class="boxed">
      <p>
        I hereby declare that this project report is my original work and has not been submitted elsewhere for any
        other academic award. All references and supporting sources used in this work have been properly acknowledged.
      </p>
      <p><strong>Student Signature:</strong> ____________________</p>
    </div>

    <h2 class="section-title page-break">Acknowledgement</h2>
    <p>
      I express my sincere gratitude to my guide, faculty members, and institution for their support, encouragement,
      and technical guidance during the completion of this project. I also thank my friends and family for their
      constant motivation. The published research papers, open-source datasets, and public pretrained model sources
      referenced in this work were especially useful in improving the project from a basic feature-based prototype into
      a more practical image-based weed-detection system.
    </p>

    <h2 class="section-title page-break">Abstract</h2>
    <p>
      Weed infestation reduces agricultural productivity because weeds compete with crops for water, nutrients,
      sunlight, and space. Manual inspection is time-consuming, while blanket herbicide spraying is inefficient and
      environmentally harmful. This project presents a final image-based weed detection system that uses a pretrained
      YOLO model as the primary detector and offers both a Streamlit application and a lightweight browser demo for user
      interaction. The system accepts crop-field RGB images, validates them for practical quality issues, and now uses a
      relaxed upload policy: very tiny images are rejected, while smaller usable images are auto-upscaled to the
      detector's preferred minimum size before inference. The application also computes supporting vegetation analytics
      such as Excess Green segmentation, vegetation coverage, texture variation, component density, and row alignment.
      During local verification, the larger accepted test image produced {positive["weed_count"]} weed detections with a top
      confidence of {positive["confidence"] * 100:.1f}%, while the smaller accepted image was auto-upscaled from
      {small["original_size"][0]} x {small["original_size"][1]} to {small["processed_size"][0]} x {small["processed_size"][1]}
      pixels and still returned {small["weed_count"]} weed detections. The final report includes the full source code,
      real program output, analytical figures, UI views, and the corrected workflow used for submission.
    </p>

    <h2 class="section-title page-break">Table of Contents</h2>
    <ol>
      <li>Certificate</li>
      <li>Declaration</li>
      <li>Acknowledgement</li>
      <li>Abstract</li>
      <li>Chapter 1: Introduction</li>
      <li>Chapter 2: Literature Survey</li>
      <li>Chapter 3: Existing System and Proposed System</li>
      <li>Chapter 4: System Design and Requirements</li>
      <li>Chapter 5: Methodology</li>
      <li>Chapter 6: User Interface and Implementation</li>
      <li>Chapter 7: Execution, Output, and Analysis</li>
      <li>Chapter 8: Complete Program Code</li>
      <li>Chapter 9: Applications</li>
      <li>Chapter 10: Conclusion and Future Scope</li>
      <li>References</li>
    </ol>

    <h2 class="section-title">List of Figures</h2>
    <ol>
      <li>Workflow of the final weed detection system</li>
      <li>System architecture</li>
      <li>Browser demo interface</li>
      <li>Primary Streamlit interface</li>
      <li>Program output after running the detector</li>
      <li>Primary YOLO detection example</li>
      <li>Secondary Random Forest analytics metrics</li>
      <li>Secondary Random Forest confusion matrix</li>
      <li>Secondary Random Forest feature importance</li>
    </ol>

    <h2 class="section-title page-break">Chapter 1: Introduction</h2>
    <p>
      Weeds are unwanted plants that reduce agricultural productivity by competing with cultivated crops for essential
      resources. Traditional weed monitoring depends heavily on manual field inspection or broad chemical spraying. In
      precision agriculture, there is a strong need for image-based decision systems that can identify weed presence
      from field images and support more targeted intervention.
    </p>
    <p>
      This project began as a simpler feature-based prototype but was redesigned into a more practical image-driven
      pipeline. The final implementation now allows users to upload crop-field images, checks whether those images are
      suitable for analysis, runs a pretrained YOLO weed detector, and displays the result alongside supporting
      vegetation analytics. The submission also includes a lightweight browser demo interface that mirrors the upload
      policy and user-facing behavior for presentation purposes.
    </p>
    <div class="boxed">
      <p><strong>Problem Statement:</strong> Build a user-friendly system that accepts crop-field images, validates
      image quality, detects weeds using a pretrained object-detection model, and displays the result with visual
      evidence and analytical explanation.</p>
      <p><strong>Objectives:</strong></p>
      <ul>
        <li>Provide a primary working weed detector with a simple upload-based user interface.</li>
        <li>Support smaller usable images through automatic upscaling instead of over-restrictive rejection.</li>
        <li>Display explainable analytics such as vegetation masks and feature summaries.</li>
        <li>Verify the program on accepted and rejected scenarios and include the real outputs in the final report.</li>
        <li>Package the final submission in Word format with full program code.</li>
      </ul>
    </div>

    <h2 class="section-title">Chapter 2: Literature Survey</h2>
    <table>
      <thead>
        <tr>
          <th>Reference</th>
          <th>Main Contribution</th>
          <th>How It Influenced This Project</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>DeepWeeds (Scientific Reports, 2019)</td>
          <td>Benchmark field-image weed dataset and image-based baseline</td>
          <td>Motivated the shift from synthetic tabular-only logic to field-image detection</td>
        </tr>
        <tr>
          <td>Machine learning for weed-plant discrimination in agriculture 5.0 (2023)</td>
          <td>Comprehensive review of weed discrimination methods in agriculture</td>
          <td>Supported the methodology, validation discussion, and literature framing</td>
        </tr>
        <tr>
          <td>Weed Detection Using Deep Learning: A Systematic Literature Review (Sensors, 2023)</td>
          <td>Summarizes modern deep learning strategies for weed detection</td>
          <td>Supported the choice of an object-detection based final system</td>
        </tr>
        <tr>
          <td>IRJET-V11I5168 and IJCRT2203197</td>
          <td>Student-provided supporting engineering/review references</td>
          <td>Used as secondary background references for report strengthening</td>
        </tr>
      </tbody>
    </table>
    <p>
      The literature consistently shows that real field-image analysis and deep learning based detectors are more
      suitable for practical weed recognition than purely synthetic tabular experiments. This directly shaped the final
      redesign of the project.
    </p>

    <h2 class="section-title">Chapter 3: Existing System and Proposed System</h2>
    <table>
      <thead>
        <tr>
          <th>Aspect</th>
          <th>Earlier / Existing Approach</th>
          <th>Final Proposed System</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Input</td><td>Feature-only or manual observation</td><td>RGB crop-field image upload</td></tr>
        <tr><td>Primary model</td><td>Random Forest prototype</td><td>Pretrained YOLO weed detector</td></tr>
        <tr><td>User interface</td><td>Limited or absent</td><td>Streamlit app + lightweight browser demo</td></tr>
        <tr><td>Image handling</td><td>Strict rejection of small images</td><td>Reject only tiny unusable images, auto-upscale smaller usable images</td></tr>
        <tr><td>Output</td><td>Label only</td><td>Bounding boxes, notes, analytics, confidence, and report-ready visuals</td></tr>
      </tbody>
    </table>

    <h2 class="section-title">Chapter 4: System Design and Requirements</h2>
    <table>
      <thead>
        <tr><th>Type</th><th>Requirement</th></tr>
      </thead>
      <tbody>
        <tr><td>Hardware</td><td>Laptop or desktop, standard RGB crop-field image source</td></tr>
        <tr><td>Software</td><td>Python 3.13, Streamlit, Pillow, NumPy, Pandas, Requests, Torch, Torchvision, Ultralytics, scikit-learn, Joblib</td></tr>
        <tr><td>Primary UI</td><td>Streamlit web application</td></tr>
        <tr><td>Secondary UI</td><td>Static browser demo page using HTML, CSS, and JavaScript</td></tr>
        <tr><td>Model file</td><td><code>models/weedblaster-vision-yolov8s.pt</code></td></tr>
      </tbody>
    </table>
    {asset_tag("project_workflow.svg", "Figure 1. Workflow of the final weed detection system.")}
    {asset_tag("system_architecture.svg", "Figure 2. System architecture of the final weed detection system.")}

    <h2 class="section-title page-break">Chapter 5: Methodology</h2>
    <h3>5.1 Input Validation</h3>
    <p>
      The final system accepts images in PNG, JPG, JPEG, and WEBP formats. Images whose shorter side is below
      <strong>128 pixels</strong> are rejected as too small for meaningful analysis. Images whose shorter side falls
      between <strong>128 pixels and 223 pixels</strong> are accepted and auto-upscaled to the detector's preferred
      minimum size of <strong>224 pixels</strong>. Brightness, exposure, vegetation visibility, blur, and scene quality
      are also checked before inference.
    </p>
    <h3>5.2 Pretrained Detection</h3>
    <p>
      The primary deployed detector is a pretrained YOLO model loaded through the <code>ultralytics</code> package. The
      model is downloaded once, stored locally, and used for CPU-based inference on uploaded images. If the detector
      produces weed-class bounding boxes, the application reports weed presence and shows the annotated image.
    </p>
    <h3>5.3 Supporting Analytics</h3>
    <p>
      In parallel, the application computes RGB vegetation analytics using Excess Green segmentation. The analytics
      module produces vegetation coverage, Excess Green statistics, texture variation, component density, and row
      alignment. These features are shown to the user for explanation and are also used by the older Random Forest
      fallback bundle if the YOLO detector becomes unavailable.
    </p>
    <h3>5.4 Dual Interface Design</h3>
    <p>
      The Streamlit application is the primary working system for final detection. A secondary browser demo interface is
      maintained to present the upload flow and browser-side validation behavior. Both interfaces now follow the same
      image acceptance rules to reduce confusion during demonstration.
    </p>

    <h2 class="section-title">Chapter 6: User Interface and Implementation</h2>
    <h3>6.1 Libraries Used</h3>
    {format_requirements()}
    <h3>6.2 Final Interfaces</h3>
    {browser_ui_figure}
    {streamlit_ui_figure}
    <p>
      The browser demo is included to show a lightweight presentation layer, while the Streamlit interface is the
      primary working implementation that runs the pretrained YOLO detector and the supporting analytics. The final
      report treats the Streamlit app as the main submission UI.
    </p>
    <h3>6.3 Model and Runtime Configuration</h3>
    <table>
      <thead>
        <tr><th>Item</th><th>Details</th></tr>
      </thead>
      <tbody>
        <tr><td>Primary deployed model</td><td>Pretrained YOLO weed detector</td></tr>
        <tr><td>Weights file</td><td><code>models/weedblaster-vision-yolov8s.pt</code></td></tr>
        <tr><td>Inference backend</td><td>CPU via Torch and Ultralytics</td></tr>
        <tr><td>Default confidence threshold</td><td>0.25</td></tr>
        <tr><td>Preferred minimum input size</td><td>224 px on the shorter side</td></tr>
        <tr><td>Absolute rejection threshold</td><td>128 px on the shorter side</td></tr>
        <tr><td>Secondary fallback</td><td>Random Forest analytics bundle</td></tr>
      </tbody>
    </table>
    <div class="boxed">
      <p><strong>How the final interface works:</strong></p>
      <ol>
        <li>The user opens the Streamlit app or browser demo and uploads a crop-field image.</li>
        <li>The system validates file type, image size, exposure, and field suitability.</li>
        <li>Accepted smaller images are auto-upscaled if needed before inference.</li>
        <li>The YOLO model predicts weed boxes and returns the final weed/no-weed decision.</li>
        <li>The UI shows the original image, annotated output, vegetation mask, overlay, summary counts, and notes.</li>
      </ol>
    </div>

    <h2 class="section-title page-break">Chapter 7: Execution, Output, and Analysis</h2>
    <div class="metric-grid">
      <div class="metric-card"><div class="label">Large Accepted Run</div><div class="value">{positive["weed_count"]} weeds</div></div>
      <div class="metric-card"><div class="label">Top Confidence</div><div class="value">{positive["confidence"] * 100:.1f}%</div></div>
      <div class="metric-card"><div class="label">Small Auto-Upscaled Run</div><div class="value">{small["weed_count"]} weeds</div></div>
      <div class="metric-card"><div class="label">Reject Threshold</div><div class="value">128 px</div></div>
    </div>
    <p>
      The final detector was executed on three verification scenarios: a larger accepted field image, a smaller usable
      field image that triggered auto-upscaling, and a deliberately dark invalid image. The larger accepted run
      produced {positive["weed_count"]} weed detections with a top confidence of {positive["confidence"] * 100:.1f}%. The
      smaller accepted run was automatically upscaled from {small["original_size"][0]} x {small["original_size"][1]} to
      {small["processed_size"][0]} x {small["processed_size"][1]} pixels and still returned {small["weed_count"]} weed detections.
      The dark validation image was rejected as unsuitable, demonstrating that the relaxed policy no longer blocks
      modestly small images but still protects the detector from poor-quality inputs.
    </p>
    {asset_tag("runtime_auto_upscaled_detection.png", "Figure 5. Annotated detector output from the smaller accepted image after auto-upscaling.")}
    {asset_tag("yolo_detection_example.png", "Figure 6. Example primary YOLO detection output from the larger accepted image.")}
    <h3>7.1 Verification Scenario: Larger Accepted Image</h3>
    <table><tbody>{format_result_rows(positive)}</tbody></table>
    {format_notes(positive)}
    {format_detection_table(positive)}
    <h3>7.2 Verification Scenario: Smaller Accepted and Auto-Upscaled Image</h3>
    <table><tbody>{format_result_rows(small)}</tbody></table>
    {format_notes(small)}
    {format_detection_table(small)}
    <h3>7.3 Verification Scenario: Invalid Dark Image</h3>
    <table><tbody>{format_result_rows(invalid)}</tbody></table>
    {format_notes(invalid)}
    {asset_tag("runtime_invalid_dark_input.png", "Figure 7. Dark invalid input used to verify rejection behavior.")}
    <h3>7.4 Supporting Analytical Figures</h3>
    {asset_tag("model_metrics.png", "Figure 8. Metrics of the secondary Random Forest analytics bundle retained for supporting analysis.")}
    {asset_tag("confusion_matrix.png", "Figure 9. Confusion matrix of the secondary Random Forest analytics bundle.")}
    {asset_tag("feature_importance.png", "Figure 10. Feature importance scores of the secondary Random Forest analytics bundle.")}

    <h2 class="section-title page-break">Chapter 8: Complete Program Code</h2>
    <p>
      The code is organized below in two separate parts so the implementation is easier to review in the final
      submission: the Python application code used for detection and analytics, and the separate web interface code
      used for the lightweight browser demo.
    </p>
    {format_code_sections("8.1 Python Implementation Code", PYTHON_CODE_FILES)}
    {format_code_sections("8.2 Web Interface Code", WEB_CODE_FILES)}

    <h2 class="section-title page-break">Chapter 9: Applications</h2>
    <ul>
      <li>Site-specific weed monitoring in precision agriculture</li>
      <li>Farmer-facing crop-field screening tools with visual evidence</li>
      <li>Decision support for targeted herbicide spraying</li>
      <li>Integration with drone or mobile-field imaging pipelines in future work</li>
      <li>Educational demonstrations of computer vision in agriculture</li>
    </ul>

    <h2 class="section-title">Chapter 10: Conclusion and Future Scope</h2>
    <p>
      The final project successfully upgrades the earlier prototype into a working image-based weed-detection system
      with a pretrained YOLO detector, dual user interfaces, real runtime verification, and report-ready outputs. The
      most important correction made during finalization was replacing the over-restrictive image rule with a more
      practical two-level policy: truly tiny images are rejected, while smaller usable images are auto-upscaled and
      analyzed with a clear note shown to the user.
    </p>
    <p>
      Future improvements should include evaluation on a full labeled crop/weed dataset, fine-tuning of the detection
      model for specific crops and weed species, and connecting the lightweight browser demo directly to the same
      backend used by the Streamlit application.
    </p>

    <h2 class="section-title">References</h2>
    <ol>
      <li>A. Olsen et al., "DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning," <em>Scientific Reports</em>, 2019.</li>
      <li>F. H. Juwono et al., "Machine learning for weed-plant discrimination in agriculture 5.0: An in-depth review," <em>Artificial Intelligence in Agriculture</em>, 2023.</li>
      <li>"Weed Detection Using Deep Learning: A Systematic Literature Review," <em>Sensors</em>, 2023.</li>
      <li>CropAndWeed dataset repository, GitHub.</li>
      <li>weedblaster-vision-yolov8s pretrained model source, Hugging Face.</li>
      <li>IRJET-V11I5168.pdf, supporting implementation-style reference supplied by the student.</li>
      <li>IJCRT2203197.pdf, supporting review reference supplied by the student.</li>
    </ol>
  </div>
</body>
</html>
"""


def main() -> None:
    html_text = build_html()
    for output_path in OUTPUT_HTML_PATHS:
        output_path.write_text(html_text, encoding="utf-8")
        print(f"Wrote {output_path.name}")


if __name__ == "__main__":
    main()
