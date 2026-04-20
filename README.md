# Crop Field Weed Detection Prototype

This project now supports a real pretrained weed detector for uploaded field images.
The app first runs a pretrained YOLO model with bounding boxes, then shows vegetation-mask analytics for explanation.
If the pretrained detector cannot run, the older Random Forest-style vegetation-pattern fallback is still available.

## What is included

- `app.py`
  - Streamlit user interface for uploading images, viewing detections, and showing analytics.
- `yolo_weed_detector.py`
  - Loads a pretrained remote YOLO weed model, validates the uploaded image, and returns weed detections.
- `image_model.py`
  - RGB vegetation-feature extraction, synthetic image-feature training, and fallback inference logic.
- `train_image_model.py`
  - Script to train and save the image-based Random Forest model bundle.
- `LITERATURE_REVIEW.md`
  - Published sources used to guide the redesign.
- `requirements.txt`
  - Python dependencies for the app.

## How it works

1. The user uploads a crop-field image.
2. The app validates whether the image is suitable for field analysis.
3. A pretrained YOLO weed detector runs on the image and produces crop / weed bounding boxes.
4. The app also computes vegetation analytics from the RGB image:
   - vegetation coverage
   - Excess Green mean
   - Excess Green spread
   - texture variation
   - vegetation component density
   - row alignment
5. The UI shows:
   - the original image
   - YOLO detection results
   - a vegetation mask
   - an overlay analytics view
   - a weed/no-weed result
   - detection table and extracted feature values

## Why the design changed

Your earlier project used synthetic tabular features like `GreenIndex` and `NDVI`.
That works for a report, but a real user cannot upload a normal phone image and get a true NDVI value from RGB alone.

This redesign uses a pretrained object detector for the final decision and RGB-friendly vegetation cues for explanation, so the app is much closer to a real project demo than the earlier synthetic-only version.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Notes:
- `python train_image_model.py` is optional now. It is only needed if you want the fallback Random Forest bundle refreshed.
- On the first real detection run, the app downloads the pretrained weed model weights automatically.
- If you want to swap to a GitHub-hosted `.pt` file later, set `WEED_MODEL_URL` before running the app.

## Recommended demo flow

1. Launch the app with `streamlit run app.py`.
2. Upload a clear crop-field image with visible rows or plantation structure.
3. Show the YOLO bounding boxes and explain whether weeds were detected.
4. Use the vegetation mask and analytics metrics as supporting explanation.
5. Mention that the detector uses pretrained remote weights and should still be validated on the target field conditions.

## Important limitation

This prototype is much stronger than the earlier synthetic-only version, but it is still not a fully field-validated production system.
The pretrained model was trained on external crop/weed imagery and may not generalize perfectly to every geography, camera angle, soil type, or crop variety.
For real-world accuracy, validate and fine-tune the pipeline on your target field data.

## Research sources

- DeepWeeds: https://www.nature.com/articles/s41598-018-38343-3
- Machine learning for weed-plant discrimination in agriculture 5.0: An in-depth review: https://www.sciencedirect.com/science/article/pii/S2589721723000363
- CWFID: https://github.com/cwfid/dataset
- Excess Green and field vegetation extraction: https://www.nature.com/articles/s41598-023-33042-0
- Systematic review: https://www.mdpi.com/1424-8220/23/7/3670
- RGB vegetation indices for weed detection: https://www.mdpi.com/2072-4292/17/11/1899
- Supplementary engineering articles:
  - IRJET 2024, "Weed Detection Using Machine Learning": https://www.irjet.net/archives/V11/i5/IRJET-V11I5168.pdf
  - IJCRT 2022, "Weed Detection Using Deep Learning Techniques: A Review": https://ijcrt.org/papers/IJCRT2203197.pdf
