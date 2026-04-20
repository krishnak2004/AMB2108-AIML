# Literature Review For The Updated Project

This project redesign was guided by published work on real-field weed detection, public datasets, and RGB vegetation analysis.

## 1. DeepWeeds

- Source: Olsen et al., *Scientific Reports* (2019)
- Link: https://www.nature.com/articles/s41598-018-38343-3
- Why it matters:
  - It is one of the best-known real-world weed datasets for image-based detection.
  - It shows that field-image weed recognition needs diverse outdoor imagery rather than only synthetic tabular features.
- Design implication for this project:
  - The new app accepts real RGB images from users instead of only synthetic feature rows.

## 2. CWFID (Crop/Weed Field Image Dataset)

- Source: Haug and Ostermann, ECCV Workshop paper and public dataset
- Link: https://github.com/cwfid/dataset
- Why it matters:
  - It provides crop/weed field imagery with annotations and segmentation-style information.
  - It highlights the importance of crop-row structure and the difference between orderly crops and scattered weeds.
- Design implication for this project:
  - The updated feature pipeline includes a `row_alignment` signal and vegetation-fragmentation analysis.

## 3. Excess Green for vegetation extraction

- Source: *Towards reducing chemical usage for weed control in agriculture using UAS imagery analysis and computer vision techniques*, *Scientific Reports* (2023)
- Link: https://www.nature.com/articles/s41598-023-33042-0
- Why it matters:
  - The study uses Excess Green (ExG) to separate green vegetation from the background in field imagery.
  - This is useful when the input is a normal RGB image and not a multispectral NDVI image.
- Design implication for this project:
  - The updated app uses ExG and green-dominance thresholding to build a vegetation mask from uploaded field photos.

## 4. Weed Detection Using Deep Learning: A Systematic Literature Review

- Source: *Sensors* (2023)
- Link: https://www.mdpi.com/1424-8220/23/7/3670
- Why it matters:
  - The review shows that image-based weed detection often uses CNNs, transfer learning, and public datasets.
  - It also shows the field is moving toward real-image pipelines rather than purely synthetic attribute tables.
- Design implication for this project:
  - The app is positioned honestly as a prototype and the documentation recommends retraining on a public image dataset for stronger performance.

## 5. RGB vegetation indices for weed detection

- Source: *Assessment of Vegetation Indices Derived from UAV Imagery for Weed Detection in Vineyards*, *Remote Sensing* (2025)
- Link: https://www.mdpi.com/2072-4292/17/11/1899
- Why it matters:
  - It compares RGB-friendly vegetation indices for weed detection and discusses practical strengths and limitations of different indices.
  - The study helps justify using RGB-derived vegetation signals when the user uploads a normal image instead of a multispectral one.
- Design implication for this project:
  - The updated project uses RGB vegetation cues instead of claiming true NDVI from a standard photograph.

## Final takeaway

The literature strongly supports moving your project from a synthetic table classifier to an image-based workflow with:

- real image input
- vegetation segmentation
- crop-row and fragmentation analysis
- a user-facing interface
- clear acknowledgement that final deployment should use a real labeled dataset

## Additional articles you shared

These three papers are useful for strengthening the project write-up, but they are not equally strong as references.

### A. Machine learning for weed-plant discrimination in agriculture 5.0: An in-depth review

- Source: Artificial Intelligence in Agriculture (Elsevier, 2023)
- Link: https://www.sciencedirect.com/science/article/pii/S2589721723000363
- Why it is strong:
  - This is the strongest of the three extra sources because it is a focused review article in a recognized journal.
  - It explicitly discusses image acquisition, background removal, feature-based recognition, ML-based algorithms, DL-based algorithms, and public datasets.
- Best use in your project:
  - cite it in the introduction, literature review, and methodology justification
  - use it to justify the move from simple tabular features to image preprocessing plus machine-learning classification

### B. Weed Detection Using Machine Learning

- Source: IRJET, Volume 11, Issue 05 (2024)
- Link: https://www.irjet.net/archives/V11/i5/IRJET-V11I5168.pdf
- Why it is useful:
  - It supports the practical idea of image-based weed detection and discusses approaches using image processing and ML for field analysis.
  - It is useful for showing how student and engineering implementations frame the problem.
- Caution:
  - This is better used as a supporting implementation-style citation, not as your main high-authority reference.

### C. Weed Detection Using Deep Learning Techniques: A Review

- Source: IJCRT, Volume 10, Issue 3 (2022)
- Link: https://ijcrt.org/papers/IJCRT2203197.pdf
- Why it is useful:
  - It summarizes crop/weed discrimination ideas such as Excess Green, image processing, ANN/CNN approaches, and site-specific spraying motivation.
  - It is helpful for explaining why image-based weed detection matters in agriculture.
- Caution:
  - Treat it as a secondary review source. For stronger academic weight, cite it after stronger journal sources such as DeepWeeds, the 2023 Elsevier review, and the 2023 Sensors review.

## Recommended citation strategy for your submission

If you want your project report to look stronger academically, use the references in this priority order:

1. DeepWeeds (Scientific Reports, 2019)
2. Machine learning for weed-plant discrimination in agriculture 5.0: An in-depth review (Artificial Intelligence in Agriculture, 2023)
3. Weed Detection Using Deep Learning: A Systematic Literature Review (Sensors, 2023)
4. CWFID dataset paper / dataset
5. The IRJET and IJCRT papers as supporting references
