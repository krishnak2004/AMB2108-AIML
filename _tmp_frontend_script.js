const dropArea = document.getElementById("drop-area");
const uploadInput = document.getElementById("fileElem");
const previewContainer = document.getElementById("preview-container");
const imagePreview = document.getElementById("image-preview");
const validationMessage = document.getElementById("validation-message");
const processingNote = document.getElementById("processing-note");

const loadingArea = document.getElementById("loading-area");
const loadingText = document.getElementById("loading-text");
const resultArea = document.getElementById("result-area");
const canvas = document.getElementById("hidden-canvas");
const ctx = canvas.getContext("2d");

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];
const MAX_FILE_SIZE = 6 * 1024 * 1024;
const MIN_FILE_SIZE = 20 * 1024;
const ABSOLUTE_MIN_DIMENSION = 128;
const PREFERRED_MIN_DIMENSION = 224;
const MAX_ASPECT_RATIO = 2.8;
const MIN_ASPECT_RATIO = 0.45;

let analyzedMetrics = null;
let lastPrediction = null;
let currentProcessingNote = "";

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
}

["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add("highlight"), false);
});

["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove("highlight"), false);
});

dropArea.addEventListener("drop", handleDrop, false);

function handleDrop(event) {
    handleFiles(event.dataTransfer.files);
}

function handleFiles(files) {
    if (!files || files.length === 0) return;

    const file = files[0];
    clearValidation();
    currentProcessingNote = "";
    resultArea.classList.add("hidden");
    loadingArea.classList.add("hidden");

    if (!ACCEPTED_TYPES.includes(file.type)) {
        showValidation("Upload only PNG, JPEG, or WEBP crop-field images.");
        return;
    }

    if (file.size < MIN_FILE_SIZE) {
        showValidation("This image is too small or too compressed. Upload a clearer crop-field photo.");
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        showValidation("This image is too large. Keep the upload size under 6 MB.");
        return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
        const imageSource = reader.result;
        imagePreview.src = imageSource;
        previewContainer.classList.remove("hidden");

        const img = new Image();
        img.onload = () => {
            const metrics = extractImageMetrics(img);
            const validation = validateImageForAnalysis(img, metrics);

            if (!validation.ok) {
                showValidation(validation.message);
                return;
            }

            analyzedMetrics = metrics;
            currentProcessingNote = validation.note || "";
            dropArea.classList.add("hidden");
            loadingArea.classList.remove("hidden");
            simulateLoadingProcess();
        };
        img.onerror = () => {
            showValidation("The selected file could not be read as a valid image.");
        };
        img.src = imageSource;
    };
    reader.readAsDataURL(file);
}

function extractImageMetrics(img) {
    const MAX_DIM = 640;
    let scale = 1;
    if (Math.min(img.width, img.height) < PREFERRED_MIN_DIMENSION) {
        scale = Math.max(scale, PREFERRED_MIN_DIMENSION / Math.min(img.width, img.height));
    }
    if (Math.max(img.width, img.height) * scale > MAX_DIM) {
        scale = MAX_DIM / Math.max(img.width, img.height);
    }

    const drawWidth = Math.max(1, Math.round(img.width * scale));
    const drawHeight = Math.max(1, Math.round(img.height * scale));

    canvas.width = drawWidth;
    canvas.height = drawHeight;
    ctx.clearRect(0, 0, drawWidth, drawHeight);
    ctx.drawImage(img, 0, 0, drawWidth, drawHeight);

    const imageData = ctx.getImageData(0, 0, drawWidth, drawHeight);
    const data = imageData.data;
    const totalPixels = drawWidth * drawHeight;

    const grayscale = new Float32Array(totalPixels);
    const mask = new Uint8Array(totalPixels);

    let vegetationPixels = 0;
    let exgSum = 0;
    let exgSumSquares = 0;
    let overallBrightness = 0;

    for (let index = 0, pixelIndex = 0; index < data.length; index += 4, pixelIndex += 1) {
        const r = data[index] / 255;
        const g = data[index + 1] / 255;
        const b = data[index + 2] / 255;

        const sum = Math.max(r + g + b, 1e-6);
        const rn = r / sum;
        const gn = g / sum;
        const bn = b / sum;
        const exg = (2 * gn) - rn - bn;

        const isVegetation = exg > 0.08 && g > (r * 0.95) && g > (b * 1.05);
        if (isVegetation) {
            vegetationPixels += 1;
            mask[pixelIndex] = 1;
            exgSum += exg;
            exgSumSquares += exg * exg;
        }

        const gray = (0.299 * r) + (0.587 * g) + (0.114 * b);
        grayscale[pixelIndex] = gray;
        overallBrightness += gray;
    }

    let diffSum = 0;
    let diffCount = 0;
    for (let row = 0; row < drawHeight; row += 1) {
        for (let col = 0; col < drawWidth; col += 1) {
            const currentIndex = (row * drawWidth) + col;
            if (col + 1 < drawWidth) {
                diffSum += Math.abs(grayscale[currentIndex] - grayscale[currentIndex + 1]);
                diffCount += 1;
            }
            if (row + 1 < drawHeight) {
                diffSum += Math.abs(grayscale[currentIndex] - grayscale[currentIndex + drawWidth]);
                diffCount += 1;
            }
        }
    }

    const vegetationCoverage = vegetationPixels / totalPixels;
    const exgMean = vegetationPixels > 0 ? (exgSum / vegetationPixels) : 0;
    const exgVariance = vegetationPixels > 0 ? Math.max((exgSumSquares / vegetationPixels) - (exgMean * exgMean), 0) : 0;
    const exgStd = Math.sqrt(exgVariance);
    const textureVariation = diffCount > 0 ? diffSum / diffCount : 0;
    const brightness = overallBrightness / totalPixels;

    const smallMask = downsampleMask(mask, drawWidth, drawHeight, 64);
    const componentCount = countConnectedComponents(smallMask.mask, smallMask.width, smallMask.height);
    const componentDensity = componentCount / Math.max((smallMask.width * smallMask.height) / 4096, 1);
    const rowAlignment = computeRowAlignment(mask, drawWidth, drawHeight);

    return {
        width: drawWidth,
        height: drawHeight,
        vegetationCoverage,
        excessGreenMean: exgMean,
        excessGreenStd: exgStd,
        textureVariation,
        componentDensity,
        rowAlignment,
        brightness,
    };
}

function downsampleMask(mask, width, height, maxSide) {
    const scale = Math.min(maxSide / Math.max(width, height), 1);
    const smallWidth = Math.max(1, Math.round(width * scale));
    const smallHeight = Math.max(1, Math.round(height * scale));
    const smallMask = new Uint8Array(smallWidth * smallHeight);

    for (let row = 0; row < smallHeight; row += 1) {
        for (let col = 0; col < smallWidth; col += 1) {
            const srcRow = Math.min(height - 1, Math.floor((row / smallHeight) * height));
            const srcCol = Math.min(width - 1, Math.floor((col / smallWidth) * width));
            smallMask[(row * smallWidth) + col] = mask[(srcRow * width) + srcCol];
        }
    }

    return { mask: smallMask, width: smallWidth, height: smallHeight };
}

function countConnectedComponents(mask, width, height) {
    const visited = new Uint8Array(mask.length);
    let componentCount = 0;

    for (let row = 0; row < height; row += 1) {
        for (let col = 0; col < width; col += 1) {
            const startIndex = (row * width) + col;
            if (!mask[startIndex] || visited[startIndex]) {
                continue;
            }

            componentCount += 1;
            const stack = [startIndex];
            visited[startIndex] = 1;

            while (stack.length > 0) {
                const current = stack.pop();
                const currentRow = Math.floor(current / width);
                const currentCol = current % width;

                for (let deltaRow = -1; deltaRow <= 1; deltaRow += 1) {
                    for (let deltaCol = -1; deltaCol <= 1; deltaCol += 1) {
                        if (deltaRow === 0 && deltaCol === 0) continue;

                        const nextRow = currentRow + deltaRow;
                        const nextCol = currentCol + deltaCol;
                        if (nextRow < 0 || nextRow >= height || nextCol < 0 || nextCol >= width) {
                            continue;
                        }

                        const nextIndex = (nextRow * width) + nextCol;
                        if (!mask[nextIndex] || visited[nextIndex]) {
                            continue;
                        }

                        visited[nextIndex] = 1;
                        stack.push(nextIndex);
                    }
                }
            }
        }
    }

    return componentCount;
}

function computeRowAlignment(mask, width, height) {
    const rowProfile = new Array(height).fill(0);
    const colProfile = new Array(width).fill(0);

    for (let row = 0; row < height; row += 1) {
        for (let col = 0; col < width; col += 1) {
            const value = mask[(row * width) + col];
            rowProfile[row] += value;
            colProfile[col] += value;
        }
    }

    for (let row = 0; row < height; row += 1) {
        rowProfile[row] /= Math.max(width, 1);
    }
    for (let col = 0; col < width; col += 1) {
        colProfile[col] /= Math.max(height, 1);
    }

    const smoothedRows = movingAverage(rowProfile, 11);
    const smoothedCols = movingAverage(colProfile, 11);
    const alignment = Math.max(computeStandardDeviation(smoothedRows), computeStandardDeviation(smoothedCols));
    return clamp(alignment / 0.22, 0, 1);
}

function movingAverage(values, windowSize) {
    const output = new Array(values.length).fill(0);
    const radius = Math.floor(windowSize / 2);

    for (let index = 0; index < values.length; index += 1) {
        let sum = 0;
        let count = 0;
        for (let offset = -radius; offset <= radius; offset += 1) {
            const current = index + offset;
            if (current < 0 || current >= values.length) continue;
            sum += values[current];
            count += 1;
        }
        output[index] = count > 0 ? (sum / count) : values[index];
    }

    return output;
}

function computeStandardDeviation(values) {
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
    return Math.sqrt(Math.max(variance, 0));
}

function validateImageForAnalysis(img, metrics) {
    const aspectRatio = img.width / img.height;
    const shorterSide = Math.min(img.width, img.height);

    if (shorterSide < ABSOLUTE_MIN_DIMENSION) {
        return {
            ok: false,
            message: "Image resolution is too low. Upload a field image whose shorter side is at least 128 pixels.",
        };
    }

    if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
        return { ok: false, message: "Image shape is unusual. Upload a normal landscape or portrait field image." };
    }

    if (metrics.brightness < 0.16) {
        return { ok: false, message: "The image is too dark for reliable weed detection. Use a brighter daylight field image." };
    }

    if (metrics.brightness > 0.94) {
        return { ok: false, message: "The image is overexposed. Upload an image with more balanced lighting." };
    }

    if (metrics.textureVariation < 0.02) {
        return { ok: false, message: "The image looks too blurred or too flat. Upload a clearer crop-field photo." };
    }

    if (metrics.vegetationCoverage < 0.03) {
        return { ok: false, message: "Not enough visible vegetation was found. Upload a crop image that clearly shows the plantation area." };
    }

    if (metrics.vegetationCoverage > 0.97) {
        return { ok: false, message: "The image is too close to the leaves. Upload a wider field view that shows crop rows." };
    }

    if (shorterSide < PREFERRED_MIN_DIMENSION) {
        return {
            ok: true,
            note: `This image was auto-adjusted from ${img.width} x ${img.height} so the shorter side reaches the preferred 224 px analysis size.`,
        };
    }

    return { ok: true, note: "" };
}

function computePrediction(metrics) {
    const coverageScore = normalize(metrics.vegetationCoverage, 0.12, 0.55);
    const fragmentationScore = normalize(metrics.componentDensity, 1.2, 8.0);
    const disorderScore = 1 - metrics.rowAlignment;
    const textureScore = normalize(metrics.textureVariation, 0.03, 0.14);
    const weakGreennessScore = normalize(0.30 - metrics.excessGreenMean, 0.0, 0.18);

    let weedProbability =
        0.18 +
        (0.24 * coverageScore) +
        (0.26 * fragmentationScore) +
        (0.22 * disorderScore) +
        (0.18 * textureScore) +
        (0.10 * weakGreennessScore);

    if (metrics.rowAlignment > 0.62 && metrics.componentDensity < 2.8 && metrics.textureVariation < 0.10) {
        weedProbability -= 0.22;
    }

    if (metrics.rowAlignment < 0.35 && metrics.componentDensity > 4.0 && metrics.vegetationCoverage > 0.12) {
        weedProbability += 0.18;
    }

    weedProbability = clamp(weedProbability, 0.03, 0.97);
    const isWeed = weedProbability >= 0.56;
    const confidence = isWeed ? weedProbability : (1 - weedProbability);

    return {
        isWeed,
        weedProbability,
        confidence,
    };
}

function simulateLoadingProcess() {
    const steps = [
        "Validating image suitability...",
        "Extracting vegetation mask...",
        "Computing field-pattern features...",
        "Estimating weed likelihood...",
    ];

    let step = 0;
    loadingText.innerText = steps[0];

    const interval = setInterval(() => {
        step += 1;
        if (step < steps.length) {
            loadingText.innerText = steps[step];
        } else {
            clearInterval(interval);
            showResults();
        }
    }, 700);
}

function showResults() {
    if (!analyzedMetrics) {
        showValidation("No valid image is ready for analysis.");
        return;
    }

    lastPrediction = computePrediction(analyzedMetrics);
    loadingArea.classList.add("hidden");
    resultArea.classList.remove("hidden");

    const badge = document.getElementById("classification-badge");
    processingNote.textContent = currentProcessingNote;
    processingNote.classList.toggle("hidden", !currentProcessingNote);
    if (lastPrediction.confidence < 0.62) {
        badge.innerText = lastPrediction.isWeed ? "Possible Weed - Retake Image" : "Likely Crop - Retake Image";
    } else {
        badge.innerText = lastPrediction.isWeed ? "Detected: WEED LIKELY" : "Detected: CLEANER CROP";
    }
    badge.className = "badge " + (lastPrediction.isWeed ? "weed" : "crop");

    animateBar("bar-green", "val-green", analyzedMetrics.vegetationCoverage, 1.0);
    animateBar("bar-ndvi", "val-ndvi", analyzedMetrics.rowAlignment, 1.0);
    animateBar("bar-texture", "val-texture", analyzedMetrics.componentDensity, 10.0);
    animateBar("bar-conf", "val-conf", lastPrediction.confidence, 1.0, true);
}

function animateBar(barId, valueId, value, maxValue, isPercent = false) {
    const bar = document.getElementById(barId);
    const text = document.getElementById(valueId);

    setTimeout(() => {
        const percent = clamp((value / maxValue) * 100, 0, 100);
        bar.style.width = percent + "%";
        text.innerText = isPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
    }, 100);
}

function normalize(value, minValue, maxValue) {
    if (maxValue <= minValue) return 0;
    return clamp((value - minValue) / (maxValue - minValue), 0, 1);
}

function clamp(value, minValue, maxValue) {
    return Math.min(Math.max(value, minValue), maxValue);
}

function clearValidation() {
    validationMessage.textContent = "";
    validationMessage.classList.add("hidden");
}

function showValidation(message) {
    validationMessage.textContent = message;
    validationMessage.classList.remove("hidden");
    loadingArea.classList.add("hidden");
    resultArea.classList.add("hidden");
    processingNote.textContent = "";
    processingNote.classList.add("hidden");
    dropArea.classList.remove("hidden");
}

function resetApp() {
    analyzedMetrics = null;
    lastPrediction = null;
    currentProcessingNote = "";
    uploadInput.value = "";
    imagePreview.src = "";
    previewContainer.classList.add("hidden");
    clearValidation();
    resultArea.classList.add("hidden");
    loadingArea.classList.add("hidden");
    processingNote.textContent = "";
    processingNote.classList.add("hidden");
    dropArea.classList.remove("hidden");
    document.querySelectorAll(".fill").forEach((element) => {
        element.style.width = "0%";
    });
    document.querySelectorAll(".feature-val").forEach((element) => {
        element.innerText = "--";
    });
}
