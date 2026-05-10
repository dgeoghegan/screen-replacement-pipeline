# Constrained Perception Pipeline for Screen Replacement in Unstructured Images

This was a Friday-to-Monday take-home. I had not done image processing before, had not integrated a vision model into an automated pipeline, and had done little beyond tinkering in Python. The goal was to detect TV screens in photographs and replace them with a provided image.

The results are imperfect. This document describes what I built, how I reasoned about it, and what I would change.

---

## System Model

This is a staged perception and validation pipeline that transforms unstructured images into constrained geometric modifications. Each stage reduces uncertainty through either deterministic computation (OpenCV) or probabilistic validation (YOLO / Gemini), with full traceability of intermediate states.

---

## What I Built First

The system was designed around a constraint: no improvement step is valid unless it is observable and measurable. Instrumentation—logging, mocking, ground truth scoring, and parallel execution—was implemented first to ensure all later changes could be evaluated against traceable system state.

The commit history reflects this order. Days one and two focused on pipeline structure, module separation, a mock mode for testing without API calls, and an IoU scoring system built from pixel-diffing the provided src/tar image pairs. Accuracy work came after.

Visual observability was part of the same investment. Every stage writes its intermediate state to disk as an image: YOLO detections with candidates annotated, the crop passed to Gemini, the OpenCV edge map under the selected lighting preset, and the highlighted quad sent for confirmation. Failures can be traced to a specific stage without debugging or rerunning the pipeline.

I would make the same call again.

---

## How It Works

Each image is processed through a fixed-stage pipeline where each step reduces uncertainty before transformation is applied.

### 1. Detection
YOLO scans the full image for TVs (class 62). If confidence exceeds a threshold, the bounding box is padded and passed forward. If YOLO finds nothing, Gemini is used as a fallback on the full image.

### 2. Classification
The detected region is sent to Gemini, which determines:
- whether a TV is present
- the lighting condition (standard, bright reflection, sharp angle, dim lighting, partial occlusion)

Images without a confirmed TV are skipped.

### 3. Quad Detection
The crop is processed with OpenCV edge detection using Canny parameters tuned per lighting preset. Contours are filtered by area and approximated to quadrilaterals, then mapped back to full-image coordinates.

The goal is recovering screen corners, not bounding boxes, which is necessary for correct perspective warping.

### 4. Confirmation
A highlighted candidate quad is sent to Gemini for validation. It rejects cases with diagonal artifacts, furniture anchoring, or incomplete boundaries.

If OpenCV candidates fail, Gemini is asked directly for quad coordinates.

### 5. Replacement
The confirmed quad defines a perspective transform. The replacement image is warped and composited into the original image.

Each image is retried up to five times before failure.

---

## Key Decisions

YOLO is used for detection with Gemini as fallback. YOLO handles most cases at low cost; Gemini covers edge cases. Reversing this would be too expensive at scale.

OpenCV is used for geometry instead of relying on Gemini bounding boxes. Bounding boxes are insufficient under perspective distortion. Lighting presets adjust contour sensitivity, including epsilon for polygon approximation.

A confirmation step is used before committing any geometry. This adds API cost but significantly reduces invalid placements.

Parallel workers were added mid-project to reduce iteration time. This was a development constraint, not an accuracy change.

A Gemini-based post-placement evaluator was built and later disabled. It lacked ground-truth geometric context and produced low-signal feedback relative to cost.

---

## Results and Metrics

Two metrics are tracked:

**success_pct**  
Process metric: any image where a quad was produced and a replacement saved. Includes incorrect placements.

**true_success_rate**  
Quality metric: IoU ≥ 0.5 against ground truth or correct skip of no-TV images.

Ground truth is generated via pixel-diffing of src/tar pairs to infer placement regions.

Coverage is incomplete. Some correct placements are not scored, leading to systematic undercounting.

Manual inspection suggests approximately ~85% correctness, though this is not fully verifiable due to ground truth limitations.

If ground truth coverage were complete, true_success_rate would be the primary metric.

---

## Failure Model

### Geometric Failures
Occasionally produces skewed or triangular outputs when non-convex contours are selected (often caused by reflections). A convexity check (`cv2.isContourConvex`) would prevent this class of error.

### Aspect Ratio Mismatch
Fixed replacement image leads to distortion when screen aspect ratios diverge significantly. Cropping the replacement prior to warping would correct this.

### Debug Artifact Leakage
Intermediate outputs (crops, edge maps, overlays) are written unconditionally to disk. These should be gated behind a debug flag.

---

## How to Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
````

Place replacement image:

```
./assets/replacement.jpg
```

Run pipeline:

```bash
python run.py --input_dir ./input_images --output_dir ./output_images --workers <N>
```

---

## Flags

| Flag             | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `--input_dir`    | Source images directory (required)                     |
| `--output_dir`   | Output directory (required)                            |
| `--workers`      | Parallel workers (default: 1)                          |
| `--evaluate`     | Gemini post-placement evaluation (disabled by default) |
| `--mock`         | Mock Gemini responses                                  |
| `--tv_noconfirm` | Force rejection in mock mode                           |
| `--compare`      | IoU evaluation against ground truth                    |

---

## Module Structure

* `run.py` — orchestration, threading, IoU scoring, stats logging
* `detector.py` — YOLO detection, Gemini fallback, classification, quad detection, confirmation loop
* `processor.py` — edge detection, contour filtering, perspective transform
* `evaluator.py` — optional Gemini-based evaluation (disabled)
* `ingestor.py` — image loading
* `gemini.py` — API wrapper
* `logger.py` — JSONL logging for runs and tokens
* `mock_gemini.py` — deterministic test fixtures

Development tools (unsupported): `test_run.py`, `analyze_runs.py`, `analyze_image_costs.py`, `extract_ground_truth.py`.

Dependency direction is strictly one-way:
`run → detector/processor/evaluator → gemini/ingestor`

---

## Logging

* `token_usage.jsonl` — per-call API usage
* `image_results.jsonl` — per-image outcomes and IoU
* `run_summary.jsonl` — aggregated run statistics
