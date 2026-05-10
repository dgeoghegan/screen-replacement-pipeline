#!/usr/bin/env python3
# extract_ground_truth.py
import cv2
import json
import numpy as np
import argparse
from pathlib import Path

DIFF_THRESHOLD = 10
MIN_AREA = 500  # ignore tiny noise contours


def extract_ground_truth(src_path: str, tar_path: str, debug_dir: Path = None) -> dict | None:
    src = cv2.imread(src_path)
    tar = cv2.imread(tar_path)

    if src is None or tar is None:
        print(f"  Could not read pair: {src_path}")
        return None
    if src.shape != tar.shape:
        print(f"  Shape mismatch: {src_path}")
        return None

    diff = cv2.absdiff(src, tar)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

    if not contours:
        print(f"  No diff found: {src_path}")
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    bbox = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}

    # attempt quad extraction
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    quad = approx.reshape(-1, 2).tolist() if len(approx) == 4 else None

    if debug_dir:
        debug_img = src.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if quad:
            cv2.drawContours(debug_img, [np.array(quad)], -1, (180, 105, 255), 2)
        stem = Path(src_path).stem
        cv2.imwrite(str(debug_dir / f"{stem}_gt.jpg"), debug_img)

    result = {"bbox": bbox}
    if quad:
        result["quad"] = quad
    else:
        print(f"  Warning: could not extract clean quad for {Path(src_path).name} (got {len(approx)} points)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract ground truth from src/tar pairs")
    parser.add_argument("--src_dir", required=True, help="Directory of _src.jpg files")
    parser.add_argument("--tar_dir", required=True, help="Directory of _tar.jpg files")
    parser.add_argument("--output", default="ground_truth.json", help="Output JSON file")
    parser.add_argument("--debug", action="store_true", help="Save annotated debug images")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    tar_dir = Path(args.tar_dir)
    debug_dir = None

    if args.debug:
        debug_dir = Path("logs/gt_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}
    missing_tar = 0
    no_diff = 0
    no_quad = 0

    for src_path in sorted(src_dir.glob("*_src.jpg")):
        stem = src_path.stem.replace("_src", "")
        tar_path = tar_dir / f"{stem}_tar.jpg"

        if not tar_path.exists():
            print(f"  No tar found for {src_path.name}")
            missing_tar += 1
            continue

        print(f"Processing {src_path.name}")
        result = extract_ground_truth(str(src_path), str(tar_path), debug_dir)

        if result:
            ground_truth[src_path.name] = result
            has_quad = "quad" in result
            if not has_quad:
                no_quad += 1
            print(f"  bbox: {result['bbox']} quad: {'yes' if has_quad else 'NO'}")
        else:
            no_diff += 1
            ground_truth[src_path.name] = {"no_tv": True}
            print(f"  No diff — marking as no_tv: {src_path.name}")

    with open(args.output, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nWrote {len(ground_truth)} entries to {args.output}")
    print(f"  with quad: {len(ground_truth) - no_quad}  bbox only: {no_quad}  no diff: {no_diff}  missing tar: {missing_tar}")


if __name__ == "__main__":
    main()
