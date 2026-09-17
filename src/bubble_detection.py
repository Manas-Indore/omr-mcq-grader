"""
bubble_detection.py
-------------------
Stage 3 of the OMR pipeline.

Once the sheet is flat we must (a) find every answer bubble, (b) group them
into question rows, and (c) decide which bubble in each row is shaded.

This stage contains our SECOND problem:

    "How does a computer decide a bubble is 'filled' when the photo has a
     lighting gradient, the student shades lightly, or the pencil strays
     outside the circle?"

A single global threshold fails on unevenly lit photos: the dark side of the
image gets classified as ink. Otsu's method picks one global value by
minimising intra-class variance -- better, but still global. Adaptive
thresholding computes a threshold per neighbourhood, which is what actually
survives a phone photo.

We then score each bubble by FILL RATIO (ink pixels / bubble area) rather
than raw pixel count, so the decision is independent of bubble size, and we
compare the best and second-best in each row to detect ambiguous answers.

Concepts used: adaptive thresholding, Otsu binarisation, contour filtering by
shape descriptors (area, aspect ratio, extent), spatial sorting, masking.
"""

import cv2
import numpy as np


# ----------------------------------------------------------- thresholding
def binarize(warped_gray, method="adaptive", block_size=41, C=10):
    """
    Produce a binary image where ink (shaded bubbles / outlines) is WHITE.

    method:
      "otsu"     - one global threshold chosen automatically
      "adaptive" - a local threshold per neighbourhood (default; robust to
                   the lighting gradients you get in hand-held photos)
    """
    if method == "otsu":
        _, thresh = cv2.threshold(
            warped_gray, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif method == "adaptive":
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, C)
    else:
        raise ValueError("method must be 'otsu' or 'adaptive'")

    return thresh


# -------------------------------------------------------- bubble finding
def _deduplicate_concentric(cands, dist_ratio=0.5):
    """
    An EMPTY bubble is a ring, so its outline yields TWO contours: the outer
    edge and the inner edge, sharing a centre. A FILLED bubble yields one.

    We keep only the outer (larger) contour of each concentric pair, so every
    bubble is represented exactly once regardless of whether it is shaded.
    """
    cands = sorted(cands, key=lambda b: b["w"] * b["h"], reverse=True)
    kept = []

    for b in cands:
        dup = False
        for k in kept:
            tol = max(k["w"], k["h"]) * dist_ratio
            if abs(b["cx"] - k["cx"]) < tol and abs(b["cy"] - k["cy"]) < tol:
                dup = True
                break
        if not dup:
            kept.append(b)

    return kept


def find_bubble_contours(thresh, min_r=10, max_r=60,
                         ar_tol=0.35, min_extent=0.55):
    """
    Keep only contours that look like answer bubbles.

    NOTE on retrieval mode: we use RETR_LIST, not RETR_EXTERNAL. The printed
    border of the answer sheet encloses every bubble, so with RETR_EXTERNAL
    the border is the only "external" contour and all 50 bubbles are silently
    discarded as its children. RETR_LIST returns every contour regardless of
    nesting; we then remove the duplicate inner rings ourselves.

    Shape descriptors used to reject text, borders and noise:
      * size      - bounding box within [min_r*2, max_r*2]
      * aspect    - w/h close to 1.0 (a circle is as wide as it is tall)
      * extent    - contour area / bounding-box area. A disc gives ~0.785
                    (pi/4), comfortably above a stray letter stroke.
    """
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cands = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if not (2 * min_r <= w <= 2 * max_r):
            continue
        if not (2 * min_r <= h <= 2 * max_r):
            continue

        ar = w / float(h)
        if abs(ar - 1.0) > ar_tol:
            continue

        rect_area = float(w * h)
        if rect_area == 0:
            continue
        extent = cv2.contourArea(c) / rect_area
        if extent < min_extent:
            continue

        cands.append({"c": c, "x": x, "y": y, "w": w, "h": h,
                      "cx": x + w / 2.0, "cy": y + h / 2.0})

    kept = _deduplicate_concentric(cands)
    return [b["c"] for b in kept]


def group_into_rows(bubbles, num_choices, row_tol_ratio=0.6):
    """
    Sort bubbles top-to-bottom, then split into question rows.

    Rather than assuming a fixed row height, we start a new row whenever the
    vertical gap to the previous bubble's centre exceeds a fraction of the
    typical bubble height. Rows are then sorted left-to-right so index 0 is
    always option A.
    """
    if not bubbles:
        return []

    info = []
    for c in bubbles:
        x, y, w, h = cv2.boundingRect(c)
        info.append({"c": c, "x": x, "y": y, "w": w, "h": h,
                     "cx": x + w / 2.0, "cy": y + h / 2.0})

    median_h = np.median([b["h"] for b in info])
    tol = median_h * row_tol_ratio

    info.sort(key=lambda b: b["cy"])

    rows, current = [], [info[0]]
    for b in info[1:]:
        if abs(b["cy"] - current[-1]["cy"]) <= tol:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)

    # left-to-right within each row; drop rows that don't have the expected
    # number of options (usually stray contours or a partial row)
    cleaned = []
    for r in rows:
        r.sort(key=lambda b: b["cx"])
        if len(r) == num_choices:
            cleaned.append(r)

    return cleaned


# --------------------------------------------------------- fill scoring
def fill_ratio(thresh, bubble):
    """
    Fraction of the bubble's disc that is ink.

    We build a filled-circle mask from the contour's bounding box and count
    white pixels inside it. Using a ratio (not a raw count) keeps the metric
    comparable across bubbles of slightly different sizes after warping.
    """
    x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]

    mask = np.zeros(thresh.shape, dtype="uint8")
    centre = (int(x + w / 2), int(y + h / 2))
    radius = int(min(w, h) / 2)
    cv2.circle(mask, centre, radius, 255, -1)

    area = cv2.countNonZero(mask)
    if area == 0:
        return 0.0

    inked = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
    return inked / float(area)


def read_answers(thresh, rows, fill_threshold=0.55, margin=0.20):
    """
    Decide the marked option for each question row.

    Returns a list of dicts, one per question:
        choice     - index of the marked option, or None
        status     - "ok" | "blank" | "ambiguous"
        ratios     - fill ratio of every option in the row

    Decision rule:
      * the best bubble must exceed `fill_threshold` (else BLANK)
      * it must beat the runner-up by at least `margin` (else AMBIGUOUS,
        i.e. the student appears to have marked two options)
    """
    results = []

    for row in rows:
        ratios = [fill_ratio(thresh, b) for b in row]
        order = np.argsort(ratios)[::-1]

        best, second = order[0], order[1] if len(order) > 1 else None
        best_r = ratios[best]
        second_r = ratios[second] if second is not None else 0.0

        if best_r < fill_threshold:
            status, choice = "blank", None
        elif (best_r - second_r) < margin:
            status, choice = "ambiguous", None
        else:
            status, choice = "ok", int(best)

        results.append({
            "choice": choice,
            "status": status,
            "ratios": [round(float(r), 3) for r in ratios],
        })

    return results


def extract(warped, num_choices=5, method="adaptive",
            fill_threshold=0.55, margin=0.20):
    """
    Full stage-3 pipeline: warped colour image -> per-question answers.

    Returns (answers, thresh, rows).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) \
        if len(warped.shape) == 3 else warped

    thresh = binarize(gray, method=method)
    bubbles = find_bubble_contours(thresh)
    rows = group_into_rows(bubbles, num_choices)
    answers = read_answers(thresh, rows,
                           fill_threshold=fill_threshold, margin=margin)

    return answers, thresh, rows


if __name__ == "__main__":
    import os
    import sys
    from preprocessing import preprocess
    from sheet_detection import detect_and_warp

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(root, "data", "input", "sheet_tilted.png")

    img = cv2.imread(src_path)
    if img is None:
        raise SystemExit(f"Could not read image: {src_path}")

    _, edged = preprocess(img)
    warped, _ = detect_and_warp(img, edged)
    if warped is None:
        raise SystemExit("Sheet not detected.")

    labels = "ABCDE"

    for method in ("otsu", "adaptive"):
        answers, thresh, rows = extract(warped, method=method)
        print(f"\n--- {method.upper()} ---")
        print(f"rows detected: {len(rows)}")
        for i, a in enumerate(answers, 1):
            mark = labels[a["choice"]] if a["choice"] is not None \
                else a["status"].upper()
            print(f"  Q{i:2d}: {mark:9s} ratios={a['ratios']}")

        out_dir = os.path.join(root, "data", "output")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"03_thresh_{method}.png"), thresh)
