"""
sheet_detection.py
------------------
Stage 2 of the OMR pipeline -- and the heart of our HOT question:

    "Why does an OMR system fail the moment you photograph the sheet with a
     phone instead of scanning it, and how do you fix it in software?"

When a sheet is photographed at an angle, the paper's rectangle is projected
onto the camera sensor as a general quadrilateral. Parallel edges stop being
parallel, and a bubble's pixel position no longer maps to a fixed grid cell.
Any grader that assumes "question 3 option B lives at (x, y)" breaks.

The fix is a homography: a 3x3 projective transform H such that

        [x']       [x]
    s * [y'] = H * [y]
        [1 ]       [1]

H has 8 degrees of freedom, so four point correspondences (the sheet's four
corners -> the four corners of a clean rectangle) determine it uniquely.
Applying H un-does the projection and gives us a top-down "bird's-eye" view.

Concepts used: contour detection, polygon approximation (Douglas-Peucker),
corner ordering, perspective transform.
"""

import cv2
import numpy as np


# --------------------------------------------------------------- corners
def order_corners(pts):
    """
    Put 4 corner points into a canonical order:
        [top-left, top-right, bottom-right, bottom-left]

    Trick: for the sum (x+y), the top-left is the smallest and the
    bottom-right is the largest. For the difference (y-x), the top-right is
    the smallest and the bottom-left is the largest. This works for any
    rotation up to ~45 degrees, which covers realistic hand-held photos.
    """
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right

    d = np.diff(pts, axis=1).ravel()  # y - x
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left

    return ordered


def find_sheet_contour(edged, min_area_ratio=0.15):
    """
    Locate the answer sheet's outline in an edge map.

    Strategy: take the largest contours by area, approximate each with a
    polygon, and accept the first one that reduces to exactly 4 vertices and
    covers a plausible fraction of the frame.

    Returns the 4 corner points, or None if no quadrilateral was found.
    """
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = edged.shape[0] * edged.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * frame_area:
            continue

        # Douglas-Peucker: simplify the contour until only corners remain.
        # epsilon is 2% of the perimeter -- large enough to absorb the
        # jitter of a hand-held photo, small enough to keep real corners.
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2))

    return None


# ------------------------------------------------------------ the warp
def target_dimensions(corners):
    """
    Choose output size from the quadrilateral itself: use the longest pair of
    opposite edges so nothing in the sheet gets squashed.
    """
    (tl, tr, br, bl) = corners

    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_w = int(max(width_bottom, width_top))

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_h = int(max(height_right, height_left))

    return max(max_w, 1), max(max_h, 1)


def four_point_transform(image, corners):
    """
    Apply the homography that maps the four detected corners onto a clean
    axis-aligned rectangle. This is the step that makes the sheet gradable.
    """
    corners = order_corners(corners)
    w, h = target_dimensions(corners)

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped, M


def detect_and_warp(image, edged):
    """
    Convenience wrapper: find the sheet, warp it flat.

    Returns (warped_image, corners) or (None, None) if detection failed.
    """
    corners = find_sheet_contour(edged)
    if corners is None:
        return None, None

    warped, _ = four_point_transform(image, corners)
    return warped, corners


def draw_corners(image, corners):
    """Visualise the detected quadrilateral -- great for the video demo."""
    vis = image.copy()
    pts = corners.astype(int).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 4)

    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), label in zip(corners.astype(int), labels):
        cv2.circle(vis, (x, y), 14, (0, 0, 255), -1)
        cv2.putText(vis, label, (x + 20, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return vis


if __name__ == "__main__":
    import os
    import sys
    from preprocessing import preprocess

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(root, "data", "input", "sheet_tilted.png")

    img = cv2.imread(src_path)
    if img is None:
        raise SystemExit(f"Could not read image: {src_path}")

    gray, edged = preprocess(img)
    corners = find_sheet_contour(edged)

    if corners is None:
        raise SystemExit("No quadrilateral sheet contour found.")

    print("Detected corners (TL, TR, BR, BL):")
    for name, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"  {name}: ({x:7.1f}, {y:7.1f})")

    warped, _ = four_point_transform(img, corners)
    vis = draw_corners(img, corners)

    out_dir = os.path.join(root, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "02_corners.png"), vis)
    cv2.imwrite(os.path.join(out_dir, "02_warped.png"), warped)

    print(f"\nOriginal : {img.shape[1]} x {img.shape[0]}")
    print(f"Warped   : {warped.shape[1]} x {warped.shape[0]}")
    print("Wrote    : data/output/02_corners.png, data/output/02_warped.png")
