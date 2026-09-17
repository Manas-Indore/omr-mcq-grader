"""
preprocessing.py
----------------
Stage 1 of the OMR pipeline.

Turns a raw photograph into (a) a denoised grayscale image and (b) a clean
edge map that the sheet-detection stage can find contours in.

Concepts used (see DA1 literature survey):
  * Grayscale conversion  - drop colour, keep intensity structure
  * Gaussian blur         - low-pass filter, kills sensor/paper noise so that
                            noise is not mistaken for an edge
  * Canny edge detection  - gradient based multi-stage edge detector
"""

import cv2
import numpy as np


def to_grayscale(image):
    """Convert a BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray, ksize=5, sigma=0):
    """
    Apply a Gaussian low-pass filter.

    ksize must be odd. Larger ksize = more smoothing, but too much will
    soften the sheet border we are trying to detect.
    """
    if ksize % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd")
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def auto_canny(image, sigma=0.33):
    """
    Canny edge detection with thresholds derived from the image's median
    intensity, instead of hand-tuned magic numbers. This makes the pipeline
    far more robust to the lighting changes you get in phone photos.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def preprocess(image, blur_ksize=5, use_auto_canny=True,
               canny_lo=75, canny_hi=200):
    """
    Full stage-1 pipeline.

    Returns
    -------
    gray   : denoised grayscale image
    edged  : binary edge map
    """
    gray = to_grayscale(image)
    blurred = denoise(gray, ksize=blur_ksize)

    if use_auto_canny:
        edged = auto_canny(blurred)
    else:
        edged = cv2.Canny(blurred, canny_lo, canny_hi)

    # Closing joins small gaps in the sheet border so findContours sees one
    # continuous loop rather than several broken arcs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return gray, edged


if __name__ == "__main__":
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(root, "data", "input", "sheet_tilted.png")

    img = cv2.imread(src_path)
    if img is None:
        raise SystemExit(f"Could not read image: {src_path}")

    gray, edged = preprocess(img)

    out_dir = os.path.join(root, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "01_gray.png"), gray)
    cv2.imwrite(os.path.join(out_dir, "01_edges.png"), edged)

    print("Input :", src_path)
    print("Shape :", img.shape)
    print("Wrote : data/output/01_gray.png, data/output/01_edges.png")
