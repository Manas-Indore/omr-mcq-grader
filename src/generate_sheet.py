"""
generate_sheet.py
-----------------
Generates synthetic OMR answer sheets for testing the grading pipeline.

Produces:
  1. A clean, top-down "scanned" sheet.
  2. A tilted / perspective-distorted "phone photo" version on a desk-like
     background -- this is the input that breaks a naive grader, and is the
     centrepiece of our HOT question.

Usage:
    python src/generate_sheet.py
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------- config
NUM_QUESTIONS = 10
NUM_CHOICES = 5                       # A B C D E
SHEET_W, SHEET_H = 800, 1000          # pixels of the clean sheet
MARGIN_X, MARGIN_Y = 90, 200
ROW_SPACING = 70
COL_SPACING = 110
BUBBLE_R = 22

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

# The "student's" marked answers (0-indexed: 0=A, 1=B, ...)
STUDENT_MARKS = [1, 4, 0, 3, 2, 1, 0, 4, 3, 2]


def draw_clean_sheet(marks=STUDENT_MARKS):
    """Render a top-down OMR sheet with the given bubbles shaded."""
    sheet = np.full((SHEET_H, SHEET_W, 3), 255, dtype=np.uint8)

    # --- outer border: gives the sheet a strong detectable contour
    cv2.rectangle(sheet, (25, 25), (SHEET_W - 25, SHEET_H - 25), (0, 0, 0), 3)

    # --- title block
    cv2.putText(sheet, "OMR ANSWER SHEET", (150, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(sheet, "Shade one bubble per question", (195, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
    cv2.line(sheet, (60, 155), (SHEET_W - 60, 155), (0, 0, 0), 2)

    # --- column headers (A B C D E)
    for c in range(NUM_CHOICES):
        cx = MARGIN_X + 60 + c * COL_SPACING
        cv2.putText(sheet, CHOICE_LABELS[c], (cx - 10, MARGIN_Y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # --- question rows
    for q in range(NUM_QUESTIONS):
        cy = MARGIN_Y + q * ROW_SPACING

        # question number
        cv2.putText(sheet, f"{q + 1:2d}.", (MARGIN_X - 55, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        for c in range(NUM_CHOICES):
            cx = MARGIN_X + 60 + c * COL_SPACING

            if marks[q] == c:
                # filled bubble
                cv2.circle(sheet, (cx, cy), BUBBLE_R, (0, 0, 0), -1)
            else:
                # empty bubble outline
                cv2.circle(sheet, (cx, cy), BUBBLE_R, (0, 0, 0), 2)

    return sheet


def warp_to_photo(sheet, canvas_size=(1100, 1300), seed=7):
    """
    Simulate a phone photo: place the sheet on a textured background and
    apply a perspective distortion so the sheet appears tilted / skewed.
    """
    rng = np.random.default_rng(seed)
    cw, ch = canvas_size

    # --- desk-like background with subtle noise so it isn't pure white
    canvas = np.full((ch, cw, 3), 165, dtype=np.uint8)
    noise = rng.normal(0, 7, (ch, cw, 3))
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    h, w = sheet.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # destination corners -> deliberately non-rectangular (perspective tilt)
    dst = np.float32([
        [210,  90],   # top-left
        [960, 215],   # top-right
        [870, 1215],  # bottom-right
        [120, 1040],  # bottom-left
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, M, (cw, ch),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

    # mask so the background shows through outside the sheet
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (cw, ch))
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) // 255

    photo = canvas * (1 - mask3) + warped * mask3
    photo = photo.astype(np.uint8)

    # --- mild lighting gradient (phone photos are never evenly lit)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    grad = 1.0 - 0.28 * (xx / cw) - 0.12 * (yy / ch)
    photo = np.clip(photo.astype(np.float32) * grad[..., None], 0, 255).astype(np.uint8)

    # --- slight blur + sensor noise
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    photo = np.clip(photo.astype(np.float32) +
                    rng.normal(0, 4, photo.shape), 0, 255).astype(np.uint8)

    return photo


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "input")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clean = draw_clean_sheet()
    cv2.imwrite(os.path.join(out_dir, "sheet_clean.png"), clean)

    photo = warp_to_photo(clean)
    cv2.imwrite(os.path.join(out_dir, "sheet_tilted.png"), photo)

    # a second student with different answers, also tilted
    other = draw_clean_sheet(marks=[1, 4, 0, 2, 2, 3, 0, 4, 1, 2])
    cv2.imwrite(os.path.join(out_dir, "sheet_student2.png"),
                warp_to_photo(other, seed=21))

    print("Generated in", out_dir)
    for f in ("sheet_clean.png", "sheet_tilted.png", "sheet_student2.png"):
        print("  -", f)
    print("\nStudent 1 marks (0=A):", STUDENT_MARKS)


if __name__ == "__main__":
    main()
