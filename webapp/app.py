"""
app.py
------
Local web interface for the OMR grading pipeline.

Lets you upload a photo of a filled answer sheet in the browser and see the
graded result -- score, per-question breakdown, and the annotated image --
without touching the command line.

This wraps the exact same pipeline used by src/main.py; nothing about the
underlying computer-vision logic changes, only how you interact with it.

Run:
    python webapp/app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import sys
import time
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash
import cv2

# --- make src/ importable -------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from preprocessing import preprocess                              # noqa: E402
from sheet_detection import find_sheet_contour, four_point_transform, draw_corners  # noqa: E402
from bubble_detection import extract                              # noqa: E402
from grader import grade, annotate, add_scorecard                 # noqa: E402

# --- app setup --------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "omr-dev-secret"

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

LABELS = "ABCDE"
DEFAULT_KEY_STR = "B,E,A,D,C,B,A,E,D,C"
ALLOWED_EXT = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def parse_key(key_str, num_choices):
    """Turn 'B,E,A,D,C' into [1,4,0,3,2]."""
    letters = [k.strip().upper() for k in key_str.split(",") if k.strip()]
    key = []
    for letter in letters:
        if letter not in LABELS[:num_choices]:
            raise ValueError(
                f"'{letter}' is not a valid option (expected A-{LABELS[num_choices-1]})")
        key.append(LABELS.index(letter))
    return key


def run_pipeline(image_path, key, num_choices, threshold_method,
                 fill_threshold, margin, naive=False):
    """
    Runs the full stage 1-4 pipeline and returns a dict describing what
    happened at each step, suitable for rendering in the results template.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    info = {"resolution": f"{image.shape[1]} x {image.shape[0]}"}

    if naive:
        working = image
        corners = None
    else:
        gray, edged = preprocess(image)
        corners = find_sheet_contour(edged)
        if corners is None:
            raise ValueError(
                "Could not detect the sheet's four corners. Try a photo "
                "with better contrast against the background, or make sure "
                "all four edges of the sheet are visible.")
        working, _ = four_point_transform(image, corners)
        info["warped_resolution"] = f"{working.shape[1]} x {working.shape[0]}"

    answers, thresh, rows = extract(
        working, num_choices=num_choices, method=threshold_method,
        fill_threshold=fill_threshold, margin=margin)

    if len(rows) != len(key):
        raise ValueError(
            f"Detected {len(rows)} question row(s) but the answer key has "
            f"{len(key)} entries. "
            + ("This is the classic tilted-photo failure -- try without "
               "'naive mode', or retake the photo with all bubbles clearly "
               "visible." if naive else
               "Try adjusting the fill threshold, or check the photo covers "
               "the full sheet."))

    result = grade(answers, key)
    vis = annotate(working, rows, result)
    vis = add_scorecard(vis, result)

    uid = uuid.uuid4().hex[:8]
    corners_path = warped_path = None

    if corners is not None:
        corners_vis = draw_corners(image, corners)
        corners_path = f"results/{uid}_corners.jpg"
        cv2.imwrite(os.path.join(RESULT_DIR, f"{uid}_corners.jpg"), corners_vis)

        warped_path = f"results/{uid}_warped.jpg"
        cv2.imwrite(os.path.join(RESULT_DIR, f"{uid}_warped.jpg"), working)

    graded_path = f"results/{uid}_graded.jpg"
    cv2.imwrite(os.path.join(RESULT_DIR, f"{uid}_graded.jpg"), vis)

    return {
        "info": info,
        "result": result,
        "corners_img": corners_path,
        "warped_img": warped_path,
        "graded_img": graded_path,
    }


# --- routes ------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", default_key=DEFAULT_KEY_STR)


@app.route("/grade", methods=["POST"])
def grade_route():
    file = request.files.get("photo")
    if not file or file.filename == "":
        flash("Please choose an image to upload.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Please upload a PNG or JPG image.")
        return redirect(url_for("index"))

    key_str = request.form.get("key", DEFAULT_KEY_STR)
    num_choices = int(request.form.get("choices", 5))
    threshold_method = request.form.get("threshold", "adaptive")
    fill_threshold = float(request.form.get("fill", 0.55))
    margin = float(request.form.get("margin", 0.20))
    naive = request.form.get("naive") == "on"

    fname = f"{uuid.uuid4().hex[:8]}_{int(time.time())}.jpg"
    save_path = os.path.join(UPLOAD_DIR, fname)
    file.save(save_path)

    try:
        key = parse_key(key_str, num_choices)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("index"))

    try:
        output = run_pipeline(
            save_path, key, num_choices, threshold_method,
            fill_threshold, margin, naive=naive)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        info=output["info"],
        result=output["result"],
        corners_img=output["corners_img"],
        warped_img=output["warped_img"],
        graded_img=output["graded_img"],
        uploaded_img=f"uploads/{fname}",
        naive=naive,
    )


if __name__ == "__main__":
    print("\n  OMR Grader running -> http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
