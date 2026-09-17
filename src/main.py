"""
main.py
-------
Entry point for the OMR automated MCQ grading system.

Runs the full pipeline:
    raw photo
      -> preprocess (grayscale, Gaussian blur, Canny)
      -> detect sheet corners, apply perspective transform
      -> binarise, find bubbles, group into rows, read answers
      -> grade against the answer key, annotate, save

Usage
-----
    python src/main.py
    python src/main.py --image data/input/sheet_tilted.png
    python src/main.py --image path/to/sheet.png --key key.json
    python src/main.py --threshold otsu --debug
    python src/main.py --naive          # skip the warp, to show it failing
"""

import argparse
import json
import os
import sys

import cv2

from preprocessing import preprocess
from sheet_detection import find_sheet_contour, four_point_transform, draw_corners
from bubble_detection import extract
from grader import grade, annotate, add_scorecard, print_report

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default answer key: 0 = A, 1 = B, 2 = C, 3 = D, 4 = E
DEFAULT_KEY = [1, 4, 0, 3, 2, 1, 0, 4, 3, 2]


def load_key(path):
    """Load an answer key from JSON. Accepts indices or letters."""
    with open(path) as f:
        data = json.load(f)

    key = data["answers"] if isinstance(data, dict) else data

    if key and isinstance(key[0], str):
        key = [ord(k.strip().upper()) - ord("A") for k in key]

    return key


def parse_args():
    p = argparse.ArgumentParser(description="OMR MCQ automated grader")
    p.add_argument("--image", default=os.path.join(
        ROOT, "data", "input", "sheet_tilted.png"),
        help="path to the answer-sheet image")
    p.add_argument("--key", default=None,
                   help="path to a JSON answer key")
    p.add_argument("--choices", type=int, default=5,
                   help="number of options per question")
    p.add_argument("--threshold", choices=["adaptive", "otsu"],
                   default="adaptive", help="binarisation method")
    p.add_argument("--fill", type=float, default=0.55,
                   help="minimum fill ratio to count a bubble as marked")
    p.add_argument("--margin", type=float, default=0.20,
                   help="minimum gap between best and runner-up bubble")
    p.add_argument("--naive", action="store_true",
                   help="skip perspective correction (demonstrates failure)")
    p.add_argument("--debug", action="store_true",
                   help="save intermediate stage images")
    p.add_argument("--outdir", default=os.path.join(ROOT, "data", "output"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"ERROR: could not read image '{args.image}'")

    key = load_key(args.key) if args.key else DEFAULT_KEY

    print(f"Image      : {args.image}")
    print(f"Resolution : {image.shape[1]} x {image.shape[0]}")
    print(f"Threshold  : {args.threshold}")
    print(f"Mode       : {'NAIVE (no warp)' if args.naive else 'full pipeline'}")

    # ---------------------------------------------------- stage 1 + 2
    if args.naive:
        # Deliberately grade the raw photo. On a tilted sheet the rows are
        # slanted, so row grouping collapses -- this is the failure our HOT
        # question is about.
        working = image
        corners = None
    else:
        gray, edged = preprocess(image)
        corners = find_sheet_contour(edged)

        if corners is None:
            sys.exit("ERROR: could not find a 4-corner sheet outline. "
                     "Try a plainer background or better lighting.")

        working, _ = four_point_transform(image, corners)
        print(f"Warped to  : {working.shape[1]} x {working.shape[0]}")

        if args.debug:
            cv2.imwrite(os.path.join(args.outdir, "01_edges.png"), edged)
            cv2.imwrite(os.path.join(args.outdir, "02_corners.png"),
                        draw_corners(image, corners))
            cv2.imwrite(os.path.join(args.outdir, "02_warped.png"), working)

    # ---------------------------------------------------- stage 3
    answers, thresh, rows = extract(
        working,
        num_choices=args.choices,
        method=args.threshold,
        fill_threshold=args.fill,
        margin=args.margin,
    )
    print(f"Rows found : {len(rows)}")

    if args.debug:
        cv2.imwrite(os.path.join(args.outdir, "03_thresh.png"), thresh)

    if len(rows) != len(key):
        print(f"\nWARNING: detected {len(rows)} question rows but the answer "
              f"key has {len(key)} entries.")
        if args.naive:
            print("This is exactly the failure mode the perspective transform "
                  "exists to prevent: on a tilted photo the bubble rows are "
                  "slanted, so they cannot be grouped into clean rows.")
        sys.exit(1)

    # ---------------------------------------------------- stage 4
    result = grade(answers, key)
    print_report(result)

    vis = annotate(working, rows, result)
    vis = add_scorecard(vis, result)

    name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.outdir, f"{name}_graded.png")
    cv2.imwrite(out_path, vis)
    print(f"Annotated sheet saved to: {out_path}")

    json_path = os.path.join(args.outdir, f"{name}_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result JSON saved to    : {json_path}")


if __name__ == "__main__":
    main()
