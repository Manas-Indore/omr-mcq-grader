"""
grader.py
---------
Stage 4 of the OMR pipeline.

Compares the detected answers against an answer key, computes the score, and
renders an annotated sheet:

    green  - correct
    red    - wrong (with the correct option circled in blue)
    orange - ambiguous (two bubbles marked)
    grey   - blank
"""

import cv2
import numpy as np

LABELS = "ABCDE"

COLOR_CORRECT = (0, 180, 0)      # BGR green
COLOR_WRONG = (0, 0, 220)        # red
COLOR_EXPECTED = (220, 120, 0)   # blue
COLOR_AMBIGUOUS = (0, 140, 255)  # orange
COLOR_BLANK = (130, 130, 130)    # grey


def grade(answers, answer_key):
    """
    Compare detected answers to the key.

    Parameters
    ----------
    answers    : list of dicts from bubble_detection.read_answers
    answer_key : list of ints (0 = A, 1 = B, ...), same length

    Returns a dict with per-question detail and summary counts.
    """
    if len(answers) != len(answer_key):
        raise ValueError(
            f"Detected {len(answers)} questions but the answer key has "
            f"{len(answer_key)}. Check that every row was detected.")

    detail = []
    correct = wrong = blank = ambiguous = 0

    for i, (a, expected) in enumerate(zip(answers, answer_key), start=1):
        status = a["status"]
        chosen = a["choice"]

        if status == "blank":
            verdict = "blank"
            blank += 1
        elif status == "ambiguous":
            verdict = "ambiguous"
            ambiguous += 1
        elif chosen == expected:
            verdict = "correct"
            correct += 1
        else:
            verdict = "wrong"
            wrong += 1

        detail.append({
            "question": i,
            "marked": LABELS[chosen] if chosen is not None else None,
            "expected": LABELS[expected],
            "verdict": verdict,
            "ratios": a["ratios"],
        })

    total = len(answer_key)
    return {
        "detail": detail,
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "ambiguous": ambiguous,
        "total": total,
        "score": correct,
        "percentage": round(100.0 * correct / total, 1) if total else 0.0,
    }


def annotate(warped, rows, result):
    """Draw the grading outcome onto the warped sheet."""
    vis = warped.copy() if len(warped.shape) == 3 else \
        cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    for row, d in zip(rows, result["detail"]):
        verdict = d["verdict"]
        expected_idx = LABELS.index(d["expected"])
        marked_idx = LABELS.index(d["marked"]) if d["marked"] else None

        if verdict == "correct":
            colour = COLOR_CORRECT
        elif verdict == "wrong":
            colour = COLOR_WRONG
        elif verdict == "ambiguous":
            colour = COLOR_AMBIGUOUS
        else:
            colour = COLOR_BLANK

        # highlight what the student marked
        if marked_idx is not None:
            b = row[marked_idx]
            centre = (int(b["cx"]), int(b["cy"]))
            radius = int(max(b["w"], b["h"]) / 2) + 6
            cv2.circle(vis, centre, radius, colour, 4)

        # if wrong, also show the expected option
        if verdict == "wrong":
            b = row[expected_idx]
            centre = (int(b["cx"]), int(b["cy"]))
            radius = int(max(b["w"], b["h"]) / 2) + 6
            cv2.circle(vis, centre, radius, COLOR_EXPECTED, 3)

        # if ambiguous or blank, mark the whole row
        if verdict in ("ambiguous", "blank"):
            x0 = int(row[0]["cx"]) - 40
            y0 = int(row[0]["cy"])
            cv2.putText(vis, verdict[:4].upper(), (x0 - 40, y0 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    return vis


def add_scorecard(vis, result):
    """Overlay a summary banner at the top of the annotated sheet."""
    out = vis.copy()
    h, w = out.shape[:2]

    banner_h = 90
    banner = np.full((banner_h, w, 3), 245, dtype=np.uint8)
    cv2.rectangle(banner, (0, 0), (w - 1, banner_h - 1), (60, 60, 60), 2)

    score_txt = f"SCORE: {result['score']}/{result['total']}  " \
                f"({result['percentage']}%)"
    cv2.putText(banner, score_txt, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

    breakdown = (f"correct {result['correct']}   wrong {result['wrong']}   "
                 f"blank {result['blank']}   ambiguous {result['ambiguous']}")
    cv2.putText(banner, breakdown, (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    return np.vstack([banner, out])


def print_report(result):
    """Human-readable console report."""
    print("\n" + "=" * 58)
    print(f"  SCORE: {result['score']}/{result['total']}  "
          f"({result['percentage']}%)")
    print("=" * 58)
    print(f"{'Q':>3}  {'MARKED':^8} {'EXPECTED':^9} {'VERDICT':^10}")
    print("-" * 58)

    for d in result["detail"]:
        marked = d["marked"] if d["marked"] else "-"
        print(f"{d['question']:>3}  {marked:^8} {d['expected']:^9} "
              f"{d['verdict']:^10}")

    print("-" * 58)
    print(f"correct={result['correct']}  wrong={result['wrong']}  "
          f"blank={result['blank']}  ambiguous={result['ambiguous']}")
    print("=" * 58 + "\n")
