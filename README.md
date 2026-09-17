# OMR-based Automated MCQ Grading System

Optical Mark Recognition (OMR) system that grades multiple-choice answer
sheets from an ordinary **phone photograph** — no flatbed scanner required.

Built with Python and OpenCV for the Computer Vision course (BCSE407L),
VIT Chennai.

---

## The problem this solves

> **Why does an OMR system fail the moment you photograph the sheet with a
> phone instead of scanning it — and how do you fix it in software?**

When a sheet is photographed at an angle, the paper's rectangle is projected
onto the camera sensor as a general quadrilateral. Parallel edges stop being
parallel, and a bubble's pixel position no longer maps to a fixed grid cell.
Any grader that assumes *"question 3, option B lives at (x, y)"* breaks
immediately.

The fix is a **homography** — a 3×3 projective transform `H` such that

```
    [x']       [x]
s * [y'] = H * [y]
    [1 ]       [1]
```

`H` has 8 degrees of freedom, so four point correspondences (the sheet's four
detected corners → the four corners of a clean rectangle) determine it
uniquely. Applying `H` undoes the projection and yields a top-down
"bird's-eye" view that can be graded reliably.

You can see the failure for yourself:

```bash
python src/main.py --naive     # skips the warp -> 0 rows detected
python src/main.py             # full pipeline   -> 10/10
```

---

## Pipeline

| Stage | Module | What happens |
|---|---|---|
| 1 | `preprocessing.py` | Grayscale → Gaussian blur → Canny edge detection |
| 2 | `sheet_detection.py` | Contour detection → 4-corner polygon approximation → perspective transform |
| 3 | `bubble_detection.py` | Adaptive thresholding → bubble contour filtering → row grouping → fill-ratio analysis |
| 4 | `grader.py` | Compare against answer key → score → annotated overlay |

### Secondary problem: what counts as "filled"?

A single global threshold fails on unevenly lit photos — the dark side of the
frame gets classified as ink. Otsu's method picks one global value by
minimising intra-class variance; **adaptive thresholding** computes a
threshold per neighbourhood, which is what actually survives a hand-held
photo. Compare them with `--threshold otsu` vs `--threshold adaptive`.

Bubbles are scored by **fill ratio** (ink pixels ÷ bubble area) rather than
raw pixel count, so the decision is independent of bubble size after warping.
The best bubble must also beat the runner-up by a margin, which is how
double-marked answers are flagged as `ambiguous` instead of being guessed.

---

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/omr-mcq-grader.git
cd omr-mcq-grader

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Generate the sample sheets (clean + tilted "phone photo" versions):

```bash
python src/generate_sheet.py
```

Grade a sheet:

```bash
python src/main.py
python src/main.py --image data/input/sheet_student2.png
python src/main.py --debug                    # save intermediate stages
python src/main.py --threshold otsu           # compare binarisation methods
python src/main.py --naive                    # demonstrate the failure case
```

Run an individual stage to inspect its output:

```bash
python src/preprocessing.py
python src/sheet_detection.py
python src/bubble_detection.py
```

### Custom answer key

Create `key.json`:

```json
{ "answers": ["B", "E", "A", "D", "C", "B", "A", "E", "D", "C"] }
```

```bash
python src/main.py --key key.json
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--image` | `data/input/sheet_tilted.png` | Input sheet |
| `--key` | built-in | JSON answer key |
| `--choices` | `5` | Options per question |
| `--threshold` | `adaptive` | `adaptive` or `otsu` |
| `--fill` | `0.55` | Min fill ratio to count as marked |
| `--margin` | `0.20` | Min gap between best and runner-up bubble |
| `--naive` | off | Skip perspective correction |
| `--debug` | off | Save intermediate images |

---

## Output

The grader writes an annotated sheet and a JSON result to `data/output/`:

- 🟢 **green** — correct
- 🔴 **red** — wrong (with the correct option circled in 🔵 blue)
- 🟠 **orange** — ambiguous (two bubbles marked)
- ⚪ **grey** — blank

---

## Project structure

```
omr-mcq-grader/
├── src/
│   ├── generate_sheet.py      # synthetic test-sheet generator
│   ├── preprocessing.py       # stage 1: filters + edges
│   ├── sheet_detection.py     # stage 2: corners + perspective transform
│   ├── bubble_detection.py    # stage 3: thresholding + bubble reading
│   ├── grader.py              # stage 4: scoring + annotation
│   └── main.py                # CLI entry point
├── data/
│   ├── input/                 # answer sheet images
│   └── output/                # graded results
├── docs/                      # DA1 report, screenshots
├── requirements.txt
└── README.md
```

---

## Limitations & future work

- Requires the full sheet to be visible against a contrasting background.
- Assumes a single rectangular sheet outline as the largest quadrilateral.
- Row grouping expects a consistent number of options per question.
- Extreme tilt (beyond ~45°) breaks the corner-ordering heuristic.
- Future: ArUco fiducial markers for robust registration, batch processing of
  multiple sheets, and handling of erased/re-marked bubbles.

---

## Concepts demonstrated

Grayscale conversion · Gaussian low-pass filtering · Canny edge detection ·
Morphological closing · Contour detection · Douglas–Peucker polygon
approximation · Corner ordering · Homography / perspective transform ·
Otsu binarisation · Adaptive thresholding · Shape descriptors (area, aspect
ratio, extent) · Masking and fill-ratio analysis
