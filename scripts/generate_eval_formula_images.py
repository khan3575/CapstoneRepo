"""
Generate transparent-background formula PNG images for the Evaluation Criteria slide.

Uses matplotlib mathtext (no usetex needed) for crisp, properly-spaced fractions.
All images have transparent backgrounds — drop directly onto dark slides.

Outputs:
  figures/formula_dice.png
  figures/formula_accuracy.png
  figures/formula_precision.png
  figures/formula_sensitivity.png
  figures/formula_specificity.png
  figures/formula_eval_all.png   ← all five stacked (single slide drop-in)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
GOLD  = "#FFD166"   # variable highlight (label names)
WHITE = "#FFFFFF"   # formula text
LBLUE = "#7EC8E3"   # italic label  (Precision, Accuracy …)
LGRAY = "#C8DCEC"   # secondary text / epsilon note


# ── individual formula renderer ───────────────────────────────────────────────
def save_formula(path, label_text, math_str,
                 fw=6.5, fh=1.8, lfs=26, mfs=30):
    """
    Render:   label_text  =  math_str
    on a transparent canvas.

    label_text : plain string  e.g. "Precision"
    math_str   : LaTeX math    e.g. r"\frac{TP}{TP + FP}"
    """
    fig = plt.figure(figsize=(fw, fh))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0, 0, 0, 0))
    ax.axis("off")

    # italic label
    ax.text(0.03, 0.52, label_text,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=lfs, style="italic", color=LBLUE,
            fontfamily="DejaVu Sans")

    # "=" sign
    ax.text(0.42, 0.52, "=",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=lfs, color=WHITE,
            fontfamily="DejaVu Sans")

    # fraction / math
    ax.text(0.52, 0.52, f"${math_str}$",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=mfs, color=WHITE)

    fig.savefig(path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor=(0, 0, 0, 0))
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Dice — wider canvas, extra note line ─────────────────────────────────────
def save_dice(path, fw=9.5, fh=2.4):
    fig = plt.figure(figsize=(fw, fh))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0, 0, 0, 0))
    ax.axis("off")

    ax.text(0.02, 0.62,
            r"$Dice\,(M_{pred},\,M_{GT})$",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=26, style="italic", color=LBLUE)

    ax.text(0.50, 0.62, "=",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=26, color=WHITE, fontfamily="DejaVu Sans")

    ax.text(0.54, 0.62,
            r"$\dfrac{2\,|M_{pred} \cap M_{GT}| + \varepsilon}"
            r"{|M_{pred}| + |M_{GT}| + \varepsilon}$",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=30, color=WHITE)

    ax.text(0.54, 0.13,
            r"$\varepsilon = 10^{-7}$  (smoothing constant)",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=16, color=LGRAY)

    fig.savefig(path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor=(0, 0, 0, 0))
    plt.close(fig)
    print(f"  Saved → {path}")


# ── combined all-in-one image ─────────────────────────────────────────────────
def save_combined(path, fw=10, fh=10):
    fig = plt.figure(figsize=(fw, fh))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0, 0, 0, 0))
    ax.axis("off")

    # rows: y positions from top (in axes-fraction coords)
    rows = [0.90, 0.72, 0.54, 0.36, 0.18]
    LFS  = 24   # label fontsize
    MFS  = 28   # math fontsize

    formulas = [
        # (label, latex_math)
        (r"$Dice$",
         r"\dfrac{2\,|M_{pred} \cap M_{GT}| + \varepsilon}{|M_{pred}| + |M_{GT}| + \varepsilon}"),
        ("Accuracy",
         r"\dfrac{TP + TN}{TP + TN + FP + FN}"),
        ("Precision",
         r"\dfrac{TP}{TP + FP}"),
        ("Sensitivity",
         r"\dfrac{TP}{TP + FN}"),
        ("Specificity",
         r"\dfrac{TN}{TN + FP}"),
    ]

    for y, (lbl, math) in zip(rows, formulas):
        # label
        ax.text(0.02, y, lbl,
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=LFS, style="italic", color=LBLUE,
                fontfamily="DejaVu Sans")
        # "="
        ax.text(0.36, y, "=",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=LFS, color=WHITE,
                fontfamily="DejaVu Sans")
        # formula
        ax.text(0.40, y, f"${math}$",
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=MFS, color=WHITE)

    fig.savefig(path, dpi=300, bbox_inches="tight",
                transparent=True, facecolor=(0, 0, 0, 0))
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
print("Generating evaluation formula images …")

save_dice(OUTDIR / "formula_dice.png")

save_formula(OUTDIR / "formula_accuracy.png",
             "Accuracy",
             r"\dfrac{TP + TN}{TP + TN + FP + FN}",
             fw=8.0, fh=1.9)

save_formula(OUTDIR / "formula_precision.png",
             "Precision",
             r"\dfrac{TP}{TP + FP}",
             fw=6.0, fh=1.9)

save_formula(OUTDIR / "formula_sensitivity.png",
             "Sensitivity",
             r"\dfrac{TP}{TP + FN}",
             fw=6.5, fh=1.9)

save_formula(OUTDIR / "formula_specificity.png",
             "Specificity",
             r"\dfrac{TN}{TN + FP}",
             fw=6.5, fh=1.9)

save_combined(OUTDIR / "formula_eval_all.png")

print("Done.")
