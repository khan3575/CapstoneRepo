"""
Generate a 3-panel paper-ready methodology figure for GNN-based brain tumor segmentation.

Pipeline illustrated:
  Panel A — Original grayscale MRI slice
  Panel B — SLIC superpixel boundaries overlaid (K=200, m=0.1)
  Panel C — Region Adjacency Graph (node per superpixel centroid, edges between neighbors)

Usage:
    python scripts/generate_gnn_methodology_figure.py
    python scripts/generate_gnn_methodology_figure.py --output figures/methodology.png
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — safe on servers without a display

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import networkx as nx
from skimage import data, filters, exposure
from skimage.segmentation import slic, mark_boundaries
from skimage import graph as skgraph

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants matching the thesis parameters
# ---------------------------------------------------------------------------
N_SEGMENTS   = 200      # K = 200 superpixels
COMPACTNESS  = 0.1      # m = 0.1  (low → boundary-adherent)
SIGMA        = 1        # Gaussian smoothing before SLIC
DPI          = 300
FIGSIZE      = (15, 5)  # inches

# Aesthetic tokens
BG_COLOR     = "#0d0d0d"
NODE_COLOR   = "#ff6b35"   # vivid orange — high contrast on dark bg
EDGE_COLOR   = "#4ecdc4"   # teal
EDGE_ALPHA   = 0.45
NODE_SIZE    = 18
BOUNDARY_COLOR = (0.0, 0.85, 0.85)   # cyan in [0,1] RGB for mark_boundaries
LABEL_COLOR  = "white"
LABEL_FS     = 11
PANEL_LABELS = ["(a)", "(b)", "(c)"]
PANEL_TITLES = [
    "Original MRI Slice",
    "SLIC Superpixels  ($K={%d}$, $m={%.1f}$)" % (N_SEGMENTS, COMPACTNESS),
    "Region Adjacency Graph",
]


# ---------------------------------------------------------------------------
# 1. Synthesise a brain-like MRI slice
# ---------------------------------------------------------------------------
def make_mri_slice(size: int = 256) -> np.ndarray:
    """
    Return a synthetic 2-D float64 image that mimics a T1-weighted axial slice:
      - Gaussian blob for white-matter background
      - Brighter ellipse for grey-matter ring
      - Bright core simulating a tumour region with speckle noise
    """
    rng  = np.random.default_rng(42)
    Y, X = np.mgrid[-1:1:size*1j, -1:1:size*1j]

    # --- skull / background ---
    skull = np.exp(-((X/0.90)**2 + (Y/0.90)**2) * 2.5)

    # --- grey-matter ring ---
    r   = np.sqrt(X**2 + Y**2)
    gm  = np.exp(-((r - 0.62)**2) / (2 * 0.055**2))

    # --- white matter interior ---
    wm  = np.exp(-((X/0.55)**2 + (Y/0.55)**2) * 3.0) * 0.75

    # --- tumour core (bright, off-centre) ---
    tx, ty = 0.18, -0.14
    tumour  = np.exp(-(((X - tx)/0.12)**2 + ((Y - ty)/0.12)**2) * 8.0) * 1.0

    # --- tumour oedema halo ---
    oedema  = np.exp(-(((X - tx)/0.21)**2 + ((Y - ty)/0.21)**2) * 5.0) * 0.45

    # Composite
    img  = skull * 0.15 + gm * 0.55 + wm * 0.60 + oedema + tumour
    img += rng.normal(0, 0.018, img.shape)       # MRI-like Rician noise
    img  = np.clip(img, 0, None)

    # Mild unsharp-mask to sharpen tissue edges
    blurred = filters.gaussian(img, sigma=2)
    img     = img + 0.3 * (img - blurred)

    # CLAHE to bring out tissue contrast
    img = exposure.equalize_adapthist(img / img.max(), clip_limit=0.03)

    return img.astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Compute SLIC segments and RAG
# ---------------------------------------------------------------------------
def build_rag(image: np.ndarray):
    """Run SLIC and build the Region Adjacency Graph. Returns (segments, G)."""
    # SLIC expects float image in [0,1]
    img_float = image / image.max() if image.max() > 0 else image

    segments = slic(
        img_float,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=0,
        channel_axis=None,   # grayscale — no channel axis
    )

    # Build a mean-colour RAG
    rag = skgraph.rag_mean_color(img_float, segments)

    return segments, rag


def segment_centroids(segments: np.ndarray, rag: nx.Graph) -> dict:
    """Map each node in the RAG to its superpixel centroid (row, col)."""
    centroids = {}
    for node in rag.nodes():
        mask = segments == node
        ys, xs = np.where(mask)
        if len(ys):
            centroids[node] = (ys.mean(), xs.mean())
    return centroids


# ---------------------------------------------------------------------------
# 3. Draw the three panels
# ---------------------------------------------------------------------------
def draw_panels(image, segments, rag, output_path: Path):
    # Font — prefer Helvetica / Arial (common on most systems)
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "text.color":       LABEL_COLOR,
        "axes.titlecolor":  LABEL_COLOR,
        "figure.facecolor": BG_COLOR,
    })

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE,
                             facecolor=BG_COLOR,
                             gridspec_kw={"wspace": 0.04})

    H, W = image.shape

    # ── Panel A: Original ────────────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos", aspect="equal")

    # ── Panel B: Superpixel Boundaries ───────────────────────────────────────
    ax = axes[1]
    boundary_img = mark_boundaries(
        np.stack([image, image, image], axis=-1),  # RGB version of grayscale
        segments,
        color=BOUNDARY_COLOR,
        mode="thick",
    )
    ax.imshow(boundary_img, interpolation="lanczos", aspect="equal")

    # ── Panel C: Region Adjacency Graph ──────────────────────────────────────
    ax = axes[2]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, alpha=0.30,
              interpolation="lanczos", aspect="equal")

    centroids = segment_centroids(segments, rag)

    # Draw edges first (so nodes sit on top)
    for u, v in rag.edges():
        if u in centroids and v in centroids:
            cy_u, cx_u = centroids[u]
            cy_v, cx_v = centroids[v]
            ax.plot([cx_u, cx_v], [cy_u, cy_v],
                    color=EDGE_COLOR, lw=0.55, alpha=EDGE_ALPHA, zorder=2)

    # Draw nodes
    node_xs = [centroids[n][1] for n in centroids]
    node_ys = [centroids[n][0] for n in centroids]
    ax.scatter(node_xs, node_ys,
               s=NODE_SIZE, c=NODE_COLOR, zorder=3,
               linewidths=0.0, edgecolors="none")

    # ── Shared styling for all axes ──────────────────────────────────────────
    for i, ax in enumerate(axes):
        ax.set_facecolor(BG_COLOR)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
            spine.set_linewidth(0.8)

        # Sub-figure label e.g. "(a)" — top-left corner
        ax.text(0.022, 0.972, PANEL_LABELS[i],
                transform=ax.transAxes,
                fontsize=LABEL_FS + 1, fontweight="bold",
                color=LABEL_COLOR, va="top", ha="left",
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground=BG_COLOR)
                ])

        # Title below the panel
        ax.set_title(PANEL_TITLES[i],
                     fontsize=LABEL_FS, pad=6,
                     color=LABEL_COLOR)

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"[✓] Figure saved → {output_path}  ({DPI} DPI)")


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate GNN methodology figure (Panels A/B/C)"
    )
    parser.add_argument(
        "--output", type=str,
        default="figures/gnn_methodology.png",
        help="Output file path (PNG or PDF)."
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print("[1/3] Synthesising MRI slice …")
    image = make_mri_slice(size=256)

    print("[2/3] Running SLIC (K=%d, m=%.1f) and building RAG …" %
          (N_SEGMENTS, COMPACTNESS))
    segments, rag = build_rag(image)
    actual_k = len(np.unique(segments))
    print(f"      → {actual_k} superpixels,  {rag.number_of_edges()} RAG edges")

    print("[3/3] Rendering figure …")
    draw_panels(image, segments, rag, output_path)


if __name__ == "__main__":
    main()
