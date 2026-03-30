"""Create a composite qualitative results figure for the paper.
Combines best, median, and worst cases into a single figure.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE = Path("/mnt/bigdata/capstone/brats_gnn_segmentation/research_results/failure_cases")
OUT = Path("/mnt/bigdata/capstone/brats_gnn_segmentation/paperWriting/overleaf_flat")

cases = [
    ("best_BraTS2021_01594_z097.png", "Best Case — Dice: 100.0%", "(a)"),
    ("median_BraTS2021_00209_z059.png", "Median Case — Dice: 92.3%", "(b)"),
    ("worst_BraTS2021_01405_z108.png", "Worst Case — Dice: ≈0.0%", "(c)"),
    ("worst_BraTS2021_01366_z105.png", "Worst Case — Dice: ≈0.0%", "(d)"),
]

fig = plt.figure(figsize=(18, 12), facecolor='white')
gs = gridspec.GridSpec(4, 1, hspace=0.25, top=0.95, bottom=0.02, left=0.02, right=0.98)

for idx, (fname, title, label) in enumerate(cases):
    ax = fig.add_subplot(gs[idx])
    img = mpimg.imread(str(BASE / fname))
    ax.imshow(img)
    ax.set_title(f"{label}  {title}", fontsize=13, fontweight='bold', pad=8)
    ax.axis('off')

# Column headers at the very top
fig.text(0.15, 0.97, "T1CE MRI", ha='center', fontsize=11, fontstyle='italic', color='gray')
fig.text(0.38, 0.97, "Ground Truth", ha='center', fontsize=11, fontstyle='italic', color='#CC3333')
fig.text(0.62, 0.97, "GNN Prediction", ha='center', fontsize=11, fontstyle='italic', color='#3366CC')
fig.text(0.85, 0.97, "Comparison", ha='center', fontsize=11, fontstyle='italic', color='gray')

# Save high-res for paper
fig.savefig(str(OUT / "fig_H_qualitative_results.jpg"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig(str(OUT / "fig_H_qualitative_results.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved to {OUT / 'fig_H_qualitative_results.jpg'}")
print(f"Saved to {OUT / 'fig_H_qualitative_results.png'}")
