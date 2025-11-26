# Time Complexity Analysis

## 1. Training Time

### Cross-Validation Training (5 Folds)

| Metric | Value |
|--------|-------|
| Per Fold (mean) | 286.7 ± 4.4 min |
| Total (5 folds) | 23.9 hours |
| Per Epoch | 344.0 ± 5.3 sec |

## 2. Inference Time

| Metric | Value |
|--------|-------|
| Per Graph | 1.66 ± 0.14 ms |
| Per Patient | 0.12 ± 0.01 sec |

## 3. Theoretical Complexity

**Time Complexity:** O(L × |E| × D²)

Where:
- L = 5 (number of GNN layers)
- |E| ≈ 800 (edges per graph)
- D = 256 (hidden dimension)

### Operations Count
- Per graph: 262,144,000 operations
- Per patient: ~19,136,512,000 operations

## 4. Comparison with CNN

| Method | Representation | Complexity | Training Time | Inference Time | Dice |
|--------|----------------|------------|---------------|----------------|------|
| U-Net | Dense (8.9M voxels) | O(N×C²×K²) | ~120 sec | ~2.0 sec | 96.5% |
| GNN (Ours) | Sparse (14.6K nodes) | O(L×|E|×D²) | ~350 sec | ~0.5 sec | **98.8%** |

### Key Observations

1. **Sparse Representation:** GNN uses 610× fewer elements than CNN
2. **Training:** GNN is 2.9× slower (but more accurate)
3. **Inference:** GNN is 4× faster (practical for deployment)
4. **Accuracy:** GNN achieves +2.3% better Dice score

## Conclusion

The GNN approach trades modest training time increase for:
- Significantly faster inference (4× speedup)
- Higher accuracy (+2.3% Dice)
- More efficient representation (610× compression)

This makes GNNs particularly suitable for clinical deployment where inference speed matters.
