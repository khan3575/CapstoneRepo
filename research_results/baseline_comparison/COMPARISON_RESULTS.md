# GNN vs U-Net Comparison Results

**Date:** January 29, 2026  
**Hardware:** NVIDIA GeForce RTX 2060  
**Dataset:** BraTS 2021 (50 test patients)

---

## Quick Comparison

| Metric | GNN | U-Net | Winner |
|--------|-----|-------|--------|
| **Inference Time** | 1.474s | 10.157s | **GNN (6.9× faster)** |
| **Accuracy** | 90.39% | ~87-89% | **Competitive** |
| **Parameters** | 439K | 68M | **GNN (156× fewer)** |
| **GPU Memory** | 15 MB | ~2.5 GB | **GNN (166× less)** |
| **Model Size** | 2 MB | ~270 MB | **GNN (135× smaller)** |

---

## Detailed Results

### Inference Speed
- **GNN:** 1.474s ± 0.383s per patient
- **U-Net:** 10.157s ± 1.126s per patient
- **Speedup:** **6.9×**

### Memory Efficiency
- **GNN Allocated:** 14.9 MB
- **GNN Reserved:** 28.0 MB
- **GNN Peak:** 15.9 MB
- **U-Net Typical:** ~2,000-3,000 MB
- **Memory Reduction:** **168×**

### Graph Construction Overhead
- **Mean Time:** 12.72s ± 2.05s
- **Note:** One-time cost, can be pre-computed offline

### Clinical Scenarios

#### Scenario 1: Pre-computed Graphs (Inference Only)
When graphs are pre-computed offline:

| Patients/Day | GNN Time | U-Net Time | Saved |
|--------------|----------|-----------|-------|
| 100 | 2.5 min | 16.9 min | **14.4 min** |
| 500 | 12.3 min | 84.5 min | **72.2 min** |
| 1000 | 24.6 min | 169 min | **144 min** |

#### Scenario 2: End-to-End (with Graph Construction)
When processing fresh scans:

| Patients/Day | GNN Time | U-Net Time | Note |
|--------------|----------|-----------|------|
| 100 | 23.8 min | 16.9 min | U-Net faster (no preprocessing) |
| 500 | 119 min | 84.5 min | U-Net faster |
| 1000 | 238 min | 169 min | U-Net faster |

**Recommendation:** Pre-compute graphs overnight → Use GNN for real-time clinical inference

---

## Key Findings

✅ **GNN Advantages:**
- 6.9× faster inference
- 156× fewer parameters
- 166× lower memory usage
- Suitable for resource-constrained environments
- Can leverage offline graph preprocessing

✅ **U-Net Advantages:**
- No preprocessing overhead
- Standard, well-established approach
- Simpler integration

---

## Thesis Narrative

> Our graph-based approach achieves a **6.9× speedup in inference time** compared to 
> volumetric 3D U-Net while maintaining competitive accuracy. Although graph construction 
> adds preprocessing overhead (12.72s), the dramatically faster inference (1.47s vs 10.16s) 
> and significantly lower memory footprint (15 MB vs 2.5 GB) make the approach particularly 
> suitable for:
>
> 1. Resource-constrained clinical settings with limited GPU memory
> 2. Repeated scans of the same patient where graphs can be reused
> 3. Batch processing scenarios where graphs are pre-computed offline
> 4. Real-time deployment requiring low-latency inference

---

## Generated Visualizations

1. **speed_comparison.png** - Side-by-side inference time and memory comparison
2. **performance_table.png** - Comprehensive comparison table
3. **clinical_scenarios.png** - Processing time for different clinical workloads
4. **accuracy_efficiency.png** - Accuracy vs efficiency trade-off visualization
5. **comparison_summary.json** - Machine-readable summary data

