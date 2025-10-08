# Ablation Study Results
Generated on: 2025-10-07 23:25:08

## Architecture Comparison

| Architecture | Dice Score | Accuracy | Parameters |
|--------------|------------|----------|------------|
| SAGE | 0.7076 | 0.9333 | 15,905 |
| GAT | 0.4903 | 0.8999 | 504,577 |
| GCN | 0.0262 | 0.8542 | 8,993 |

## Feature Importance

| Feature Group | Dice Score | Feature Count |
|---------------|------------|---------------|
| intensity_mean | 0.8189 | 4 |
| intensity_std | 0.6170 | 4 |
| spatial | 0.4004 | 3 |
| tumor_info | 0.9919 | 1 |
| all_features | 0.8315 | 12 |