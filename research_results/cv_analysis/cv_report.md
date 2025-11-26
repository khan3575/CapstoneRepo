# Cross-Validation Results Report

**Generated:** 2025-11-26 08:35:33

**Number of Folds:** 5

## Overall Performance

### Test Set Metrics (Mean ± Std)

| Metric | Mean | Std | 95% CI | Min | Max |
|--------|------|-----|--------|-----|-----|
| Dice | 0.9880 | 0.0038 | [0.9832, 0.9928] | 0.9828 | 0.9923 |
| Accuracy | 0.9976 | 0.0009 | [0.9965, 0.9986] | 0.9963 | 0.9985 |
| Sensitivity | 0.9861 | 0.0132 | [0.9698, 1.0025] | 0.9721 | 1.0000 |
| Specificity | 0.9988 | 0.0018 | [0.9966, 1.0011] | 0.9959 | 1.0000 |
| Precision | 0.9902 | 0.0149 | [0.9717, 1.0087] | 0.9663 | 1.0000 |

## Per-Fold Results

| Fold | Dice | Accuracy | Sensitivity | Specificity | Precision |
|------|------|----------|-------------|-------------|----------|
| 0 | 0.9828 | 0.9963 | 1.0000 | 0.9959 | 0.9663 |
| 1 | 0.9858 | 0.9972 | 0.9721 | 1.0000 | 1.0000 |
| 2 | 0.9880 | 0.9976 | 0.9764 | 1.0000 | 1.0000 |
| 3 | 0.9910 | 0.9982 | 0.9821 | 1.0000 | 1.0000 |
| 4 | 0.9923 | 0.9985 | 1.0000 | 0.9983 | 0.9848 |

## Statistical Analysis

### One-Sample t-test
- **Null Hypothesis:** Mean Dice ≤ 0.95
- **t-statistic:** 22.1365
- **p-value:** 0.000025
- **Result:** ✅ Significant (p < 0.05). Mean Dice is significantly > 0.95

### Normality Test (Shapiro-Wilk)
- **Statistic:** 0.9666
- **p-value:** 0.853204
- **Result:** Data is normally distributed (p > 0.05)

## Training Time Analysis

- **Total training time:** 23.89 hours
- **Average per fold:** 286.7 minutes
- **Std per fold:** 4.4 minutes

## Conclusion

The model achieved a mean Dice score of **0.9880 ± 0.0038** across 5-fold cross-validation. This performance is statistically significantly better than 0.95 (p < 0.05), demonstrating strong and consistent segmentation performance.

## Visualizations

- Box plots: `cv_boxplots.png`
- Dice per fold: `cv_dice_per_fold.png`
- Training curves: `cv_training_curves.png`
