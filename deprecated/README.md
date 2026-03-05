# Deprecated Scripts and Code

This directory contains old/unused scripts that are no longer actively maintained but kept for historical reference.

## Moved on: February 9, 2026

### `src/train_brats_gnn.py`
- **Reason**: Replaced by `src/train_cv_fold.py` which handles cross-validation properly
- **Status**: No longer used in final experiments
- **Keep?**: For reference only

### `src/train_maxpower.py`
- **Reason**: Replaced by `src/train_cv_fold.py` which is the official training script
- **Status**: No longer used in final experiments
- **Keep?**: For reference only

### `scripts/visualize_qualitative.py`
- **Reason**: Superseded by `scripts/simple_qualitative_viz.py` and `scripts/create_qualitative_examples.py`
- **Status**: Redundant visualization script
- **Keep?**: Can be deleted if other viz scripts work

### `scripts/analyze_time_complexity.py`
- **Reason**: Time complexity analysis not included in final thesis results
- **Status**: Optional analysis, not referenced in FINAL_THESIS_RESULTS.md
- **Keep?**: For future analysis only

### `scripts/analyze_space_complexity.py`
- **Reason**: Space complexity analysis not included in final thesis results
- **Status**: Optional analysis, not referenced in FINAL_THESIS_RESULTS.md
- **Keep?**: For future analysis only

---

## Notes

- These files are **not updated** to use the new `config.yaml` system
- If you need to use them, they will need to be adapted
- The actively maintained scripts are in `src/` and `scripts/`
