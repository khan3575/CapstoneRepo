# Project Cleanup and Organization Plan

## Files to KEEP (Essential)

### Core Source Code (src/)
- ✅ dataset.py - Data loading
- ✅ gnn_model.py - Model architecture
- ✅ graph_construction.py - Graph building
- ✅ train_brats_gnn.py - Training script
- ✅ train_cv_fold.py - CV training
- ✅ evaluation.py - Metrics
- ✅ visualization.py - Plotting
- ✅ aggregate_cv_results.py - Results aggregation

### Essential Scripts (scripts/)
- ✅ train_unet_baseline.py - Baseline comparison
- ✅ analyze_time_complexity.py - Complexity analysis
- ✅ analyze_space_complexity.py - Space analysis
- ✅ create_qualitative_examples.py - Visualizations
- ✅ run_ablation_study.py - Ablation experiments

### Key Results (research_results/)
- ✅ cv_analysis/ - 5-fold CV results (616K) - MAIN RESULTS
- ✅ ablation_study/ - Ablation results (16M) - NEEDS REVIEW
- ✅ baseline_comparison/ - U-Net comparison (12K)
- ✅ qualitative_examples/ - Visualizations (2.5M)
- ✅ plots/ - Performance plots (460K)
- ✅ mathematical_formulation.md - Theory (20K)
- ✅ PUBLICATION_READINESS.md - Summary

### Documentation
- ✅ README.md - Main documentation
- ✅ requirements.txt - Dependencies
- ✅ paper_ieee_format.tex - Paper draft

## Files to DELETE (Redundant/Temporary)

### Duplicate Scripts (root level - move to archive/)
- ❌ ablation_study.py (duplicate of scripts/run_ablation_study.py)
- ❌ baseline_comparison.py (duplicate of scripts/train_unet_baseline.py)
- ❌ comprehensive_evaluation.py (already done)
- ❌ evaluate_model.py (use src/evaluation.py)
- ❌ run_ablation_study.py (duplicate)
- ❌ run_baseline_comparison.py (duplicate)
- ❌ run_comprehensive_evaluation.py (duplicate)
- ❌ fix_inconsistent_graphs.py (one-time fix, no longer needed)
- ❌ test_installation.py (installation done)

### Planning Documents (archive/)
- ❌ ACCELERATED_1MONTH_PLAN.md
- ❌ CONTINUOUS_PLAN.md
- ❌ DAILY_TRACKER.md
- ❌ EXECUTION_ROADMAP.md
- ❌ JOURNAL_SUBMISSION_ANALYSIS.md
- ❌ NEXT_STEPS_ACTION_PLAN.md
- ❌ PROGRESS_TRACKER.md
- ❌ WEEK1_CHECKLIST.md
- ❌ WEEK1_GUIDE.md
- ❌ WEEK1_PACKAGE.md
- ❌ research_enhancement_plan.md
- ❌ research_paper_template.md
- ❌ START_HERE.md
- ❌ PIPELINE_SUMMARY.md
- ❌ TECHNICAL_DOCUMENTATION.md

### Temporary/Generated Files
- ❌ *.log files (13 files)
- ❌ *.aux, *.out files (LaTeX temp)
- ❌ ablation_study_failed.log
- ❌ ablation_study_old.log
- ❌ ablation_study_slow.log
- ❌ __pycache__/ directories

### Status Files (ephemeral)
- ❌ ABLATION_STATUS.md (ablation done)
- ❌ QUICK_REFERENCE.txt (no longer needed)
- ❌ INSTALLATION_COMPLETE.md (installation done)
- ❌ check_ablation_progress.sh (monitoring done)

### Questionable Results (need review)
- ⚠️ research_results/ablation_studies/ (12K - different from ablation_study?)
- ⚠️ research_results/qualitative/ (8K - vs qualitative_examples?)
- ⚠️ research_results/baseline_comparison_report.md (shows Random Forest 100%!)

## Organization Structure (After Cleanup)

```
brats_gnn_segmentation/
├── src/                          # Core source code
├── scripts/                      # Analysis scripts
├── research_results/             # Essential results only
│   ├── cv_analysis/             # Main 5-fold CV results
│   ├── baseline_comparison/     # GNN vs U-Net
│   ├── ablation_study/          # Architecture ablations
│   ├── qualitative_examples/    # Visualizations
│   └── plots/                   # Figures
├── paper_ieee_format.tex        # Paper
├── README.md                    # Documentation
├── requirements.txt             # Dependencies
└── archive/                     # Moved old files
    ├── planning_docs/
    ├── duplicate_scripts/
    └── temp_logs/
```

## Action Items

1. Create archive/ directory
2. Move duplicate/planning files to archive/
3. Delete temporary files (.log, .aux, .out)
4. Review and fix inconsistent results
5. Keep only essential documentation

Total space to free: ~50MB of duplicates/logs
