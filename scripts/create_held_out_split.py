#!/usr/bin/env python3
"""
Phase 1.1 — Create Held-Out Test Split
=======================================
Creates a properly sealed held-out test set from the full 1,251-patient pool.

Strategy:
  - Stratify patients by tumour volume quartile (ensures representative distribution)
  - 20% (≈250 patients) → data/splits/held_out_test.json  (SEALED — never used in training)
  - 80% (≈1,001 patients) → data/splits/cv_pool.json        (all CV training uses only this)

After running:
  - Verify held_out_test.json has ~250 patients
  - Verify cv_pool.json has ~1,001 patients
  - Verify ZERO overlap between the two sets
  - Then run Phase 1.2: generate new fold files from cv_pool only

Usage:
    python scripts/create_held_out_split.py
    python scripts/create_held_out_split.py --held_out_frac 0.20 --seed 42
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def compute_tumor_volumes(patient_ids, preprocessed_dir):
    """
    Load each patient's label volume and return tumour voxel counts.
    Used for quartile-stratified splitting.
    """
    import numpy as np

    volumes = {}
    errors = []

    print(f"Computing tumour volumes for {len(patient_ids)} patients...")
    for i, pid in enumerate(patient_ids):
        if i % 100 == 0:
            print(f"  {i}/{len(patient_ids)}")

        npz_path = Path(preprocessed_dir) / pid / f'{pid}_preprocessed.npz'
        if not npz_path.exists():
            errors.append(pid)
            volumes[pid] = 0
            continue

        try:
            data = np.load(npz_path)
            tumor_voxels = int((data['label'] > 0).sum())
            volumes[pid] = tumor_voxels
        except Exception as e:
            print(f"  Warning: could not load {pid}: {e}")
            errors.append(pid)
            volumes[pid] = 0

    if errors:
        print(f"  Warning: {len(errors)} patients had load errors (assigned volume=0)")

    return volumes


def stratified_split(patient_ids, volumes, held_out_frac, seed):
    """
    Split patients into held_out and cv_pool using quartile stratification.

    Stratification ensures held-out set has the same tumour-size distribution
    as the training pool — no cherry-picking easy or hard cases.
    """
    rng = np.random.default_rng(seed)

    # Compute volume quartiles
    vol_array = np.array([volumes[pid] for pid in patient_ids])
    quartile_boundaries = np.percentile(vol_array, [25, 50, 75])

    def get_quartile(v):
        if v <= quartile_boundaries[0]:
            return 0
        elif v <= quartile_boundaries[1]:
            return 1
        elif v <= quartile_boundaries[2]:
            return 2
        else:
            return 3

    # Group patients by quartile
    quartile_groups = {0: [], 1: [], 2: [], 3: []}
    for pid in patient_ids:
        q = get_quartile(volumes[pid])
        quartile_groups[q].append(pid)

    held_out = []
    cv_pool = []

    for q in range(4):
        group = quartile_groups[q]
        rng.shuffle(group)
        n_held = round(len(group) * held_out_frac)
        held_out.extend(group[:n_held])
        cv_pool.extend(group[n_held:])

        print(f"  Quartile {q}: {len(group)} patients → {n_held} held-out, {len(group)-n_held} cv_pool")

    return sorted(held_out), sorted(cv_pool)


def main():
    parser = argparse.ArgumentParser(description='Create held-out test split')
    parser.add_argument('--held_out_frac', type=float, default=0.20,
                        help='Fraction of patients for held-out set (default: 0.20 → ~250 patients)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--preprocessed_dir', type=str,
                        default='data/preprocessed',
                        help='Directory with preprocessed .npz files')
    parser.add_argument('--folds_dir', type=str,
                        default='data/cv_folds',
                        help='Existing CV folds directory (to get full patient list)')
    parser.add_argument('--output_dir', type=str,
                        default='data/splits',
                        help='Output directory for held_out_test.json and cv_pool.json')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    preprocessed_dir = project_root / args.preprocessed_dir
    folds_dir = project_root / args.folds_dir
    output_dir = project_root / args.output_dir

    print("=" * 70)
    print("Phase 1.1 — Create Held-Out Test Split")
    print("=" * 70)
    print(f"Held-out fraction: {args.held_out_frac:.0%}")
    print(f"Random seed: {args.seed}")

    # ── Step 1: Collect all patients from existing fold files ─────────────────
    print("\nCollecting all patients from existing fold files...")
    all_patients = set()
    for i in range(5):
        fold_file = folds_dir / f'fold_{i}.json'
        if not fold_file.exists():
            print(f"  ERROR: {fold_file} not found")
            sys.exit(1)
        with open(fold_file) as f:
            d = json.load(f)
        all_patients.update(d['train_patients'])
        all_patients.update(d['val_patients'])
        all_patients.update(d['test_patients'])

    all_patients = sorted(all_patients)
    print(f"Total unique patients: {len(all_patients)}")

    # ── Step 2: Compute tumour volumes for stratification ─────────────────────
    volumes = compute_tumor_volumes(all_patients, preprocessed_dir)

    vol_values = [volumes[p] for p in all_patients]
    print(f"\nTumour volume statistics (voxels):")
    print(f"  Min:    {min(vol_values):,}")
    print(f"  Q25:    {int(np.percentile(vol_values, 25)):,}")
    print(f"  Median: {int(np.percentile(vol_values, 50)):,}")
    print(f"  Q75:    {int(np.percentile(vol_values, 75)):,}")
    print(f"  Max:    {max(vol_values):,}")
    print(f"  Zero-volume (no tumour): {sum(1 for v in vol_values if v == 0)}")

    # ── Step 3: Stratified split ──────────────────────────────────────────────
    print("\nPerforming stratified split by quartile...")
    held_out, cv_pool = stratified_split(all_patients, volumes, args.held_out_frac, args.seed)

    print(f"\nSplit results:")
    print(f"  Held-out test:  {len(held_out)} patients")
    print(f"  CV pool:        {len(cv_pool)} patients")
    print(f"  Total:          {len(held_out) + len(cv_pool)} patients")

    # ── Step 4: Verify zero overlap ───────────────────────────────────────────
    overlap = set(held_out) & set(cv_pool)
    assert len(overlap) == 0, f"CRITICAL: {len(overlap)} patients appear in both sets!"
    print(f"\n✓ Zero overlap verified between held-out and cv_pool")

    # ── Step 5: Verify all patients accounted for ─────────────────────────────
    total_split = set(held_out) | set(cv_pool)
    missing = set(all_patients) - total_split
    extra = total_split - set(all_patients)
    assert len(missing) == 0, f"CRITICAL: {len(missing)} patients missing from split!"
    assert len(extra) == 0, f"CRITICAL: {len(extra)} unknown patients in split!"
    print(f"✓ All {len(all_patients)} patients accounted for")

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    held_out_file = output_dir / 'held_out_test.json'
    cv_pool_file = output_dir / 'cv_pool.json'

    held_out_data = {
        'description': 'Held-out test set — SEALED. Never used in training or CV.',
        'n_patients': len(held_out),
        'seed': args.seed,
        'held_out_frac': args.held_out_frac,
        'stratification': 'tumour_volume_quartile',
        'patients': held_out
    }

    cv_pool_data = {
        'description': 'CV pool — all 5-fold cross-validation runs use ONLY these patients.',
        'n_patients': len(cv_pool),
        'seed': args.seed,
        'held_out_frac': args.held_out_frac,
        'stratification': 'tumour_volume_quartile',
        'patients': cv_pool
    }

    with open(held_out_file, 'w') as f:
        json.dump(held_out_data, f, indent=2)

    with open(cv_pool_file, 'w') as f:
        json.dump(cv_pool_data, f, indent=2)

    print(f"\nSaved:")
    print(f"  {held_out_file}")
    print(f"  {cv_pool_file}")

    # ── Step 7: Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1.1 COMPLETE")
    print("=" * 70)
    print(f"  Held-out patients: {len(held_out)}")
    print(f"  CV pool patients:  {len(cv_pool)}")
    print(f"  Overlap check:     PASSED (0 overlap)")
    print(f"  Coverage check:    PASSED (all {len(all_patients)} patients covered)")
    print(f"\nNext step: Run Phase 1.2 (generate cv_folds_v2) with:")
    print(f"  python scripts/create_cv_folds_v2.py")


if __name__ == '__main__':
    main()
