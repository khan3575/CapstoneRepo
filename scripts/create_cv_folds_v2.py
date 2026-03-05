#!/usr/bin/env python3
"""
Phase 1.2 — Generate cv_folds_v2 from cv_pool Only
====================================================
Creates 5 stratified CV fold files using ONLY the 1,000-patient cv_pool.
The 251 held-out patients are NEVER included in any fold.

Output:
    data/cv_folds_v2/fold_0.json … fold_4.json
    data/cv_folds_v2/all_folds.json

Each fold has:
    - test_patients:  ~200 patients (1/5 of cv_pool)
    - val_patients:   ~80 patients  (10% of remaining 4/5)
    - train_patients: ~720 patients (remaining)

Verification:
    - No held-out patient in any fold (hard guarantee)
    - Non-overlapping test sets across folds
    - All 1,000 cv_pool patients appear in exactly one test set

Usage:
    python scripts/create_cv_folds_v2.py
"""

import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def compute_tumor_volumes(patient_ids, preprocessed_dir):
    """Return tumour voxel count per patient for quartile stratification."""
    volumes = {}
    errors = []
    print(f"Computing tumour volumes for {len(patient_ids)} patients...")
    for i, pid in enumerate(patient_ids):
        if i % 200 == 0:
            print(f"  {i}/{len(patient_ids)}")
        npz_path = Path(preprocessed_dir) / pid / f'{pid}_preprocessed.npz'
        if not npz_path.exists():
            errors.append(pid)
            volumes[pid] = 0
            continue
        try:
            data = np.load(npz_path)
            volumes[pid] = int((data['label'] > 0).sum())
        except Exception as e:
            print(f"  Warning: {pid}: {e}")
            errors.append(pid)
            volumes[pid] = 0
    if errors:
        print(f"  {len(errors)} patients had load errors (assigned volume=0)")
    return volumes


def get_graph_files(patient_id, graphs_dir):
    """Return graph file path for a patient (one .pt file per patient)."""
    gf = Path(graphs_dir) / patient_id / f'{patient_id}_graphs_200.pt'
    if gf.exists():
        return [str(gf)]
    return []


def create_stratified_folds(patient_ids, volumes, k, seed, val_ratio):
    """
    Create k stratified folds using tumour volume quartiles.

    Returns list of k dicts, each with keys: train, val, test (lists of patient IDs).
    """
    rng = random.Random(seed)
    vol_array = np.array([volumes[pid] for pid in patient_ids])
    q25, q50, q75 = np.percentile(vol_array, [25, 50, 75])

    def quartile(v):
        if v <= q25: return 0
        if v <= q50: return 1
        if v <= q75: return 2
        return 3

    # Group by quartile
    groups = defaultdict(list)
    for pid in patient_ids:
        groups[quartile(volumes[pid])].append(pid)

    # Within each quartile, shuffle and assign round-robin to k folds
    fold_buckets = [[] for _ in range(k)]
    for q in range(4):
        g = list(groups[q])
        rng.shuffle(g)
        for idx, pid in enumerate(g):
            fold_buckets[idx % k].append(pid)

    # For each fold i: test=fold_buckets[i], train+val=rest
    fold_results = []
    for fold_idx in range(k):
        test = sorted(fold_buckets[fold_idx])

        remaining = []
        for i in range(k):
            if i != fold_idx:
                remaining.extend(fold_buckets[i])

        # Shuffle remaining with fold-specific seed
        fold_rng = random.Random(seed + fold_idx)
        fold_rng.shuffle(remaining)

        n_val = int(len(remaining) * val_ratio)
        val = sorted(remaining[:n_val])
        train = sorted(remaining[n_val:])

        fold_results.append({'train': train, 'val': val, 'test': test})

    return fold_results


def main():
    parser = argparse.ArgumentParser(description='Generate cv_folds_v2 from cv_pool')
    parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='Validation fraction of train+val pool (default: 0.10)')
    parser.add_argument('--cv_pool', type=str, default='data/splits/cv_pool.json',
                        help='cv_pool.json from Phase 1.1')
    parser.add_argument('--held_out', type=str, default='data/splits/held_out_test.json',
                        help='held_out_test.json for cross-check')
    parser.add_argument('--preprocessed_dir', type=str, default='data/preprocessed')
    parser.add_argument('--graphs_dir', type=str, default='data/graphs')
    parser.add_argument('--output_dir', type=str, default='data/cv_folds_v2')
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    cv_pool_file = root / args.cv_pool
    held_out_file = root / args.held_out
    preprocessed_dir = root / args.preprocessed_dir
    graphs_dir = root / args.graphs_dir
    output_dir = root / args.output_dir

    print("=" * 70)
    print("Phase 1.2 — Generate cv_folds_v2")
    print("=" * 70)

    # ── Load cv_pool and held_out ─────────────────────────────────────────────
    if not cv_pool_file.exists():
        print(f"ERROR: {cv_pool_file} not found. Run Phase 1.1 first.")
        sys.exit(1)

    with open(cv_pool_file) as f:
        cv_pool_data = json.load(f)
    with open(held_out_file) as f:
        held_out_data = json.load(f)

    cv_patients = cv_pool_data['patients']
    held_out_patients = set(held_out_data['patients'])

    print(f"CV pool patients:  {len(cv_patients)}")
    print(f"Held-out patients: {len(held_out_patients)}")

    # Hard check: no cv patient is in held-out
    contamination = set(cv_patients) & held_out_patients
    assert len(contamination) == 0, f"CRITICAL: {len(contamination)} patients in both cv_pool and held_out!"
    print(f"✓ Zero contamination from held-out into cv_pool")

    # ── Compute volumes for stratification ────────────────────────────────────
    volumes = compute_tumor_volumes(cv_patients, preprocessed_dir)

    # ── Create stratified folds ───────────────────────────────────────────────
    print(f"\nCreating {args.k} stratified folds (seed={args.seed}, val_ratio={args.val_ratio})...")
    folds = create_stratified_folds(cv_patients, volumes, args.k, args.seed, args.val_ratio)

    # ── Verify: test sets are non-overlapping and cover all cv_pool ───────────
    all_test = []
    for fold in folds:
        all_test.extend(fold['test'])

    assert len(all_test) == len(set(all_test)), "CRITICAL: duplicate patients across test sets!"
    assert set(all_test) == set(cv_patients), \
        f"CRITICAL: test sets don't cover all cv_pool patients! " \
        f"({len(all_test)} in test sets vs {len(cv_patients)} in cv_pool)"

    print(f"✓ Test sets non-overlapping and cover all {len(cv_patients)} cv_pool patients")

    # ── Verify: no held-out patient in any fold ───────────────────────────────
    for i, fold in enumerate(folds):
        all_in_fold = set(fold['train']) | set(fold['val']) | set(fold['test'])
        leakage = all_in_fold & held_out_patients
        assert len(leakage) == 0, f"CRITICAL: Fold {i} contains {len(leakage)} held-out patients!"
    print(f"✓ No held-out patients in any fold")

    # ── Save fold files ───────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    all_folds_data = {}

    print(f"\nFold sizes:")
    for fold_idx, fold in enumerate(folds):
        train_graphs = []
        val_graphs = []
        test_graphs = []

        for pid in fold['train']:
            train_graphs.extend(get_graph_files(pid, graphs_dir))
        for pid in fold['val']:
            val_graphs.extend(get_graph_files(pid, graphs_dir))
        for pid in fold['test']:
            test_graphs.extend(get_graph_files(pid, graphs_dir))

        fold_data = {
            'fold_idx': fold_idx,
            'source': 'cv_pool_only',
            'held_out_excluded': True,
            'seed': args.seed,
            'train_patients': fold['train'],
            'val_patients': fold['val'],
            'test_patients': fold['test'],
            'train_graphs': train_graphs,
            'val_graphs': val_graphs,
            'test_graphs': test_graphs,
            'stats': {
                'n_train_patients': len(fold['train']),
                'n_val_patients': len(fold['val']),
                'n_test_patients': len(fold['test']),
                'n_train_graphs': len(train_graphs),
                'n_val_graphs': len(val_graphs),
                'n_test_graphs': len(test_graphs)
            }
        }

        fold_file = output_dir / f'fold_{fold_idx}.json'
        with open(fold_file, 'w') as f:
            json.dump(fold_data, f, indent=2)

        all_folds_data[f'fold_{fold_idx}'] = fold_data

        print(f"  Fold {fold_idx}: train={len(fold['train'])}, "
              f"val={len(fold['val'])}, test={len(fold['test'])} "
              f"| graphs: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")

    # Save all_folds.json
    all_folds_file = output_dir / 'all_folds.json'
    with open(all_folds_file, 'w') as f:
        json.dump({
            'k': args.k,
            'seed': args.seed,
            'val_ratio': args.val_ratio,
            'source': 'cv_pool_only',
            'held_out_excluded': True,
            'total_cv_patients': len(cv_patients),
            'total_held_out_patients': len(held_out_patients),
            'folds': all_folds_data
        }, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  fold_0.json … fold_4.json")
    print(f"  all_folds.json")

    print("\n" + "=" * 70)
    print("PHASE 1.2 COMPLETE")
    print("=" * 70)
    print("Verification summary:")
    print(f"  ✓ All folds use cv_pool patients only ({len(cv_patients)} patients)")
    print(f"  ✓ Held-out set ({len(held_out_patients)} patients) never appears in any fold")
    print(f"  ✓ Test sets non-overlapping, cover all cv_pool")
    print(f"\nNext: update config.yaml to point to data/cv_folds_v2/")
    print(f"Then: retrain all 5 folds (Phase 2)")


if __name__ == '__main__':
    main()
