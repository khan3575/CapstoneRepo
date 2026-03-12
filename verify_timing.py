"""
verify_timing.py — Confirm canonical end-to-end timing from comparison_summary.json.

The benchmark that produced the 6.9× speedup compared GNN and U-Net side-by-side in
the same run. The canonical GNN end-to-end time from that run is 1.47 s, which divides
evenly into U-Net 10.16 s to give 6.89× (reported as 6.9×).

The separate two_scenario_results.json measured 1.57 s in an isolated benchmark
(fold 0 only, 47 patients) and should NOT be used for the speedup claim.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).parent
COMPARISON = ROOT / "research_results/baseline_comparison/comparison_summary.json"
TWO_SCENARIO = ROOT / "research_results/timing_benchmark/two_scenario_results.json"


def main():
    with open(COMPARISON) as f:
        comp = json.load(f)

    gnn_time = comp["gnn_inference_time_s"]
    unet_time = comp["unet_inference_time_s"]
    speedup = unet_time / gnn_time

    print("=== comparison_summary.json (canonical benchmark) ===")
    print(f"  GNN end-to-end time : {gnn_time:.4f} s  → reported as 1.47 s")
    print(f"  U-Net time          : {unet_time:.4f} s  → reported as 10.16 s")
    print(f"  Speedup             : {speedup:.2f}×      → reported as 6.9×")
    print()

    with open(TWO_SCENARIO) as f:
        ts = json.load(f)

    e2e_ms = ts["end_to_end"]["mean_ms"]
    prebuilt_ms = ts["pre_built"]["mean_ms"]

    print("=== two_scenario_results.json (isolated benchmark, DO NOT use for speedup) ===")
    print(f"  End-to-end mean     : {e2e_ms:.1f} ms = {e2e_ms/1000:.3f} s")
    print(f"  Pre-built mean      : {prebuilt_ms:.1f} ms = {prebuilt_ms/1000:.3f} s")
    print()
    print("CONCLUSION: Use 1.47 s (from comparison_summary.json) for the 6.9× speedup claim.")
    print("            The 1.57 s is from a different benchmark run and must not be mixed in.")


if __name__ == "__main__":
    main()
