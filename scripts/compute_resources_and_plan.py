#!/usr/bin/env python3
"""
Compute resource allocation based on user's choice
Generates taskset commands and PyTorch settings
"""

import subprocess
import math
import json
import sys
from pathlib import Path

def get_system_resources():
    """Get CPU and GPU information"""
    
    # Get CPU cores
    try:
        total_cores = int(subprocess.check_output(['nproc', '--all']).decode().strip())
    except:
        print("⚠️  Could not detect CPU cores, assuming 4")
        total_cores = 4
    
    # Get GPU info
    try:
        gpu_output = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total',
            '--format=csv,noheader,nounits'
        ]).decode().strip()
        
        gpu_info = []
        for line in gpu_output.split('\n'):
            parts = line.split(',')
            if len(parts) >= 3:
                gpu_info.append({
                    'index': int(parts[0].strip()),
                    'name': parts[1].strip(),
                    'memory_mb': int(parts[2].strip())
                })
    except:
        print("⚠️  Could not detect GPU, assuming 6GB GPU")
        gpu_info = [{'index': 0, 'name': 'Unknown', 'memory_mb': 6144}]
    
    return total_cores, gpu_info

def compute_allocation(use_percent_cpu, use_percent_gpu):
    """Compute resource allocation"""
    
    total_cores, gpu_info = get_system_resources()
    
    # CPU allocation
    cores_to_use = max(1, math.floor(total_cores * use_percent_cpu / 100.0))
    cores_reserved = total_cores - cores_to_use
    
    # PyTorch threads (use about half of allocated cores)
    torch_threads = max(1, cores_to_use // 2)
    
    # DataLoader workers (use about 1/4 of allocated cores)
    dataloader_workers = max(1, cores_to_use // 4)
    
    # GPU memory fraction
    gpu_memory_fraction = use_percent_gpu / 100.0
    
    # Generate taskset command
    if cores_to_use >= total_cores:
        taskset_cmd = ""  # Use all cores
        core_range = f"0-{total_cores-1}"
    else:
        core_range = f"0-{cores_to_use-1}"
        taskset_cmd = f"taskset -c {core_range}"
    
    results = {
        'system': {
            'total_cores': total_cores,
            'gpu_info': gpu_info
        },
        'allocation': {
            'cpu_percent': use_percent_cpu,
            'gpu_percent': use_percent_gpu,
            'cores_to_use': cores_to_use,
            'cores_reserved': cores_reserved,
            'torch_threads': torch_threads,
            'dataloader_workers': dataloader_workers,
            'gpu_memory_fraction': gpu_memory_fraction,
            'core_range': core_range
        },
        'commands': {
            'taskset': taskset_cmd,
            'nice': 'nice -n 10',
            'full_prefix': f"{taskset_cmd} nice -n 10" if taskset_cmd else "nice -n 10"
        },
        'pytorch_code': {
            'memory_fraction': f"torch.cuda.set_per_process_memory_fraction({gpu_memory_fraction:.2f}, device=0)",
            'num_threads': f"torch.set_num_threads({torch_threads})",
            'dataloader_workers': f"num_workers={dataloader_workers}"
        }
    }
    
    return results

def print_report(results, mode_name):
    """Print human-readable report"""
    
    print("="*80)
    print(f"RESOURCE ALLOCATION PLAN: {mode_name}")
    print("="*80)
    
    sys_info = results['system']
    alloc = results['allocation']
    cmds = results['commands']
    
    print(f"\n📊 SYSTEM RESOURCES:")
    print(f"   CPU Cores: {sys_info['total_cores']}")
    for gpu in sys_info['gpu_info']:
        print(f"   GPU {gpu['index']}: {gpu['name']} ({gpu['memory_mb']} MB)")
    
    print(f"\n📌 ALLOCATION ({alloc['cpu_percent']}% CPU, {alloc['gpu_percent']}% GPU):")
    print(f"   ✅ Cores to use: {alloc['cores_to_use']} (cores {alloc['core_range']})")
    print(f"   🚫 Cores reserved: {alloc['cores_reserved']} (for your browsing/work)")
    print(f"   🧵 PyTorch threads: {alloc['torch_threads']}")
    print(f"   👷 DataLoader workers: {alloc['dataloader_workers']}")
    print(f"   🎮 GPU memory: {alloc['gpu_memory_fraction']:.1%} ({int(sys_info['gpu_info'][0]['memory_mb'] * alloc['gpu_memory_fraction'])} MB)")
    
    print(f"\n💻 COMMAND PREFIX:")
    print(f"   {cmds['full_prefix']} python3 your_script.py")
    
    print(f"\n🐍 PYTORCH CODE TO ADD:")
    print(f"   {results['pytorch_code']['memory_fraction']}")
    print(f"   {results['pytorch_code']['num_threads']}")
    print(f"   DataLoader(..., {results['pytorch_code']['dataloader_workers']})")
    
    print("\n" + "="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compute_resources_and_plan.py <mode>")
        print("  mode: conservative (75%), aggressive (85%), full (95%), or custom:CPU%:GPU%")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    # Parse mode
    if mode == 'conservative':
        cpu_pct, gpu_pct = 75, 75
        mode_name = "CONSERVATIVE (75% CPU, 75% GPU)"
    elif mode == 'aggressive':
        cpu_pct, gpu_pct = 85, 85
        mode_name = "AGGRESSIVE (85% CPU, 85% GPU)"
    elif mode == 'full':
        cpu_pct, gpu_pct = 95, 95
        mode_name = "FULL POWER (95% CPU, 95% GPU) - USE DURING SLEEP"
    elif mode.startswith('custom:'):
        parts = mode.split(':')
        if len(parts) != 3:
            print("❌ Custom format: custom:CPU%:GPU%  (e.g., custom:80:85)")
            sys.exit(1)
        cpu_pct, gpu_pct = int(parts[1]), int(parts[2])
        mode_name = f"CUSTOM ({cpu_pct}% CPU, {gpu_pct}% GPU)"
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)
    
    # Compute allocation
    results = compute_allocation(cpu_pct, gpu_pct)
    
    # Print report
    print_report(results, mode_name)
    
    # Save to file
    output_file = f"/mnt/bigdata/capstone/brats_gnn_segmentation/logs/resource_plan_{mode.replace(':', '_')}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Plan saved to: {output_file}")

if __name__ == '__main__':
    main()
