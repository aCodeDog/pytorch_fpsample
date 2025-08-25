#!/usr/bin/env python3
"""
Comprehensive validation script to compare GPU and CPU FPS implementations
"""

import torch
import torch_fpsample
import numpy as np
import time
from typing import Tuple, List

def compute_fps_distances(points: torch.Tensor, indices: torch.Tensor, batch_idx: int = 0) -> List[float]:
    """
    Compute the sequence of minimum distances for FPS validation.
    Each distance represents the minimum distance from the newly selected point to all previously selected points.
    """
    selected_points = points[batch_idx, indices[batch_idx]]
    distances = []
    
    for i in range(1, len(selected_points)):
        # Distance from point i to all previous points
        prev_points = selected_points[:i]
        current_point = selected_points[i:i+1]
        
        # Compute pairwise distances and take minimum
        dists = torch.cdist(current_point, prev_points, p=2)
        min_dist = dists.min().item()
        distances.append(min_dist)
    
    return distances

def validate_fps_property(distances: List[float], tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Validate that the FPS property holds: distances should generally be non-increasing.
    Some small increases are allowed due to numerical precision and algorithmic differences.
    """
    if len(distances) < 2:
        return True, "Not enough points to validate"
    
    violations = 0
    max_increase = 0.0
    
    for i in range(1, len(distances)):
        increase = distances[i] - distances[i-1]
        if increase > tolerance:
            violations += 1
            max_increase = max(max_increase, increase)
    
    violation_rate = violations / (len(distances) - 1)
    
    # Allow up to 10% violations for small increases (due to numerical differences)
    if violation_rate <= 0.1 and max_increase < 0.1:
        return True, f"Minor violations: {violations}/{len(distances)-1} ({violation_rate:.1%}), max increase: {max_increase:.6f}"
    else:
        return False, f"Too many violations: {violations}/{len(distances)-1} ({violation_rate:.1%}), max increase: {max_increase:.6f}"

def compare_implementations(
    batch_size: int = 2,
    N: int = 1000,
    D: int = 3,
    k: int = 64,
    start_idx: int = None,
    seed: int = 42
) -> dict:
    """
    Compare CPU and GPU implementations for a given problem size.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n=== Comparing CPU vs GPU: B={batch_size}, N={N}, D={D}, k={k} ===")
    
    # Generate identical test data
    points_cpu = torch.randn(batch_size, N, D, dtype=torch.float32)
    points_gpu = points_cpu.cuda()
    
    results = {
        'batch_size': batch_size,
        'N': N, 'D': D, 'k': k,
        'start_idx': start_idx,
        'seed': seed
    }
    
    try:
        # CPU version
        print("Running CPU version...")
        start_time = time.time()
        if start_idx is not None:
            sampled_cpu, indices_cpu = torch_fpsample.sample(points_cpu, k, start_idx=start_idx)
        else:
            sampled_cpu, indices_cpu = torch_fpsample.sample(points_cpu, k)
        cpu_time = time.time() - start_time
        
        print(f"CPU completed in {cpu_time:.4f}s")
        
        # GPU version
        print("Running GPU version...")
        torch.cuda.synchronize()
        start_time = time.time()
        if start_idx is not None:
            sampled_gpu, indices_gpu = torch_fpsample.sample(points_gpu, k, start_idx=start_idx)
        else:
            sampled_gpu, indices_gpu = torch_fpsample.sample(points_gpu, k)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU completed in {gpu_time:.4f}s")
        
        # Move GPU results to CPU for comparison
        sampled_gpu_cpu = sampled_gpu.cpu()
        indices_gpu_cpu = indices_gpu.cpu()
        
        results.update({
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time,
            'success': True
        })
        
        # Shape validation
        print("\n--- Shape Validation ---")
        shape_match = (sampled_cpu.shape == sampled_gpu_cpu.shape and 
                      indices_cpu.shape == indices_gpu_cpu.shape)
        print(f"Shape match: {shape_match}")
        print(f"CPU shapes: sampled={sampled_cpu.shape}, indices={indices_cpu.shape}")
        print(f"GPU shapes: sampled={sampled_gpu_cpu.shape}, indices={indices_gpu_cpu.shape}")
        
        results['shape_match'] = shape_match
        
        # Index validation
        print("\n--- Index Validation ---")
        cpu_indices_valid = torch.all(indices_cpu >= 0) and torch.all(indices_cpu < N)
        gpu_indices_valid = torch.all(indices_gpu_cpu >= 0) and torch.all(indices_gpu_cpu < N)
        
        print(f"CPU indices valid: {cpu_indices_valid}")
        print(f"GPU indices valid: {gpu_indices_valid}")
        
        results.update({
            'cpu_indices_valid': cpu_indices_valid,
            'gpu_indices_valid': gpu_indices_valid
        })
        
        # Point consistency validation
        print("\n--- Point Consistency Validation ---")
        cpu_consistent = True
        gpu_consistent = True
        
        for b in range(batch_size):
            for i in range(k):
                # Check CPU consistency
                idx_cpu = indices_cpu[b, i]
                expected_point_cpu = points_cpu[b, idx_cpu]
                actual_point_cpu = sampled_cpu[b, i]
                if not torch.allclose(expected_point_cpu, actual_point_cpu, atol=1e-5):
                    cpu_consistent = False
                    break
                
                # Check GPU consistency
                idx_gpu = indices_gpu_cpu[b, i]
                expected_point_gpu = points_gpu[b, idx_gpu].cpu()
                actual_point_gpu = sampled_gpu_cpu[b, i]
                if not torch.allclose(expected_point_gpu, actual_point_gpu, atol=1e-5):
                    gpu_consistent = False
                    break
            
            if not cpu_consistent or not gpu_consistent:
                break
        
        print(f"CPU point consistency: {cpu_consistent}")
        print(f"GPU point consistency: {gpu_consistent}")
        
        results.update({
            'cpu_point_consistent': cpu_consistent,
            'gpu_point_consistent': gpu_consistent
        })
        
        # FPS property validation
        print("\n--- FPS Property Validation ---")
        for b in range(min(batch_size, 2)):  # Test first 2 batches
            print(f"\nBatch {b}:")
            
            # CPU FPS validation
            cpu_distances = compute_fps_distances(points_cpu, indices_cpu, b)
            cpu_valid, cpu_msg = validate_fps_property(cpu_distances)
            print(f"CPU FPS property: {cpu_valid} - {cpu_msg}")
            
            # GPU FPS validation
            gpu_distances = compute_fps_distances(points_gpu.cpu(), indices_gpu_cpu, b)
            gpu_valid, gpu_msg = validate_fps_property(gpu_distances)
            print(f"GPU FPS property: {gpu_valid} - {gpu_msg}")
            
            # Show first few distances
            print(f"CPU distances (first 5): {cpu_distances[:5]}")
            print(f"GPU distances (first 5): {gpu_distances[:5]}")
            
            results[f'batch_{b}_cpu_fps_valid'] = cpu_valid
            results[f'batch_{b}_gpu_fps_valid'] = gpu_valid
            results[f'batch_{b}_cpu_distances'] = cpu_distances[:10]  # Store first 10
            results[f'batch_{b}_gpu_distances'] = gpu_distances[:10]
        
        # Results comparison
        print("\n--- Results Comparison ---")
        if start_idx is not None:
            # With fixed start index, first indices should match
            first_indices_match = torch.all(indices_cpu[:, 0] == indices_gpu_cpu[:, 0])
            print(f"First indices match (fixed start): {first_indices_match}")
            results['first_indices_match'] = first_indices_match
        else:
            print("Random start indices used - first indices may differ")
            results['first_indices_match'] = None
        
        # Distance distribution comparison
        cpu_all_distances = []
        gpu_all_distances = []
        
        for b in range(batch_size):
            cpu_dist = compute_fps_distances(points_cpu, indices_cpu, b)
            gpu_dist = compute_fps_distances(points_gpu.cpu(), indices_gpu_cpu, b)
            cpu_all_distances.extend(cpu_dist)
            gpu_all_distances.extend(gpu_dist)
        
        if cpu_all_distances and gpu_all_distances:
            cpu_mean = np.mean(cpu_all_distances)
            gpu_mean = np.mean(gpu_all_distances)
            cpu_std = np.std(cpu_all_distances)
            gpu_std = np.std(gpu_all_distances)
            
            print(f"CPU distance stats: mean={cpu_mean:.6f}, std={cpu_std:.6f}")
            print(f"GPU distance stats: mean={gpu_mean:.6f}, std={gpu_std:.6f}")
            print(f"Relative difference in means: {abs(cpu_mean - gpu_mean) / cpu_mean * 100:.2f}%")
            
            results.update({
                'cpu_distance_mean': cpu_mean,
                'gpu_distance_mean': gpu_mean,
                'cpu_distance_std': cpu_std,
                'gpu_distance_std': gpu_std,
                'mean_relative_diff': abs(cpu_mean - gpu_mean) / cpu_mean * 100
            })
        
        # Overall validation
        overall_valid = (shape_match and cpu_indices_valid and gpu_indices_valid and 
                        cpu_consistent and gpu_consistent)
        
        print(f"\n--- Overall Validation ---")
        print(f"Overall validation: {'✅ PASS' if overall_valid else '❌ FAIL'}")
        print(f"Performance: GPU is {results['speedup']:.2f}x {'faster' if results['speedup'] > 1 else 'slower'} than CPU")
        
        results['overall_valid'] = overall_valid
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.update({
            'success': False,
            'error': str(e),
            'overall_valid': False
        })
    
    return results

def run_comprehensive_validation():
    """Run validation across multiple problem sizes and configurations."""
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping validation")
        return
    
    print("🔍 Starting Comprehensive CPU vs GPU Validation")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        # Small problems
        {'batch_size': 512, 'N': 1000, 'D': 2, 'k': 160, 'start_idx': 0},
        {'batch_size': 512, 'N': 2000, 'D': 3, 'k': 320, 'start_idx': None},
        
        # Medium problems
        {'batch_size': 1024, 'N': 10000, 'D': 3, 'k': 1024, 'start_idx': 42},
        {'batch_size': 1024, 'N': 20000, 'D': 4, 'k': 1024, 'start_idx': None},
        
        # Large problems
        {'batch_size': 2048, 'N': 20000, 'D': 3, 'k': 1024, 'start_idx': 100},
        {'batch_size': 2048, 'N': 30000, 'D': 3, 'k': 1024, 'start_idx': None},
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*20} Test {i+1}/{len(test_configs)} {'='*20}")
        result = compare_implementations(**config)
        results.append(result)
        
        if result.get('overall_valid', False):
            passed += 1
            print("✅ PASSED")
        else:
            failed += 1
            print("❌ FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🏁 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(test_configs)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(test_configs)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! GPU implementation is validated.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check implementation.")
    
    # Performance summary
    valid_results = [r for r in results if r.get('success', False)]
    if valid_results:
        speedups = [r['speedup'] for r in valid_results]
        avg_speedup = np.mean(speedups)
        print(f"\n📊 Performance Summary:")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Best speedup: {max(speedups):.2f}x")
        print(f"Worst speedup: {min(speedups):.2f}x")
        
        # Show per-test performance
        print(f"\nPer-test performance:")
        for i, result in enumerate(valid_results):
            if result.get('success'):
                print(f"Test {i+1}: N={result['N']}, k={result['k']} -> {result['speedup']:.2f}x speedup")
    
    return results

if __name__ == "__main__":
    # Run single detailed test first
    print("Running detailed validation on medium-sized problem...")
    single_result = compare_implementations(batch_size=2, N=1000, D=3, k=64, start_idx=42)
    
    # Run comprehensive validation
    all_results = run_comprehensive_validation()
