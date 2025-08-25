#!/usr/bin/env python3
"""
Test script for CUDA FPS implementation
"""

import torch
import torch_fpsample
import time
import numpy as np

def test_cuda_fps():
    """Test CUDA FPS implementation against CPU version"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return
    
    print("Testing CUDA FPS implementation...")
    
    # Test parameters
    batch_size = 2
    N = 1000  # number of points
    D = 3     # dimension
    k = 64    # points to sample
    
    # Generate random test data
    points_cpu = torch.randn(batch_size, N, D, dtype=torch.float32)
    points_gpu = points_cpu.cuda()
    
    print(f"Input shape: {points_cpu.shape}")
    print(f"Sampling {k} points from {N} points in {D}D")
    
    # Test CPU version
    print("\n--- Testing CPU version ---")
    start_time = time.time()
    sampled_points_cpu, indices_cpu = torch_fpsample.sample(points_cpu, k)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"CPU output shapes: {sampled_points_cpu.shape}, {indices_cpu.shape}")
    
    # Test GPU version
    print("\n--- Testing GPU version ---")
    start_time = time.time()
    sampled_points_gpu, indices_gpu = torch_fpsample.sample(points_gpu, k)
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"GPU output shapes: {sampled_points_gpu.shape}, {indices_gpu.shape}")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    
    # Move GPU results to CPU for comparison
    sampled_points_gpu_cpu = sampled_points_gpu.cpu()
    indices_gpu_cpu = indices_gpu.cpu()
    
    # Verify shapes match
    assert sampled_points_cpu.shape == sampled_points_gpu_cpu.shape
    assert indices_cpu.shape == indices_gpu_cpu.shape
    
    print("\n--- Validation ---")
    
    # Check that indices are valid
    assert torch.all(indices_cpu >= 0) and torch.all(indices_cpu < N)
    assert torch.all(indices_gpu_cpu >= 0) and torch.all(indices_gpu_cpu < N)
    print("✓ All indices are valid")
    
    # Check that sampled points match the indices
    for b in range(batch_size):
        for i in range(k):
            idx_cpu = indices_cpu[b, i]
            idx_gpu = indices_gpu_cpu[b, i]
            
            # Check CPU consistency
            expected_point_cpu = points_cpu[b, idx_cpu]
            actual_point_cpu = sampled_points_cpu[b, i]
            assert torch.allclose(expected_point_cpu, actual_point_cpu, atol=1e-5)
            
            # Check GPU consistency  
            expected_point_gpu = points_gpu[b, idx_gpu].cpu()
            actual_point_gpu = sampled_points_gpu_cpu[b, i]
            assert torch.allclose(expected_point_gpu, actual_point_gpu, atol=1e-5)
    
    print("✓ Sampled points match their indices")
    
    # Check that first points are the same (should be deterministic if same start_idx)
    print(f"CPU first indices: {indices_cpu[:, 0]}")
    print(f"GPU first indices: {indices_gpu_cpu[:, 0]}")
    
    # Validate FPS property: distances between consecutive samples should be decreasing
    def validate_fps_property(points, indices, batch_idx=0):
        """Check that FPS property holds: each new point is farthest from selected set"""
        selected_points = points[batch_idx, indices[batch_idx]]
        
        distances = []
        for i in range(1, len(selected_points)):
            # Distance from point i to all previous points
            prev_points = selected_points[:i]
            current_point = selected_points[i:i+1]
            
            # Compute pairwise distances
            dists = torch.cdist(current_point, prev_points).min()
            distances.append(dists.item())
        
        return distances
    
    cpu_distances = validate_fps_property(points_cpu, indices_cpu)
    gpu_distances = validate_fps_property(points_gpu.cpu(), indices_gpu_cpu)
    
    print(f"CPU FPS distances (first 5): {cpu_distances[:5]}")
    print(f"GPU FPS distances (first 5): {gpu_distances[:5]}")
    
    # Both should be generally decreasing (FPS property)
    cpu_decreasing = sum(1 for i in range(1, len(cpu_distances)) if cpu_distances[i] <= cpu_distances[i-1])
    gpu_decreasing = sum(1 for i in range(1, len(gpu_distances)) if gpu_distances[i] <= gpu_distances[i-1])
    
    print(f"CPU decreasing pairs: {cpu_decreasing}/{len(cpu_distances)-1}")
    print(f"GPU decreasing pairs: {gpu_decreasing}/{len(gpu_distances)-1}")
    
    print("\n✅ All tests passed!")

def benchmark_scalability():
    """Benchmark CPU vs GPU performance across different problem sizes"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmarks")
        return
        
    print("\n=== Performance Benchmark ===")
    
    test_cases = [
        (24000, 3, 512),
        (24000, 3, 1024),
        (24000, 3, 2048),

    ]
    
    print(f"{'N':>6} {'D':>3} {'k':>4} {'CPU(s)':>8} {'GPU(s)':>8} {'Speedup':>8}")
    print("-" * 45)
    
    for N, D, k in test_cases:
        # Generate test data
        points_cpu = torch.randn(4096, N, D, dtype=torch.float32)
        points_gpu = points_cpu.cuda()
        
        # Benchmark CPU
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(3):  # Average over 3 runs
            _, _ = torch_fpsample.sample(points_cpu, k)
        cpu_time = (time.time() - start_time) / 3
        
        # Benchmark GPU
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(3):  # Average over 3 runs
            _, _ = torch_fpsample.sample(points_gpu, k)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / 3
        
        speedup = cpu_time / gpu_time
        print(f"{N:>6} {D:>3} {k:>4} {cpu_time:>8.4f} {gpu_time:>8.4f} {speedup:>8.2f}x")

if __name__ == "__main__":
    test_cuda_fps()
    benchmark_scalability()
