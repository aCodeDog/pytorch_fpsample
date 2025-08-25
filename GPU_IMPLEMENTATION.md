# GPU Implementation of Farthest Point Sampling (FPS)

## Overview

This document describes the CUDA implementation of the Farthest Point Sampling algorithm in `torch_fpsample`. The GPU implementation aims to accelerate the FPS algorithm while maintaining compatibility with the existing CPU version.

## Algorithm Comparison

### CPU Implementation (KDTree-based)
- **Data Structure**: Adaptive KDTree with recursive space partitioning
- **Bucketing**: Variable-sized buckets based on spatial subdivision
- **Complexity**: O(k * #buckets + N log N) where #buckets ≈ 2^h
- **Optimization**: Lazy distance updates with bounding box pruning

### GPU Implementation (Parallel FPS)
- **Data Structure**: Direct point array processing
- **Parallelism**: Block-level parallelization per batch
- **Complexity**: O(k * N) with high parallelism factor
- **Optimization**: Vectorized distance computation and parallel reductions

## GPU Implementation Details

### Memory Layout
```
Input points:  [B, N, D] -> Processed as B separate [N, D] arrays
Working distances: [N] per batch (minimum distance to selected set)
Output indices: [k] per batch (selected point indices)
```

### Kernel Architecture

#### 1. **Main FPS Kernel** (`fps_cuda_kernel`)
- **Grid Configuration**: 1 block per batch (for simplicity)
- **Block Size**: 256 threads
- **Shared Memory**: Used for parallel reductions

```cpp
template<typename T>
__global__ void fps_cuda_kernel(
    const T* points,           // [N, D] points for this batch
    int64_t* indices,          // [k] output indices  
    T* distances,              // [N] working distances
    int N, int D, int k,       // dimensions
    int start_idx,             // starting point
    int grid_size              // spatial grid parameter
)
```

#### 2. **Distance Computation**
- **Squared Euclidean Distance**: Avoids expensive sqrt operations
- **Loop Unrolling**: Manual unrolling for dimensions ≤ 16
- **Memory Coalescing**: Contiguous memory access patterns

```cpp
template<typename T>
__device__ __forceinline__ T squared_distance(
    const T* a, const T* b, int D
) {
    T dist = 0;
    #pragma unroll
    for (int i = 0; i < 16 && i < D; i++) {
        T diff = a[i] - b[i];
        dist += diff * diff;
    }
    // Handle remaining dimensions...
    return dist;
}
```

#### 3. **Parallel Reductions**
- **Block-Level Maximum**: Find point with maximum distance
- **Shared Memory**: Efficient intra-block communication
- **Warp-Level Optimizations**: Future enhancement opportunity

```cpp
template<typename T>
__device__ void block_reduce_max_with_index(
    T val, int idx, 
    T* shared_vals, int* shared_indices
)
```

### Algorithm Flow

1. **Initialization**
   - Select starting point (provided or random)
   - Compute initial distances from start point to all points

2. **Iterative Selection** (k-1 iterations)
   - **Find Maximum**: Parallel reduction to find point with max distance
   - **Update Distances**: Vectorized update of min distances to selected set
   - **Synchronization**: Block-wide synchronization between steps

3. **Output Generation**
   - Gather selected points using computed indices
   - Reshape tensors to match input dimensions

## Performance Characteristics

### Theoretical Speedup
- **CPU Complexity**: O(k * #buckets) ≈ O(k * 2^h)
- **GPU Complexity**: O(k * N / P) where P is parallelism factor
- **Expected Speedup**: 5-20x for typical problem sizes (N > 1000, k > 64)

### Memory Usage
- **Working Memory**: O(N) per batch for distance array
- **Shared Memory**: O(block_size) for reductions
- **Global Memory**: Coalesced access patterns for optimal bandwidth

### Limitations
- **Single Block per Batch**: Current implementation limits parallelism
- **No Spatial Optimization**: Unlike CPU KDTree, no spatial pruning
- **Fixed Block Size**: Not adaptive to problem size

## Usage

### Building with CUDA Support
```bash
# Enable CUDA compilation
export WITH_CUDA=1
pip install -e .
```

### API Compatibility
The CUDA implementation maintains full API compatibility:

```python
import torch
import torch_fpsample

# Works on both CPU and GPU tensors
points_gpu = torch.randn(batch_size, N, D).cuda()
sampled_points, indices = torch_fpsample.sample(points_gpu, k)
```

### Performance Testing
```bash
python test_cuda_fps.py
```

## Optimization Opportunities

### Short-term Improvements
1. **Multi-block Processing**: Launch multiple blocks per batch
2. **Warp-level Reductions**: Use `__shfl_*` intrinsics
3. **Memory Optimization**: Reduce global memory accesses
4. **Dynamic Block Size**: Adapt to problem size

### Long-term Enhancements
1. **Hierarchical FPS**: GPU-friendly spatial partitioning
2. **Approximate FPS**: Trade accuracy for speed using sampling
3. **Multi-GPU Support**: Scale across multiple devices
4. **Mixed Precision**: Use half-precision for better throughput

## Implementation Comparison

| Aspect | CPU (KDTree) | GPU (Parallel) |
|--------|--------------|----------------|
| **Data Structure** | Adaptive tree | Flat arrays |
| **Memory Pattern** | Tree traversal | Coalesced access |
| **Parallelism** | Thread-level | Massive parallel |
| **Complexity** | O(k * #buckets) | O(k * N / P) |
| **Best Use Case** | Small k, large N | Large k, massive N |
| **Memory Overhead** | Tree storage | Working arrays |

## Testing and Validation

### Correctness Tests
- **Index Validity**: All indices in range [0, N)
- **Point Consistency**: Sampled points match indices
- **FPS Property**: Distances generally decreasing (stochastic)

### Performance Tests
- **Scalability**: Test across various (N, D, k) combinations
- **Memory Usage**: Monitor GPU memory consumption
- **Speedup Measurement**: Compare against CPU baseline

## Future Work

1. **Adaptive Algorithm Selection**: Choose CPU vs GPU based on problem size
2. **Batched Operations**: Optimize for large batch processing
3. **Integration with Point Cloud Libraries**: PCL, Open3D compatibility
4. **Benchmark Suite**: Comprehensive performance evaluation

## References

- Original FPS Paper: "Farthest Point Sampling" algorithms
- CUDA Programming Guide: Memory optimization patterns
- PyTorch Extension Documentation: Custom CUDA operators
