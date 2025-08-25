#include <torch/library.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>

#include "../utils.h"

using torch::Tensor;

// CUDA kernel declarations
template<typename T>
__global__ void fps_cuda_kernel(
    const T* __restrict__ points,    // [N, D] points for this batch
    int64_t* __restrict__ indices,   // [k] output indices
    T* __restrict__ distances,       // [N] working distances
    const int N,                     // number of points
    const int D,                     // dimension
    const int k,                     // number samples to select
    const int start_idx,             // starting point index
    const int grid_size              // spatial grid size for bucketing
);

template<typename T>
__global__ void init_distances_kernel(
    const T* __restrict__ points,    // [N, D] points
    T* __restrict__ distances,       // [N] distances to initialize
    const int N,                     // number of points  
    const int D,                     // dimension
    const int start_idx              // index of starting point
);

template<typename T>
__global__ void update_distances_kernel(
    const T* __restrict__ points,    // [N, D] points
    T* __restrict__ distances,       // [N] current min distances
    const int N,                     // number of points
    const int D,                     // dimension
    const int selected_idx           // newly selected point index
);

template<typename T>
__global__ void find_max_distance_kernel(
    const T* __restrict__ distances, // [N] current distances
    int64_t* __restrict__ max_idx,    // [1] output max index
    T* __restrict__ max_dist,        // [1] output max distance
    const int N                      // number of points
);

// Helper function to compute squared distance
template<typename T>
__device__ __forceinline__ T squared_distance(
    const T* __restrict__ a, 
    const T* __restrict__ b, 
    const int D
) {
    T dist = 0;
    #pragma unroll
    for (int i = 0; i < 16 && i < D; i++) {
        if (i < D) {
            T diff = a[i] - b[i];
            dist += diff * diff;
        }
    }
    // Handle remaining dimensions if D > 16
    for (int i = 16; i < D; i++) {
        T diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// Optimized block-level reduction for finding maximum
template<typename T>
__device__ void block_reduce_max_with_index(
    T val, int idx, T* __restrict__ shared_vals, int* __restrict__ shared_indices
) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Store values in shared memory
    shared_vals[tid] = val;
    shared_indices[tid] = idx;
    __syncthreads();
    
    // Reduction loop
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_indices[tid] = shared_indices[tid + s];
            }
        }
        __syncthreads();
    }
}

// Initialize distances from start point
template<typename T>
__global__ void init_distances_kernel(
    const T* __restrict__ points,
    T* __restrict__ distances,
    const int N,
    const int D,
    const int start_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const T* start_point = points + start_idx * D;
    const T* current_point = points + idx * D;
    
    distances[idx] = squared_distance(start_point, current_point, D);
}

// Update distances after selecting a new point
template<typename T>
__global__ void update_distances_kernel(
    const T* __restrict__ points,
    T* __restrict__ distances,
    const int N,
    const int D,
    const int selected_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const T* selected_point = points + selected_idx * D;
    const T* current_point = points + idx * D;
    
    T new_dist = squared_distance(selected_point, current_point, D);
    distances[idx] = fminf(distances[idx], new_dist);
}

// Find point with maximum distance using block-level reduction
template<typename T>
__global__ void find_max_distance_kernel(
    const T* __restrict__ distances,
    int64_t* __restrict__ max_idx,
    T* __restrict__ max_dist,
    const int N
) {
    extern __shared__ char shared_memory[];
    T* shared_vals = (T*)shared_memory;
    int* shared_indices = (int*)(shared_vals + blockDim.x);
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local max
    T local_max = (idx < N) ? distances[idx] : T(-1e30);
    int local_idx = (idx < N) ? idx : -1;
    
    // Handle multiple elements per thread if N > blockDim.x
    for (int i = idx + blockDim.x * gridDim.x; i < N; i += blockDim.x * gridDim.x) {
        if (distances[i] > local_max) {
            local_max = distances[i];
            local_idx = i;
        }
    }
    
    // Block-level reduction
    block_reduce_max_with_index(local_max, local_idx, shared_vals, shared_indices);
    
    // First thread of each block writes to global memory
    if (tid == 0) {
        max_dist[blockIdx.x] = shared_vals[0];
        max_idx[blockIdx.x] = shared_indices[0];
    }
}

// Main FPS kernel - processes one batch
template<typename T>
__global__ void fps_cuda_kernel(
    const T* __restrict__ points,
    int64_t* __restrict__ indices,
    T* __restrict__ distances,
    const int N,
    const int D,
    const int k,
    const int start_idx,
    const int grid_size
) {
    // This kernel assumes one block handles one complete FPS operation
    // For multiple batches, launch multiple blocks
    
    if (blockIdx.x != 0) return; // Only first block does the work for now
    
    // Set first sample
    if (threadIdx.x == 0) {
        indices[0] = start_idx;
    }
    __syncthreads();
    
    // Initialize distances from start point
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
        const T* start_point = points + start_idx * D;
        const T* current_point = points + idx * D;
        distances[idx] = squared_distance(start_point, current_point, D);
    }
    __syncthreads();
    
    // Iteratively select remaining k-1 points
    for (int sample = 1; sample < k; sample++) {
        __shared__ T shared_max_dist;
        __shared__ int shared_max_idx;
        
        // Find point with maximum distance
        T local_max = T(-1e30);
        int local_idx = -1;
        
        for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
            if (distances[idx] > local_max) {
                local_max = distances[idx];
                local_idx = idx;
            }
        }
        
        // Simple block reduction (could be optimized further)
        extern __shared__ char shared_memory[];
        T* shared_vals = (T*)shared_memory;
        int* shared_indices = (int*)(shared_vals + blockDim.x);
        
        block_reduce_max_with_index(local_max, local_idx, shared_vals, shared_indices);
        
        if (threadIdx.x == 0) {
            shared_max_dist = shared_vals[0];
            shared_max_idx = shared_indices[0];
            indices[sample] = shared_max_idx;
        }
        __syncthreads();
        
        // Update distances based on newly selected point
        for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
            const T* selected_point = points + shared_max_idx * D;
            const T* current_point = points + idx * D;
            T new_dist = squared_distance(selected_point, current_point, D);
            distances[idx] = fminf(distances[idx], new_dist);
        }
        __syncthreads();
    }
}

// Host function to launch CUDA kernels
std::tuple<Tensor, Tensor> sample_cuda(const Tensor &x, int64_t k,
                                       torch::optional<int64_t> h,
                                       torch::optional<int64_t> start_idx) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor, but found on ", x.device());
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be greater than or equal to 1, but got ", k);
    
    // Reshape input tensor
    auto [old_size, x_reshaped_raw] = bnorm_reshape(x);
    auto x_reshaped = x_reshaped_raw.to(torch::kFloat32).contiguous();
    
    int batch_size = x_reshaped.size(0);
    int N = x_reshaped.size(1);
    int D = x_reshaped.size(2);
    
    // Generate start indices
    torch::Tensor cur_start_idx;
    if (start_idx.has_value()) {
        cur_start_idx = torch::ones({batch_size}, 
                                   x_reshaped.options().dtype(torch::kInt64)) * start_idx.value();
    } else {
        cur_start_idx = torch::randint(0, N, {batch_size}, 
                                      x_reshaped.options().dtype(torch::kInt64));
    }
    
    // Allocate output tensors
    Tensor ret_indices = torch::empty({batch_size, k}, 
                                     x_reshaped.options().dtype(torch::kInt64));
    
    // Allocate working memory
    Tensor distances = torch::empty({batch_size, N}, x_reshaped.options());
    
    // CUDA kernel configuration
    const int block_size = 256;
    const int grid_size = h.value_or(8); // Use h parameter as spatial grid size
    
    // Launch kernels for each batch
    for (int b = 0; b < batch_size; b++) {
        const float* points_ptr = x_reshaped[b].data_ptr<float>();
        int64_t* indices_ptr = ret_indices[b].data_ptr<int64_t>();
        float* distances_ptr = distances[b].data_ptr<float>();
        int start_point = cur_start_idx[b].item<int64_t>();
        
        // Calculate shared memory size
        size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));
        
        // Launch FPS kernel
        fps_cuda_kernel<float><<<1, block_size, shared_mem_size>>>(
            points_ptr, indices_ptr, distances_ptr, 
            N, D, k, start_point, grid_size
        );
    }
    
    // Wait for kernels to complete
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    auto cuda_error = cudaGetLastError();
    TORCH_CHECK(cuda_error == cudaSuccess, 
                "CUDA kernel failed: ", cudaGetErrorString(cuda_error));
    
    // Gather sampled points
    Tensor ret_tensor = torch::gather(
        x_reshaped_raw, 1,
        ret_indices.view({ret_indices.size(0), ret_indices.size(1), 1})
            .repeat({1, 1, D}));
    
    // Reshape to original size
    auto ret_tensor_sizes = old_size.vec();
    ret_tensor_sizes[ret_tensor_sizes.size() - 2] = k;
    auto ret_indices_sizes = old_size.vec();
    ret_indices_sizes.pop_back();
    ret_indices_sizes[ret_indices_sizes.size() - 1] = k;
    
    return std::make_tuple(
        ret_tensor.view(ret_tensor_sizes),
        ret_indices.view(ret_indices_sizes).to(torch::kLong));
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(torch_fpsample, CUDA, m) { 
    m.impl("sample", &sample_cuda); 
}
