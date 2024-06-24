/*
Global norm, used in gradient clipping
*/
#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// CUDA kernels

template<class T>
float global_norm_squared_for_range(sycl::nd_item<2> id, const T* data, size_t count) {
    size_t index = blockIdx_x(id) * blockDim_x(id) + threadIdx_x(id);
    size_t grid_width = blockDim_x(id) * gridDim_x(id);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // block-level reduce
    return sycl::reduce_over_group(id.get_group(), accumulator, sycl::plus<float>());
}

template<class T>
void global_norm_squared_kernel(sycl::nd_item<2> id, float* out, const T* data, size_t count, ptrdiff_t stride) {
    float block_sum = global_norm_squared_for_range(id, data + blockIdx_y(id) * stride, count);
    // each block accumulates its partial sum to out[out_index]
    // we want to avoid using atomic add here so we combine this kernel with another kernel call
    // that sums up the partial block sums
    if(threadIdx_x(id) == 0) {
        size_t out_index = blockIdx_y(id) * gridDim_x(id) + blockIdx_x(id);
        out[out_index] = out[out_index] + block_sum;
    }
}

void global_norm_aggregate_kernel(sycl::nd_item<1> id, float* out, size_t grid_size) {
    size_t index = threadIdx_x(id);
    // grab block sums from the previous kernel, use 0. as the neutral sum element
    float block_sum = (index < grid_size) ? out[index] : 0.f;
    float sum = sycl::reduce_over_group(id.get_group(), block_sum, sycl::plus<float>());
    if(threadIdx_x(id) == 0) {
        out[0] = sum;  // out[0] ends up with the final norm squared
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// Helper function determines the maximum number of block sums
int get_max_num_block_sums(int* num_slices_all, int numel) {
    // NOTE: this needs to be kept in sync with `global_norm_squared` below.
    const int block_size = 512;
    // const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    int grid_size = main_stream->get_device().get_info<sycl::info::device::max_compute_units>();
    grid_size *= WARP_SIZE;
    grid_size = CEIL_DIV(grid_size, block_size);
    assert(grid_size > 0);
    int max_num_block_sums = 0;
    for (int i = 0; i < numel; i++) {
        int num_slices = num_slices_all[i];
        const int gx = CEIL_DIV(grid_size, num_slices);
        const int gy = num_slices;
        max_num_block_sums = sycl::max(max_num_block_sums, gx * gy);
    }

    return max_num_block_sums;
}

template<typename T>
void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, sycl::queue* stream) {
    const int block_size = 512;
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    //const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    int size = stream->get_device().get_info<sycl::info::device::max_compute_units>();
    size *= WARP_SIZE;
    const int grid_size = CEIL_DIV(size, block_size);
    assert(grid_size > 0);      // gives a better error than letting the call below fail

    const int gx = CEIL_DIV(grid_size, num_slices);
    const int gy = num_slices;

    assert(gx * gy < 1024);  // we want to later accumulate the block sums in a single block

    if (reset) {
        stream->memset(out, 0, max_num_block_sums * sizeof(float));
    }
    sycl::range<2> grid_dim(gy, gx);
    sycl::range<2> block_dim(1, block_size);
    stream->parallel_for(sycl::nd_range<2>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<2> id) {
        global_norm_squared_kernel(id, out, values, count, stride);
    });
}

void global_norm_squared_aggregate(float* out, int max_num_block_sums, sycl::queue* stream) {
    // Yeah, the SMs on Intel GPUs just aren't as big.
    // assert(max_num_block_sums > 0 && max_num_block_sums < 1024);  // we need to accumulate the block sums in a single block
    // important to use 1024 here for determinism, otherwise blockreduce might introduce errors
    // well, we can't do more than 512 on Intel GPUs
    stream->parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> id) {
        global_norm_aggregate_kernel(id, out, max_num_block_sums);
    });
}
