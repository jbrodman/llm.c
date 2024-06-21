/*
Kernels for matmul backward pass bias only.

Compile example:
nvcc -O3 matmul_backward_bias.cu -lineinfo -o matmul_backward_bias
This requires a github compiler until at least 2024.2 comes out for the non-uniform group algs.

./matmul_backward_bias 1
./matmul_backward_bias 2
./matmul_backward_bias 3
./matmul_backward_bias 4
./matmul_backward_bias 5
ncu:
sudo ncu --set full --import-source yes -o bias -f ./matmul_backward_bias 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sycl/sycl.hpp>

#define ENABLE_BF16
#include "common.h"


// ----------------------------------------------------------------------------
// utility functions
bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

int largestPowerOfTwoLessOrEqual(int n) {
    // Return the largest power of 2 less than or equal to n
    if (n < 1) {
        return 0;
    }

    while ((n & (n - 1)) > 0) {
        n = n & (n - 1);
    }

    return n;
}

float shfl_down(sycl::sub_group warp, float val, int delta, int width) {
    int id = warp.get_local_id();
    int subid = id % width;
    float result = sycl::shift_group_left(warp, val, delta);
    if (subid + delta >= width)
        result = val;
    return result;
}

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_bias_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    for (int o = 0; o < OC; o++) {
        // Change to float for Gen12LP/HPG
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                sum += dout_bt[o];
            }
        }
        dbias[o] = sum;
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

float* dbias_buffer;

void matmul_backward_bias_kernel1(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC) {
    int o = blockIdx_x(id); // range [0, OC)
    int tid = threadIdx_x(id); // range [0, block_size)
    int block_size = blockDim_x(id);
    const floatX* x = dout + o;
    // thread coarsening
    float sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += (float)x[i * OC];
    }
    sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<float>());
    
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = (floatX)((float)dbias[o] + sum);
    }
}

// cooperative groups solution, one warp per output channel
void matmul_backward_bias_kernel2(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC) {
    // dout is (B, T, OC), dbias is (OC)
    // e.g. if block_size = 128, then we have 4 warps per block, each in charge of one output channel
    sycl::sub_group warp = id.get_sub_group();
    // meta_group_size is the number of warps in a block (e.g. 4), meta_group_rank is the warp index (0,1,2,3)
    int idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);
    if(idx >= OC) { return; }
    int BT = B * T; // number of elements to reduce in total, per channel
    // first, thread coarsening to sum reduce the problem size from B*T to 32
    float sum = 0.0f;
    for(int i = thread_rank(warp); i < BT; i += size(warp)) {
        sum += (float)dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in this warp
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>());
    // write the result to output (global memory)
    if(warp.leader()) {
        dbias[idx] += (floatX)sum;
    }
}

void matmul_backward_bias_kernel3(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC) {
    // dout is (B, T, OC), dbias is (OC)
    // in this version of the kernel the entire block of block_size is dedicated to one output channel
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int BT = B * T; // number of elements to reduce in total, per channel
    /*
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    */ 
    // Let's try to be sub-group size agnostic.
    int idx = blockIdx_x(id); // simply one block per row
    // round 1: thread coarsening to reduce the problem size from B*T to 32
    float thread_sum = 0.0f;
    for(int i = threadIdx_x(id); i < BT; i += blockDim_x(id)) {
        thread_sum += (float)dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in each warp
    float block_sum = sycl::reduce_over_group(block, thread_sum, sycl::plus<float>());
    // write the result to output (global memory)
    if(block.leader()) {
        dbias[idx] += block_sum;
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
void matmul_backward_bias_kernel4(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC, 
                                  sycl::local_accessor<float> lmem) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    float *smem = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // of size block_size (128)
    sycl::sub_group warp = id.get_sub_group();

    const int warpSize = size(warp); 
    const int warp_id = warp.get_group_linear_id(); // warp index in the block, 0,1,2,3
    const int lane_id = warp.get_local_linear_id(); // thread index in the warp, 0,1,2,...,31
    const int tl = blockIdx_x(id) * warpSize; // pointer to the start column for this block
    const int vstep = warp.get_group_linear_range(); // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const floatX* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += (float)dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    sycl::group_barrier(id.get_group());

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}

#ifndef ENABLE_BF16
void matmul_backward_bias_kernel5(sycl::nd_item<2> id, floatX* dbias, const floatX* dout, int B, int T, int OC) {
    int oc = blockIdx_x(id) * blockDim_x(id) + threadIdx_x(id);
    if(oc >= OC) return;
    float sum = 0.0;
    // grid-wide loop for maximum parallelism
    for (int i = blockIdx_y(id); i < B * T; i += gridDim_y(id)) {
        sum += (float)dout[i * OC + oc];
    }
    // and atomically add everything together. atomics within one block are conflict-free!
    atomicAdd(dbias + oc, sum);
}
#endif


void cast_and_add_kernel(sycl::nd_item<1> id, floatX* dst, const float* src, size_t n) {
    // used only for matmul_backward_bias kernel, a little bit embarassing TODO delete later
    const size_t idx = id.get_global_id(0);
    if (idx < n) { dst[idx] = (floatX)((float)dst[idx] + src[idx]); } // have to += because dbias is a paramater
}

void matmul_backward_bias_kernel7(sycl::nd_item<2> id, float* dbias, const floatX* dout, int B, int T, int OC, const int block_size,
                                  sycl::local_accessor<float> lmem) {
    // note: this kernel reads in floatX, but it writes to float!
    // this is because we're using atomics, which are super slow in < fp32 precision on < H100 GPUs
    // so the trick is do fp32 atomics to a buffer, and then copy_and_cast the result to floatX
    // (this also results in higher accuracy than doing accumulation directly in floatX)

    // see comments in matmul_backward() for an explanation of block/grid dimensions etc.
    const int block_size_x = 32;
    const int block_size_y = block_size / block_size_x; // 16
    const int OC_per_warp = block_size_x * x128::size;  // 256 at BF16

    int local_oc = threadIdx_x(id) * x128::size;
    int global_oc = blockIdx_x(id) * OC_per_warp + local_oc;
    float accumulators[x128::size];
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();;

    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }
    int thread_id = threadIdx_y(id) * block_size_x + threadIdx_x(id);
    for (int idx = thread_id; idx < OC_per_warp; idx += block_size) {
        shared[idx] = 0.0f;
    }
    sycl::group_barrier(id.get_group());
    if(global_oc < OC) {
        for (int idx = blockIdx_y(id)*block_size_y + threadIdx_y(id); idx < B * T; idx += gridDim_y(id)*block_size_y) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
        // we need to avoid shared memory bank conflicts for the atomicAdd to maximise performance,
        // so we accumulate in a conflict-free order, then reorder to match the global memory order
        for (int k = 0; k < x128::size; k++) {
            atomicAdd(shared + threadIdx_x(id) + (k * block_size_x), accumulators[k]);
        }
    }
    // If this happens, XE1 GPUs will croak.
    if (threadIdx_y(id) >= x128::size) { return; } // only need this many warps to reorder the data
    sycl::group_barrier(id.get_group());
    // read the accumulated values in the conflict-free order
    int i = threadIdx_x(id) + (threadIdx_y(id) * block_size_x);
    float tmp = shared[i];
    sycl::group_barrier(id.get_group());
    // write them back to shared memory in the global memory order
    // 8-way bank conflict for BF16 x128, but only 8x per threadblock (rather than 8x per warp)
    shared[local_oc + threadIdx_y(id)] = tmp;
    sycl::group_barrier(id.get_group());
    // now we do a perfectly coalesced atomic add to global memory (1x 128-byte cacheline per warp)
    if (i + blockIdx_x(id)*OC_per_warp < OC) {
        atomicAdd(dbias + i + blockIdx_x(id)*OC_per_warp, shared[i]);
    }
}

// We want to decrease the amount of channels handled by each block, so that we need fewer across-block reductions.
// We do this by realizing the following: For scalar memory access, we need to read one element per thread in a warp
// to read an entire cacheline, but for vectorized memory access, with 128 bit of data per thread, we only need eight
// threads to fetch a cacheline, which means that we can already operate on a "depth" of four within a single warp.
// => blockDim.x == 4, blockDim.y == 32/4 = 8
//
template<typename OutFloat, bool Atomic>
void matmul_backward_bias_kernel8(sycl::nd_item<3> id, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<Atomic>, sycl::local_accessor<float> lmem) {
    sycl::sub_group warp = id.get_sub_group();
    constexpr const int bdx = 4;
    constexpr const int bdy = 32 / bdx;


    int warp_d = (int)threadIdx_x(id);
    int warp_c = (int)threadIdx_y(id);
    int block_d = (int)threadIdx_z(id);

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx_x(id) * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim_z(id);

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx_y(id) * bt_per_block + local_bt; idx < B * T; idx += gridDim_y(id) * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    float (*sub_results)[32][bdy] = (float (*)[32][bdy])shared;
    auto Partition = syclex::get_fixed_size_group<4>(warp);

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        // v += __shfl_down_sync(0xffffffff, v, 1, 4);
        // v += __shfl_down_sync(0xffffffff, v, 2, 4);
        // Ask John why this didn't work. Thread to warp mapping difference?
        //v += sycl::shift_group_left(Partition, v, 1);
        //v += sycl::shift_group_left(Partition, v, 2);
        v = sycl::reduce_over_group(Partition, v, sycl::plus<float>());
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    sycl::group_barrier(id.get_group());

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim_z(id)) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim_z(id); r += bdx) {
            float v = sub_results[k][r][warp_c];
            //v += __shfl_down_sync(0xffffffff, v, 1, 4);
            //v += __shfl_down_sync(0xffffffff, v, 2, 4);
            //v += sycl::shift_group_left(Partition, v, 1);
            //v += sycl::shift_group_left(Partition, v, 2);
            v = sycl::reduce_over_group(Partition, v, sycl::plus<float>());
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            // coalesced, but not cacheline-sized
            if constexpr (!Atomic) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                atomicAdd(dbias + global_oc + k, a);
            }
        }
    }
}

// Like kernel 8, but instead of accumulating to the auxiliary buffer, it writes
// multiple values that need to be summed up in a separate kernel call.
// If UseAuxBuffer is false, gridDim.y has to be one, and results are added directly
// to dbias.
template<typename OutFloat, bool UseAuxBuffer>
void matmul_backward_bias_kernel9(sycl::nd_item<3> id, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>, sycl::local_accessor<float> lmem) {
    sycl::sub_group warp = id.get_sub_group();
    constexpr const int bdx = 4;
    constexpr const int bdy = 32 / bdx;
   
    int warp_d = (int)threadIdx_x(id);
    int warp_c = (int)threadIdx_y(id);
    int block_d = (int)threadIdx_z(id);

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx_x(id) * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim_z(id);

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx_y(id) * bt_per_block + local_bt; idx < B * T; idx += gridDim_y(id) * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    float (*sub_results)[32][bdy] = (float (*)[32][bdy])shared;
    auto Partition = syclex::get_fixed_size_group<4>(warp);

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        //v += __shfl_down_sync(0xffffffff, v, 1, 4);
        //v += __shfl_down_sync(0xffffffff, v, 2, 4);
        //v += sycl::shift_group_left(Partition, v, 1);
        //v += sycl::shift_group_left(Partition, v, 2);
        //v = sycl::reduce_over_group(Partition, v, sycl::plus<float>());
        v += shfl_down(warp, v, 1, 4);
        v += shfl_down(warp, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    sycl::group_barrier(id.get_group());

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim_z(id)) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim_z(id); r += bdx) {
            float v = sub_results[k][r][warp_c];
            //v += __shfl_down_sync(0xffffffff, v, 1, 4);
            //v += __shfl_down_sync(0xffffffff, v, 2, 4);
            //v += sycl::shift_group_left(Partition, v, 1);
            //v += sycl::shift_group_left(Partition, v, 2);
            //v = sycl::reduce_over_group(Partition, v, sycl::plus<float>());
            v += shfl_down(warp, v, 1, 4);
            v += shfl_down(warp, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            // coalesced, but not cacheline-sized
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx_y(id) * OC] = a;
            }
        }
    }
}


void reduce_add_sum_kernel(sycl::nd_item<1> id, floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (id.get_global_id(0)) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}
// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward_bias1(floatX* dbias, const floatX* dout,
                      int B, int T, int OC, int block_size) {
    
    block_size = largestPowerOfTwoLessOrEqual(block_size);
    assert(isPowerOfTwo(block_size)); // block_size needs to be power of 2 due to the reduction
    const int block_dim = block_size;
    const int grid_dim = OC;
    size_t shared_mem_size = block_size * sizeof(float);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel1(id, dbias, dout, B, T, OC);
    }); 
}

void matmul_backward_bias2(floatX* dbias, const floatX* dout,
                      int B, int T, int OC, int block_size) {
    // block_size 512 seems best
    const int grid_size = ceil_div(OC * 32, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel2(id, dbias, dout, B, T, OC);
    }); 
}

void matmul_backward_bias3(floatX* dbias, const floatX* dout,
                      int B, int T, int OC, int block_size) {
    // block_size 256 seems best
    DefaultQueue->parallel_for(sycl::nd_range<1>(OC * block_size, block_size), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel3(id, dbias, dout, B, T, OC);
    });
}

void matmul_backward_bias4(floatX* dbias, const floatX* dout,
                      int B, int T, int OC, int block_size) {
    assert(OC % 32 == 0); // OC must be divisible by 32 for this kernel
    const int grid_size = OC / 32;
    DefaultQueue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> lmem(block_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            matmul_backward_bias_kernel4(id, dbias, dout, B, T, OC, lmem);
        });
    });
}

#ifndef ENABLE_BF16
void matmul_backward_bias5(floatX* dbias, const floatX* dout,
                      int B, int T, int OC, int block_size) {
    const int grid_size_x = ceil_div(OC, block_size);
    constexpr int warpSize = 32;
    int size = DefaultQueue->get_device().get_info<sycl::info::device::max_compute_units>();
    size *= warpSize;
    const int grid_size_y = std::max(1, size / block_size);
    sycl::range<2> block_dim(1, block_size);
    sycl::range<2> grid_dim(grid_size_y, grid_size_x);

    DefaultQueue->parallel_for(sycl::nd_range<2>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]] {
        matmul_backward_bias_kernel5(id, dbias, dout, B, T, OC);
    });
}
#endif

void matmul_backward_bias7(floatX* dbias, const floatX* dout,
                      int B, int T, int OC, int block_size) {
    if(block_size < 256) {
        block_size = 256;
    }
    if (block_size > 256) {
        // larger block sizes provokes the early exit problem on Xe1.
        block_size = 256;
    }
    // Each warp is responsible for 32 * "x128::size" = 256 OCs at BF16 (OC must be a multiple of 256!)
    // Block size is 512 threads (16 warps) and we reduce those 16 values into 1 at the end
    // blockDim.x is 32 --> single warp being responsible for those 256 OCs
    // blockDim.y is 16 --> 16 parallel independent warps processing the same OCs for different BTs
    // gridDim.x is OC / 256 --> each block processes 256 OCs
    // grimDim.y is max(1, (cuda_num_SMs * threads_per_SM) / (512 * gridDim.x)); --> fill up the entire GPU!
    const int warp_size = 32;
    const int OC_per_warp = warp_size * x128::size; // 256 at BF16
    const int block_size_x = 32;
    const int block_size_y = block_size / block_size_x; // 16
    const int grid_size_x = ceil_div(OC, OC_per_warp); // e.g. 3 horizontal blocks for 768 OCs at BF16
    int size = DefaultQueue->get_device().get_info<sycl::info::device::max_compute_units>();
    size *= warp_size;
    const int grid_size_y = std::max(1, ceil_div(size, (block_size * grid_size_x))); // full GPU!

    assert(block_size_y >= x128::size); // part of the kernel assumes this is large enough to avoid loops

    sycl::range<2> block_dim(block_size_y, block_size_x);
    sycl::range<2> grid_dim(grid_size_y, grid_size_x);
    float *dbias_buf = dbias_buffer;

    DefaultQueue->memset(dbias_buf, 0, OC * sizeof(float));
        DefaultQueue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> lmem(OC_per_warp * sizeof(float), h);
        h.parallel_for(sycl::nd_range<2>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]] {
            matmul_backward_bias_kernel7(id, dbias_buf, dout, B, T, OC, block_size, lmem);
        });
    });
    DefaultQueue->parallel_for(sycl::nd_range<1>(ceil_div(OC, 256)*256, 256), [=](sycl::nd_item<1> id) {
        cast_and_add_kernel(id, dbias, dbias_buf, OC);
    });
}

void matmul_backward_bias8(floatX* dbias, const floatX* dout,
                      int B, int T, int OC, int block_size) {
    sycl::range<3> block_dim((unsigned)block_size/32, 8, 4); 
    const int OC_per_warp = block_dim[1] * x128::size; // 64 at BF16
    const int grid_size_x = ceil_div(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    int size = DefaultQueue->get_device().get_info<sycl::info::device::max_compute_units>();
    int warpSize = 32;
    size *= warpSize;
    const int grid_size_y = std::max(1, size / (block_size * grid_size_x)); // full GPU!

    sycl::range<3> grid_dim(1, grid_size_y, grid_size_x);

    // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
    // and write results directly to the output.
    if(grid_size_y == 1) {
         DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel8(id, dbias, dout, B, T, OC, std::bool_constant<false>{}, lmem);
            });
        });
    } else {
        float *dbias_buf = dbias_buffer;
        DefaultQueue->memset(dbias_buf, 0, OC * sizeof(float));
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel8(id, dbias_buf, dout, B, T, OC, std::bool_constant<true>{}, lmem);
            });
        });
        DefaultQueue->parallel_for(sycl::nd_range<1>(ceil_div(OC, 256)*256, 256), [=](sycl::nd_item<1> id) {
            cast_and_add_kernel(id, dbias, dbias_buf, OC);
        });
    }
}


void matmul_backward_bias9(floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {
    sycl::range<3> block_dim((unsigned)block_size/32, 8, 4); 
    const int OC_per_warp = block_dim[1] * x128::size; // 64 at BF16
    const int grid_size_x = ceil_div(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    int size = DefaultQueue->get_device().get_info<sycl::info::device::max_compute_units>();
    int warpSize = 32;
    size *= warpSize;
    const int grid_size_y = std::max(1, size / (block_size * grid_size_x)); // full GPU!

    sycl::range<3> grid_dim(1, grid_size_y, grid_size_x);

    // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
    // and write results directly to the output.
    if(grid_size_y == 1) {
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel9(id, dbias, dout, B, T, OC, std::bool_constant<false>{}, lmem);
            });
        });
    } else {
        // kernel 9 overwrites temp buffer, so no need to memset
        float *dbias_buf = dbias_buffer;
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel9(id, dbias_buf, dout, B, T, OC, std::bool_constant<true>{}, lmem);
            });
        });
        DefaultQueue->parallel_for(sycl::nd_range<1>(ceil_div(OC, 256*f128::size)*256, 256), [=](sycl::nd_item<1> id) {
            reduce_add_sum_kernel(id, dbias, dbias_buf, OC, grid_size_y);
        });
    }
}


void matmul_backward_bias(int kernel_num, floatX* dbias, floatX* dout,
                     int B, int T, int OC, int block_size) {
    switch (kernel_num) {
        case 1:
            matmul_backward_bias1(dbias, dout, B, T, OC, block_size);
            break;
        case 2:
            matmul_backward_bias2(dbias, dout, B, T, OC, block_size);
            break;
        case 3:
            matmul_backward_bias3(dbias, dout,  B, T, OC, block_size);
            break;
        case 4:
            matmul_backward_bias4(dbias, dout, B, T, OC, block_size);
            break;
        case 5:
#ifndef ENABLE_BF16
            matmul_backward_bias5(dbias, dout, B, T, OC, block_size);
#else
            fprintf(stderr, "Kernel 5 is only supported for fp32");
            exit(1);
#endif
            break;
        case 7:
            matmul_backward_bias7(dbias, dout, B, T, OC, block_size);
            break;
        case 8:
            matmul_backward_bias8(dbias, dout, B, T, OC, block_size);
            break;
        case 9:
            matmul_backward_bias9(dbias, dout, B, T, OC, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }
    DefaultQueue = &defaultQueue;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // create host memory of random numbers
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);

    // move to GPU
    floatX* d_dbias;
    floatX* d_dout;
    syclMallocCheck(d_dbias = sycl::malloc_device<floatX>(OC, defaultQueue));
    syclMallocCheck(d_dout = sycl::malloc_device<floatX>(B * T * OC, defaultQueue));
    syclMallocCheck(dbias_buffer = sycl::malloc_device<float>(OC*32, defaultQueue));
    memcpy_convert(d_dbias, dbias, OC);
    memcpy_convert(d_dout, dout, B * T * OC);
    defaultQueue.wait();
    

    // ncu debugging / profiling, do a single call
    // int block_size_debug;
    // if (kernel_num == 1) { block_size_debug = 512;
    // } else if (kernel_num == 2) { block_size_debug = 512;
    // } else { block_size_debug = 256; }
    // printf("kernel %d, block_size %d\n", kernel_num, block_size_debug);
    // matmul_backward_bias(kernel_num, NULL, NULL, d_dbias, d_dout, NULL, NULL, NULL, B, T, C, OC, block_size_debug);
    // exit(EXIT_SUCCESS);

    // no 1024 for Intel GPUs
    int block_sizes[] = {32, 64, 128, 256};

    // calculate the CPU reference
    matmul_backward_bias_cpu(NULL, NULL, dbias, dout, NULL, NULL, B, T, C, OC);

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        // memset the bias to zero
        defaultQueue.memset(d_dbias, 0, OC * sizeof(floatX));
        // calculate the GPU version
        matmul_backward_bias(kernel_num, d_dbias, d_dout, B, T, OC, block_size);
        // compare
        printf("Checking correctness...\n");
        float tol = std::is_same_v<floatX, float> ? 5e-3f : 1.0f;
        validate_result(d_dbias, dbias, "dbias", OC, tol);
        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now benchmark the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_backward_bias, kernel_num,
                                            d_dbias, d_dout, B, T, OC, block_size);
        printf("block_size %d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(dbias);
    free(dout);
    sycl::free(dbias_buffer, defaultQueue);
    sycl::free(d_dbias, defaultQueue);
    sycl::free(d_dout, defaultQueue);

    return 0;
}
