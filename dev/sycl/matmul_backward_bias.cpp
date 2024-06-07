/*
Kernels for matmul backward pass bias only.

Compile example:
nvcc -O3 matmul_backward_bias.cu -lineinfo -o matmul_backward_bias

./matmul_backward_bias 1
./matmul_backward_bias 2
./matmul_backward_bias 3
./matmul_backward_bias 4

ncu:
sudo ncu --set full --import-source yes -o bias -f ./matmul_backward_bias 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <omp.h>
#include "common.h"

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

void matmul_backward_bias_kernel1(sycl::nd_item<1> id, float* dbias, const float* dout, int B, int T, int OC) {
    int o = id.get_group(0); // range [0, OC)
    int tid = id.get_local_linear_id(); // range [0, block_size)
    int block_size = id.get_local_range(0);
    const float* x = dout + o;
    // thread coarsening
    float sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<float>());
    
    // write the final result (at thread 0) to global memory
    if (id.get_group().leader()) {
        dbias[o] += sum;
    }
}

// cooperative groups solution, one warp per output channel
void matmul_backward_bias_kernel2(sycl::nd_item<1> id, float* dbias, const float* dout, int B, int T, int OC) {
    // dout is (B, T, OC), dbias is (OC)
    // e.g. if block_size = 128, then we have 4 warps per block, each in charge of one output channel
    sycl::sub_group warp = id.get_sub_group();
    // meta_group_size is the number of warps in a block (e.g. 4), meta_group_rank is the warp index (0,1,2,3)
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= OC) { return; }
    int BT = B * T; // number of elements to reduce in total, per channel
    // first, thread coarsening to sum reduce the problem size from B*T to 32
    float sum = 0.0f;
    for(int i = warp.get_local_linear_id(); i < BT; i += warp.get_max_local_range()[0]) {
        sum += dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in this warp
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>());
    // write the result to output (global memory)
    if(warp.leader()) {
        dbias[idx] += sum;
    }
}

void matmul_backward_bias_kernel3(sycl::nd_item<1> id, float* dbias, const float* dout, int B, int T, int OC) {
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
    int num_warps = id.get_local_range(0) / warp.get_max_local_range()[0];
    int warp_id = id.get_local_id(0) / warp.get_max_local_range()[0];
    int lane_id = id.get_local_linear_id();
    int idx = id.get_group(0); // simply one block per row
    // round 1: thread coarsening to reduce the problem size from B*T to 32
    float thread_sum = 0.0f;
    for(int i = id.get_local_linear_id(); i < BT; i += id.get_local_range(0)) {
        thread_sum += dout[i * OC + idx];
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
void matmul_backward_bias_kernel4(sycl::nd_item<1> id, float* dbias, const float* dout, int B, int T, int OC, 
                                  sycl::local_accessor<float> lmem) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    float *smem = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // of size block_size (128)
    sycl::sub_group warp = id.get_sub_group();

    const int warpSize = warp.get_max_local_range()[0]; 
    const int warp_id = warp.get_group_linear_id(); // warp index in the block, 0,1,2,3
    const int lane_id = warp.get_local_linear_id(); // thread index in the warp, 0,1,2,...,31
    const int tl = id.get_group(0) * warpSize; // pointer to the start column for this block
    const int vstep = warp.get_group_linear_range(); // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const float* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += dout_col[row * OC];
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

// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward_bias1(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    const int block_dim = block_size;
    const int grid_dim = OC;
    size_t shared_mem_size = block_size * sizeof(float);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel1(id, dbias, dout, B, T, OC);
    }); 
}

void matmul_backward_bias2(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    // block_size 512 seems best
    const int grid_size = ceil_div(OC * 32, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel2(id, dbias, dout, B, T, OC);
    }); 
}

void matmul_backward_bias3(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    // block_size 256 seems best
    DefaultQueue->parallel_for(sycl::nd_range<1>(OC * block_size, block_size), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel3(id, dbias, dout, B, T, OC);
    });
}

void matmul_backward_bias4(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    assert(OC % 32 == 0); // OC must be divisible by 32 for this kernel
    const int grid_size = OC / 32;
    DefaultQueue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> lmem(block_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            matmul_backward_bias_kernel4(id, dbias, dout, B, T, OC, lmem);
        });
    });
}

void matmul_backward_bias(int kernel_num,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC, int block_size) {
    switch (kernel_num) {
        case 1:
            matmul_backward_bias1(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
            break;
        case 2:
            matmul_backward_bias2(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
            break;
        case 3:
            matmul_backward_bias3(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
            break;
        case 4:
            matmul_backward_bias4(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
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
    float* d_dbias;
    float* d_dout;
    syclMallocCheck(d_dbias = sycl::malloc_device<float>(OC, defaultQueue));
    syclMallocCheck(d_dout = sycl::malloc_device<float>(B * T * OC, defaultQueue));
    defaultQueue.memcpy(d_dbias, dbias, OC * sizeof(float));
    defaultQueue.memcpy(d_dout, dout, B * T * OC * sizeof(float));
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
    int block_sizes[] = {32, 64, 128, 256, 512};

    // calculate the CPU reference
    matmul_backward_bias_cpu(NULL, NULL, dbias, dout, NULL, NULL, B, T, C, OC);

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        // memset the bias to zero
        defaultQueue.memset(d_dbias, 0, OC * sizeof(float));
        // calculate the GPU version
        matmul_backward_bias(kernel_num, NULL, NULL, d_dbias, d_dout, NULL, NULL, NULL, B, T, C, OC, 128);
        // compare
        printf("Checking correctness...\n");
        validate_result(d_dbias, dbias, "dbias", OC, 5e-3f);
        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now benchmark the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        float *d_dinp, *d_dweight, *d_inp, *d_weight, *d_ones;
        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_backward_bias, kernel_num,
                                            d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                            B, T, C, OC, block_size);
        printf("block_size %d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(dbias);
    free(dout);
    sycl::free(d_dbias, defaultQueue);
    sycl::free(d_dout, defaultQueue);

    return 0;
}