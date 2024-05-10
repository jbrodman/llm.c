/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math encoder_backward.cu -o encoder_backward

version 1 is naive port from CPU code to kernel
parallelizes over B,T,C, uses atomics to add to dwte, dwpe
./encoder_backward 1

version 2 is another naive port
parallelizes over C, loops over B,T; much slower than version 1
./encoder_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
void encoder_backward_cpu(float* dwte, float* dwpe,
                            float* dout, int* inp,
                            int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation with atomics
void encoder_backward_kernel1(sycl::nd_item<1> id, float* dwte, float* dwpe,
                             const float* dout, const int* inp,
                             int B, int T, int C) {
    int idx = id.get_global_id(0);
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dwte_ix_atomic(*dwte_ix);
        dwte_ix_atomic += *dout_btc;    
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dwpe_tc_atomic(*dwpe_tc);
        dwpe_tc_atomic += *dout_btc;
    }
}

// naive implementation that parallelizes over C and loops over B,T
// but it gets rid of atomics
void encoder_backward_kernel2(sycl::nd_item<1> id, float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              int B, int T, int C) {
    int c = id.get_global_id(0);
    if (c >= C) { return; } // guard
    int BT = B * T;
    for (int i = 0; i < BT; i++) {
        int t = i % T;
        int ix = inp[i];
        float dout_btc = dout[i * C + c];
        dwte[ix * C + c] += dout_btc;
        dwpe[t * C + c] += dout_btc;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_backward1(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_backward_kernel1(id, dwte, dwpe, dout, inp, B, T, C);
    });
}

void encoder_backward2(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int grid_size = ceil_div(C, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_backward_kernel2(id, dwte, dwpe, dout, inp, B, T, C);
    });
}

// kernel version dispatch
void encoder_backward(int kernel_num,
                     float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_backward1(dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        case 2:
            encoder_backward2(dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    printf("Using Platform: %s\n", defaultQueue.get_device().get_platform().get_info<sycl::info::platform::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }
    DefaultQueue = &defaultQueue;

    // create host memory of random numbers
    float* dout = make_random_float(B * T * C);
    int* inp = make_random_int(B * T, V);
    float* dwte = make_zeros_float(V * C);
    float* dwpe = make_zeros_float(T * C);

    // move to GPU
    float* d_dout;
    int* d_inp;
    float* d_dwte;
    float* d_dwpe;
    syclMallocCheck(d_dout = sycl::malloc_device<float>(B * T * C, defaultQueue));
    syclMallocCheck(d_inp = sycl::malloc_device<int>(B * T, defaultQueue));
    syclMallocCheck(d_dwte = sycl::malloc_device<float>(V * C, defaultQueue));
    syclMallocCheck(d_dwpe = sycl::malloc_device<float>(T * C, defaultQueue));
    defaultQueue.memcpy(d_dout, dout, B * T * C * sizeof(float));
    defaultQueue.memcpy(d_inp, inp, B * T * sizeof(int));
    defaultQueue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // set up block sizes
    // Nix 1024 for Intel GPUs
    int block_sizes[] = {32, 64, 128, 256, 512};

    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        encoder_backward_cpu(dwte, dwpe, dout, inp, B, T, C);
        encoder_backward(kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        validate_result(d_dwte, dwte, "dwte", V * C, 1e-5f);
        validate_result(d_dwpe, dwpe, "dwpe", T * C, 1e-5f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, encoder_backward,
                                              kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(dout);
    free(inp);
    free(dwte);
    free(dwpe);
    sycl::free(d_dout, defaultQueue);
    sycl::free(d_inp, defaultQueue);
    sycl::free(d_dwte, defaultQueue);
    sycl::free(d_dwpe, defaultQueue);

    return 0;
}
