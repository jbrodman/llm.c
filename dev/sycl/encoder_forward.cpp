/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math encoder_forward.cu -o encoder_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./encoder_forward 1

version 2 is more optimized, parallelizes over all of B,T,C
./encoder_forward 2

version 3 is like version 2 but uses float4 reads/writes
./encoder_forward 3
*/

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include "common.h"
#include <cassert>

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
void encoder_forward_cpu(float* out,
                   const int* inp, const float* wte, const float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation into kernel, parallelize over B,T, loop over C
void encoder_forward_kernel1(sycl::nd_item<1> id, float* out,
                               const int* inp, const float* wte, const float* wpe,
                               int B, int T, int C) {
    int idx = id.get_global_id(0);
    int N = B * T;

    if (idx < N) {
        int b = idx / T;
        int t = idx % T;
        float* out_bt = out + b * T * C + t * C;
        int ix = inp[b * T + t];
        const float* wte_ix = wte + ix * C;
        const float* wpe_t = wpe + t * C;
        for (int i = 0; i < C; i++) {
            out_bt[i] = wte_ix[i] + wpe_t[i];
        }
    }
}

// optimized implementation: parallelize over all of B,T,C
void encoder_forward_kernel2(sycl::nd_item<1> id, float* out,
                               const int* inp, const float* wte, const float* wpe,
                               int B, int T, int C) {
    int idx = id.get_global_id(0);
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        float* out_btc = out + b * T * C + t * C + c;
        const float* wte_ix = wte + ix * C + c;
        const float* wpe_tc = wpe + t * C + c;
        *out_btc = *wte_ix + *wpe_tc;
    }
}

// use of float4 leads to using 128-bit LDG / STG instructions in SASS,
// very helpful in memory-bound kernels like encoder_forward
void encoder_forward_kernel3(sycl::nd_item<1> id, sycl::float4* out,
                               const int* inp, const sycl::float4* wte, const sycl::float4* wpe,
                               int B, int T, int C) {
    int C4 = C / 4;
    int idx = id.get_global_id(0);
    int N = B * T * C4;
    if (idx < N) {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[b * T + t];
        // SYCL supports adds on vec types
        // out[b * T * C4 + t * C4 + c4] = add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
        out[b * T * C4 + t * C4 + c4] = wte[ix * C4 + c4] + wpe[t * C4 + c4];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_forward1(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_forward_kernel1(id, out, inp, wte, wpe, B, T, C);
    });
}

void encoder_forward2(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_forward_kernel2(id, out, inp, wte, wpe, B, T, C);
    });
}

void encoder_forward3(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C,
                     const int block_size) {
    assert(C % 4 == 0);
    const int N = B * T * C;
    const int grid_size = ceil_div(N / 4, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_forward_kernel3(id, (sycl::float4*) out, inp, (sycl::float4*) wte, (sycl::float4*) wpe, B, T, C);
    });
}

// kernel version dispatch
void encoder_forward(int kernel_num,
                     float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C,
                     const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_forward1(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 2:
            encoder_forward2(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 3:
            encoder_forward3(out, inp, wte, wpe, B, T, C, block_size);
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
    float* out = (float*)malloc(B * T * C * sizeof(float));
    int* inp = make_random_int(B * T, V);
    float* wte = make_random_float(V * C);
    float* wpe = make_random_float(T * C);

    // move to GPU
    float* d_out;
    int* d_inp;
    float* d_wte;
    float* d_wpe;
    syclMallocCheck(d_out = sycl::malloc_device<float>(B * T * C, *DefaultQueue));
    syclMallocCheck(d_inp = sycl::malloc_device<int>(B * T, *DefaultQueue));
    syclMallocCheck(d_wte = sycl::malloc_device<float>(V * C, *DefaultQueue));
    syclMallocCheck(d_wpe = sycl::malloc_device<float>(T * C, *DefaultQueue));
    defaultQueue.memcpy(d_inp, inp, B * T * sizeof(int));
    defaultQueue.memcpy(d_wte, wte, V * C * sizeof(float));
    defaultQueue.memcpy(d_wpe, wpe, T * C * sizeof(float));
    defaultQueue.wait();

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);


    // time the kernel at different block sizes
    // Nix 1024 for Intel GPUs
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        encoder_forward(kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);
        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, encoder_forward,
                                              kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size
                                              );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);
    free(wte);
    free(wpe);
    sycl::free(d_out, defaultQueue);
    sycl::free(d_inp, defaultQueue);
    sycl::free(d_wte, defaultQueue);
    sycl::free(d_wpe, defaultQueue);

    return 0;
}