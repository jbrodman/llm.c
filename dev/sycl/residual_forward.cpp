/*
Kernels for residual forward pass.

Compile example:
icpx -fsycl -O3 residual_forward.cpp -o residual_forward

version 1 is naive port from CPU code to kernel
./residual_forward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference lol

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
void residual_forward_kernel(sycl::nd_item<1> id, float* out, const float* inp1, const float* inp2, int N) {
    int idx = id.get_global_id(0);
    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward1(float* out, const float* inp1, const float* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        residual_forward_kernel(id, out, inp1, inp2, N);
    });
}

// kernel version dispatch
void residual_forward(int kernel_num,
                  float* out,
                  const float* inp1,
                  const float* inp2,
                  int N,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
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
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);

    // move to GPU
    float* d_out = sycl::malloc_device<float>(B * T * C, defaultQueue);
    float* d_inp1 = sycl::malloc_device<float>(B * T * C, defaultQueue);
    float* d_inp2 = sycl::malloc_device<float>(B * T * C, defaultQueue);
  
    syclMallocCheck(d_out);
    syclMallocCheck(d_inp1);
    syclMallocCheck(d_inp2);  

    defaultQueue.memcpy(d_inp1, inp1, B * T * C * sizeof(float));    
    defaultQueue.memcpy(d_inp2, inp2, B * T * C * sizeof(float));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, B * T * C);


    // time the kernel at different block sizes
    // 1024 is not supported on Intel GPUs
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size);
        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, residual_forward,
                                              kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size
                                              );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp1);
    free(inp2);
    
    sycl::free(d_out, defaultQueue);
    sycl::free(d_inp1, defaultQueue);
    sycl::free(d_inp2, defaultQueue);

    return 0;
}