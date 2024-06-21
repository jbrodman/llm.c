/*
Kernels for layernorm forward pass.

Compile example:
icpx -fsycl -O3 layernorm_forward.cpp -o layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
          (allowing us to do a single pass over x on load)
./layernorm_forward 4

verstion 5 allocates blocks per row instead of warps per row, same alg as 4 otherwise
./layernorm_forward 5
*/

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include <assert.h>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
void layernorm_forward_kernel1(sycl::nd_item<1> id, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               int N, int C) {
    int idx = id.get_global_id(0);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const float* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void mean_kernel(sycl::nd_item<1> id, float* mean, const float* inp, int N, int C, int block_size) {
    int idx = blockIdx_x(id); // range [0, B*T)
    int tid = threadIdx_x(id); // range [0, block_size)
    const float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<float>());
    
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = sum / C;
    }
}

void rstd_kernel(sycl::nd_item<1> id, float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    int idx = blockIdx_x(id); // range [0, B*T)
    int tid = threadIdx_x(id); // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<float>());
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(sum / C + 1e-5f);
    }
}

void normalization_kernel(sycl::nd_item<1> id, float* out, const float* inp, float* mean, float* rstd,
                          const float* weight, const float* bias, int B, int T, int C) {
    int idx = id.get_global_id(0);

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

// ----------------------------------------------------------------------------

void layernorm_forward_kernel3(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                               const float*  __restrict__ inp, const float*  __restrict__ weight,
                               const float* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = thread_rank(warp); i < C; i += size(warp)) {
        sum += x[i];
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    float m = sum / C;
    if(warp.leader() && mean != nullptr) {
        // Fix later
        //__stcs(mean + idx, m);
        mean[idx] = m;
    }

    // rstd
    sum = 0.0f;
    for (int i = thread_rank(warp); i < C; i += size(warp)) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if(warp.leader() && rstd != nullptr) {
        // Fix later
        //__stcs(rstd + idx, s);
        rstd[idx] = s;
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = thread_rank(warp); c < C; c += size(warp)) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        // Fix later
        // float n = s * (__ldcs(x+c) - m);
        float n = s * (x[c] - m);
        //__stcs(o+c, n * weight[c] + bias[c]);
        o[c] = n * weight[c] + bias[c];
    }
}

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
void layernorm_forward_kernel4(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // thread coarsening through the row, reduce the sum in series
    float sum = 0.0; // stores sum(x)
    float sum2 = 0.0; // stores sum(x**2)
    for (int i = thread_rank(warp); i < C; i += size(warp)) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // warp-level reduction at the end
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{}); // sum(x)
    sum2 = sycl::reduce_over_group(warp, sum2, sycl::plus<float>{}); // sum(x**2)
    sum /= C; // mean(x)
    sum2 /= C; // mean(x**2)

    // mean, var, rstd
    float m = sum;
    float var = sum2 - sum * sum;
    float s = sycl::rsqrt(var + 1e-5f);

    // store the mean, no need to cache it
    if(warp.leader() && mean != nullptr) {
        // Fix this later
        //__stcs(mean + idx, m);
        mean[idx] = m;
    }
    // store the rstd, no need to cache it
    if(warp.leader() && rstd != nullptr) {
        //__stcs(rstd + idx, s);
        rstd[idx] = s;
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = thread_rank(warp); c < C; c += size(warp)) {
        // Fix later
        // float n = s * (__ldcs(x+c) - m);
        float n = s * (x[c] - m);
        //__stcs(o+c, n * weight[c] + bias[c]);
        o[c] = n * weight[c] + bias[c];
    }
}

// like 4, but in kernel 5 we have each block doing one row, not just a single warp
void layernorm_forward_kernel5(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int idx = blockIdx_x(id); // simpoy one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum = 0.0; // stores sum(x)
    float thread_sum2 = 0.0; // stores sum(x**2)
    // for (int i = C + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
    for (int i = threadIdx_x(id); i < C; i += blockDim_x(id)) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    // block-level reduction
    float block_sum = sycl::reduce_over_group(block, thread_sum, sycl::plus<float>{}); // sum(x)
    float block_sum2 = sycl::reduce_over_group(block, thread_sum2, sycl::plus<float>{}); // sum(x**2)
    // mean, var, rstd
    block_sum /= C; // mean(x)
    block_sum2 /= C; // mean(x**2)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = sycl::rsqrt(var + 1e-5f);
    // store the mean, no need to cache it
    if(threadIdx_x(id) == 0 && mean != nullptr) {
        //__stcs(mean + idx, m);
        mean[idx] = m;
    }
    // store the rstd, no need to cache it
    if(threadIdx_x(id) == 0 && rstd != nullptr) {
        //__stcs(rstd + idx, s);
        rstd[idx] = s;
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = threadIdx_x(id); i < C; i += blockDim_x(id)) {
        /*float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);*/
        float n = s * (x[i] - m);
        o[i] = n * weight[i] + bias[i];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel1(id, out, mean, rstd, inp, weight, bias, N, C);
    });
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    // Do these still need the block_size param in the kernels?
    DefaultQueue->parallel_for(sycl::nd_range<1>(N * block_size, block_size), [=](sycl::nd_item<1> id) {
        mean_kernel(id, mean, inp, N, C, block_size);
    });
    DefaultQueue->parallel_for(sycl::nd_range<1>(N * block_size, block_size), [=](sycl::nd_item<1> id) {
        rstd_kernel(id, rstd, inp, mean, N, C, block_size);
    });
       // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size2, block_size2), [=](sycl::nd_item<1> id) {
        normalization_kernel(id, out, inp, mean, rstd, weight, bias, B, T, C);
    });
}

void layernorm_forward3(float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            layernorm_forward_kernel3(id, out, mean, rstd, inp, weight, bias, N, C);
    });
}

void layernorm_forward4(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel4(id, out, mean, rstd, inp, weight, bias, N, C);
    });
}

void layernorm_forward5(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N;
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel5(id, out, mean, rstd, inp, weight, bias, N, C);
    });
}

// kernel version dispatch
void layernorm_forward(int kernel_num,
                       float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 4:
            layernorm_forward4(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 5:
            layernorm_forward5(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
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
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // move to GPU
    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    syclMallocCheck(d_out = sycl::malloc_device<float>(B * T * C, defaultQueue));
    syclMallocCheck(d_mean = sycl::malloc_device<float>(B * T, defaultQueue));
    syclMallocCheck(d_rstd = sycl::malloc_device<float>(B * T, defaultQueue));
    syclMallocCheck(d_inp = sycl::malloc_device<float>(B * T * C, defaultQueue));
    syclMallocCheck(d_weight = sycl::malloc_device<float>(C, defaultQueue));
    syclMallocCheck(d_bias = sycl::malloc_device<float>(C, defaultQueue));

    defaultQueue.memcpy(d_inp, inp, B * T * C * sizeof(float));
    defaultQueue.memcpy(d_weight, weight, C * sizeof(float));
    defaultQueue.memcpy(d_bias, bias, C * sizeof(float));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // 1024 is not supported on Intel GPUs
    int block_sizes[] = {32, 64, 128, 256, 512};
    float* out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* mean_gpu = (float*)malloc(B * T * sizeof(float));
    float* rstd_gpu = (float*)malloc(B * T * sizeof(float));

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_forward,
                                              kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);

    sycl::free(d_out, defaultQueue);
    sycl::free(d_mean, defaultQueue);
    sycl::free(d_rstd, defaultQueue);
    sycl::free(d_inp, defaultQueue);
    sycl::free(d_weight, defaultQueue);
    sycl::free(d_bias, defaultQueue);

    return 0;
}