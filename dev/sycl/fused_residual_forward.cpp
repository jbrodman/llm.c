/*
Kernels for residual forward pass fused with layernorm

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt fused_residual_forward.cu -o fused_residual_forward

version 1 is naive port from CPU code to kernel
./fused_residual_forward 1
version 2 packs input into 128 bit memory reads
./fused_residual_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <sycl/sycl.hpp>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference lol

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

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

// elementwise ops are nice and ez
void residual_forward_kernel1(sycl::nd_item<1> id, floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = id.get_global_id(0);
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
void layernorm_forward_kernel1(sycl::nd_item<1> id, floatX* out, floatX* mean, floatX* rstd,
                                          const floatX* inp, const floatX* weight, const floatX* bias,
                                          int N, int C) {
    int idx = id.get_global_id(0);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const floatX* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += (float)x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = (float)x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sycl::sqrt(v + eps);
        // seek to the output position in out[idx,:]
        floatX* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * ((float)x[i] - m)); // normalized output
            float o = n * (float)weight[i] + (float)bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// naive fusion; uncoalesced access pattern leads to terrible performance
void fused_residual_forward2(sycl::nd_item<1> id, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    int idx = id.get_global_id(0);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;

    float m = 0.0f;
    for(int c = 0; c < C; ++c) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = out;
    }

    m = m / C;
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = 0; c < C; c++) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = m;
    rstd[idx] = s;
}

// For the rest, for now, let's just set reqd wg size of 16 and call it a day.

// handle one token per warp for coalesced access
void fused_residual_forward3(sycl::nd_item<2> id, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    constexpr const int WarpSize = 16;
    int idx = blockIdx_x(id) * blockDim_y(id) + threadIdx_y(id);
    if(idx > N) return;

    sycl::sub_group warp = id.get_sub_group();

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;
    float m = 0.0f;
    for(int c = threadIdx_x(id); c < C; c += WarpSize) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = out;
    }

    m = warpReduceSum(warp, m);

    m = m / C;
    float v = 0.0f;
    for(int c = threadIdx_x(id); c < C; c += WarpSize) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }

    v = warpReduceSum(warp, v);
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sycl::sqrt(v + eps);
    for(int c = threadIdx_x(id); c < C; c += WarpSize) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx_x(id) == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// vectorized loading, single pass stats, streaming access and zigzag loop
void fused_residual_forward_kernel4(sycl::nd_item<2> id, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    using x128 = Packed128<floatX>;
    constexpr const int WarpSize = 32;
    int idx = blockIdx_x(id) * blockDim_y(id) + threadIdx_y(id);
    if(idx > N) return;

    sycl::sub_group warp = id.get_sub_group();

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int c = threadIdx_x(id) * x128::size;
    for(; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (float)in1[k] + (float)in2[k];
            sum += (float)out[k];
            sum_sq += (float)out[k] * (float)out[k];
        }
        store128(residual + c, out);
    }

    sum = warpReduceSum(warp, sum);
    sum_sq = warpReduceSum(warp, sum_sq);

    float m = sum / C;
    float v = sum_sq / C - m * m;
    float s = sycl::rsqrt(v + eps);

    c -= WarpSize * x128::size;
    for(; c >= 0; c -= WarpSize * x128::size) {
        const x128 res = load128cs(residual + c);
        const x128 w = load128(weight + c);
        const x128 b = load128(bias + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx_x(id) == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// what do you want in shared memory? EVERYTHING!
// thus, we no longer require zigzag loops and can do the numerically more stable variance estimation
// needs special attention in the kernel launcher to ensure we have enough smem.
void fused_residual_forward_kernel5(sycl::nd_item<2> id, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C, sycl::local_accessor<char> lmem) {
    constexpr const int WarpSize = 32;

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    char* params = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res = reinterpret_cast<x128*>(params) + ((2 + threadIdx_y(id)) * C / x128::size);

    int sidx = (threadIdx_x(id) + WarpSize * threadIdx_y(id)) * x128::size;
    for(int i = sidx; i < C; i += blockDim_y(id) * WarpSize * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    sycl::group_barrier(id.get_group());

    int idx = blockIdx_x(id) * blockDim_y(id) + threadIdx_y(id);
    if(idx > N) return;

    sycl::sub_group warp = id.get_sub_group();

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx_x(id) * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (float)in1[k] + (float)in2[k];
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum = warpReduceSum(warp, sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx_x(id) * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(warp, v) / C;
    float s = sycl::rsqrt(v + eps);

    for(int c = threadIdx_x(id) * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx_x(id) == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}


// using multiple warps per token, and keep threads persistent, so we never have to reload weights and biases
// if we had one warp per token, though, this would require us to use a huge amount of shared memory. Therefore,
// we use multiple warps per token; but generally we cannot use the entire block, because that would give too
// little work per warp to be effective (each warp processes 256 bfloat16 elements, so for C=768 more than 3 warps
// will just mean idle). Therefore, we add a z dimension, where warps with different z handle different tokens.
// all this makes the launcher logic more complicated :(
void fused_residual_forward_kernel6(sycl::nd_item<3> id, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C, sycl::local_accessor<char> lmem) {
    constexpr const int WarpSize = 32;
    sycl::sub_group warp = id.get_sub_group();

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    char* params = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    // weights and biases are  shared among all tokens
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params + C * sizeof(floatX));
    // residual output (input to layernorm) is indpendent for each sub-block indicates by threadIdx.z
    x128* s_res = reinterpret_cast<x128*>(params + (2 + threadIdx_z(id)) * C * sizeof(floatX)  );
    // similarly, each sub-block needs its own reduction buffers
    float* s_mean = reinterpret_cast<float*>(params + (2 + blockDim_z(id)) * C * sizeof(floatX) + threadIdx_z(id) * WarpSize * sizeof(float));
    float* s_var = reinterpret_cast<float*>(params + (2 + blockDim_z(id)) * C * sizeof(floatX) + WarpSize * sizeof(float) * (blockDim_z(id) + threadIdx_z(id)));

    int cidx = (threadIdx_x(id) + WarpSize * threadIdx_y(id)) * x128::size;
    int step = blockDim_y(id) * WarpSize * x128::size;

    for(int c = cidx; c < C; c += step) {
        s_weight[c / x128::size] = load128(weight + c);
        s_bias[c / x128::size] = load128(bias + c);
    }
    // the block-level reductions will cause sync before the first time we read these
    // => no syncthreads needed here


    // loop over all tokens
    for(int tidx = blockIdx_x(id) * blockDim_z(id) + threadIdx_z(id); 
        tidx < N; 
        tidx += gridDim_x(id) * blockDim_z(id)) {
        // adjust pointers to current token
        floatX* residual_bt = residual + C * tidx;
        floatX* normed_bt = normed + C * tidx;
        const floatX* inp1_bt = inp1 + C * tidx;
        const floatX* inp2_bt = inp2 + C * tidx;

        const float eps = 1e-5f;
        float sum = 0.0f;
        for (int c = cidx; c < C; c += step) {
            const x128 in1 = load128cs(inp1_bt + c);
            const x128 in2 = load128cs(inp2_bt + c);
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                out[k] = (float) in1[k] + (float) in2[k];
                sum += (float) out[k];
            }
            store128cs(residual_bt + c, out);
            s_res[c / x128::size] = out;
        }
        sum = warpReduceSum(warp, sum);
        if(threadIdx_x(id) == 0) {
            s_mean[threadIdx_y(id)] = sum;
        }
        sycl::group_barrier(id.get_group());
        float m = warpReduceSum(warp, threadIdx_x(id) < blockDim_y(id) ? s_mean[threadIdx_x(id)] : 0.f) / C;
        // normally, we'd syncthread here to make sure that no warp is already at the next
        // iteration of the loop, messing with s_mean. The fact that we interleave s_mean and s_var means
        // we don't need these additional syncs.
        float v = 0.f;

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            for (int k = 0; k < x128::size; ++k) {
                v += ((float) res[k] - m) * ((float) res[k] - m);
            }
        }

        v = warpReduceSum(warp, v);
        if(threadIdx_x(id)== 0) {
            s_var[threadIdx_y(id)] = v;
        }
        sycl::group_barrier(id.get_group());
        v = warpReduceSum(warp, threadIdx_x(id) < blockDim_y(id) ? s_var[threadIdx_x(id)] : 0.f) / C;
        float s = sycl::rsqrt(v + eps);

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            const x128 w = s_weight[c / x128::size];
            const x128 b = s_bias[c / x128::size];
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                float n = s * ((float) res[k] - m); // normalized output
                float o = n * (float) w[k] + (float) b[k]; // scale and shift it
                out[k] = o;
            }

            store128(normed_bt + c, out);
        }
        // cache the mean and rstd for the backward pass later
        if (threadIdx_x(id) == 0 && threadIdx_y(id) == 0) {
            mean[tidx] = m;
            rstd[tidx] = s;
        }
    }
}



// ----------------------------------------------------------------------------
// kernel launcher

void fused_residual_forward1(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size_resid*block_size, block_size), [=](sycl::nd_item<1> id) {
        residual_forward_kernel1(id, residual, inp1, inp2, N*C);
    });
    const int grid_size_ln = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size_ln*block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel1(id, normed, mean, rstd, residual, weight, bias, N, C);
    });
}

void fused_residual_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size));
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        fused_residual_forward2(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    });
}

void fused_residual_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    // Let's go with SIMD16 for now
    int block_y = block_size / 16;
    const int grid_size = ceil_div(N, block_y);
    sycl::range<2> grid_range(1, grid_size);
    sycl::range<2> block_range(block_y, 16);
    DefaultQueue->parallel_for(sycl::nd_range<2>(grid_range*block_range, block_range), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(16)]] {
        fused_residual_forward3(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    });
}

void fused_residual_forward4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    // Going with SIMD16 for now
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    sycl::range<2> grid_range(1, grid_size);
    sycl::range<2> block_range(block_y, 32);
    DefaultQueue->parallel_for(sycl::nd_range<2>(grid_range*block_range, block_range), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]] {
        fused_residual_forward_kernel4(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    });
}

void fused_residual_forward5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    // SIMD16
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);
    sycl::range<2> grid_range(1, grid_size);
    sycl::range<2> block_range(block_y, 32);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    /*
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    */
    sycl::device d = DefaultQueue->get_device();
    uint64_t local_mem_size = d.get_info<sycl::info::device::local_mem_size>();
    if(local_mem_size > smem) {
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<char> lmem(smem, h);
            h.parallel_for(sycl::nd_range<2>(grid_range*block_range, block_range), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]] {
                fused_residual_forward_kernel5(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, lmem);
            });
        });
    } else {
        DefaultQueue->parallel_for(sycl::nd_range<2>(grid_range*block_range, block_range), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]] {
            fused_residual_forward_kernel4(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
        });
    }
}

void fused_residual_forward6(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    constexpr int warpSize = 32;
    int warps_per_token = std::max(1, C / Packed128<floatX>::size / warpSize);
    int total_warps = block_size / warpSize;
    int block_z = std::max(1, total_warps / warps_per_token);
    int block_y = std::max(1, total_warps / block_z);
    size_t smem = (2 + block_z) * C * sizeof(floatX) + 64 * sizeof(float) * block_z;
   
    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    /*
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    */
    sycl::device d = DefaultQueue->get_device();
    uint64_t local_mem_size = d.get_info<sycl::info::device::local_mem_size>();
    if(local_mem_size > smem) {
        // Approximate something
        int size = DefaultQueue->get_device().get_info<sycl::info::device::max_compute_units>();
        size *= warpSize;
        const int num_blocks = std::max(1, size / block_size);
        sycl::range<3> grid_range(1, 1, num_blocks);
        sycl::range<3> block_range(block_z, block_y, warpSize);
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<char> lmem(smem, h);
            h.parallel_for(sycl::nd_range<3>(grid_range*block_range, block_range), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                fused_residual_forward_kernel6(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, lmem);
            });
        });
    } else {
        const int grid_size = ceil_div(N, total_warps);
        sycl::range<2> grid_range(1, grid_size);
        sycl::range<2> block_range(total_warps, warpSize);
        DefaultQueue->parallel_for(sycl::nd_range<2>(grid_range*block_range, block_range), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]] {
            fused_residual_forward_kernel4(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
        });
    }
}

// kernel version dispatch
void fused_residual_forward(int kernel_num, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                            const floatX* inp1, const floatX* inp2,
                            const floatX* weight, const floatX* bias,
                            int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            fused_residual_forward1(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 2:
            fused_residual_forward2(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 3:
            fused_residual_forward3(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 4:
            fused_residual_forward4(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 5:
            fused_residual_forward5(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 6:
            fused_residual_forward6(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
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

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // create host memory of random numbers
    float* residual = (float*)malloc(B * T * C * sizeof(float));
    float* normed = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);
    
    // move to GPU
    floatX* d_residual;
    floatX* d_normed;
    floatX* d_inp1;
    floatX* d_inp2;
    floatX* d_mean;
    floatX* d_rstd;
    floatX* d_weight;
    floatX* d_bias;
    syclMallocCheck(d_residual = sycl::malloc_device<floatX>(B * T * C, *DefaultQueue));
    syclMallocCheck(d_normed = sycl::malloc_device<floatX>(B * T * C, *DefaultQueue));
    syclMallocCheck(d_inp1 = sycl::malloc_device<floatX>(B * T * C, *DefaultQueue));
    syclMallocCheck(d_inp2 = sycl::malloc_device<floatX>(B * T * C, *DefaultQueue));
    // Is this a bug in llm.c? floatX* and sizeof(float)?
    syclMallocCheck(d_mean = (floatX*)sycl::malloc_device<float>(B * T, *DefaultQueue));
    syclMallocCheck(d_rstd = (floatX*)sycl::malloc_device<float>(B * T, *DefaultQueue));
    syclMallocCheck(d_weight = (floatX*)sycl::malloc_device<float>(C, *DefaultQueue));
    syclMallocCheck(d_bias = (floatX*)sycl::malloc_device<float>(C, *DefaultQueue));
    memcpy_convert(d_inp1, inp1, B * T * C);
    memcpy_convert(d_inp2, inp2, B * T * C);
    memcpy_convert(d_weight, weight, C);
    memcpy_convert(d_bias, bias, C);

    // first check the correctness of the kernel
    residual_forward_cpu(residual, inp1, inp2, B * T * C);
    layernorm_forward_cpu(normed, mean, rstd, residual, weight, bias, B, T, C);

    // time the kernel at different block sizes
    // 512 borking my GPU on kernel6- investigate later
    int block_sizes[] = {32, 64, 128, 256};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        DefaultQueue->memset(d_residual, 0, B * T * C * sizeof(floatX));
        fused_residual_forward(kernel_num, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                               B*T, C, block_size);
        float tol = std::is_same_v<floatX, float> ? 1e-5 : 5e-2;
        validate_result(d_residual, residual, "residual", B * T * C, tol);
        validate_result(d_mean, mean, "mean", B * T, tol);
        validate_result(d_rstd, rstd, "rstd", B * T, tol);
        validate_result(d_normed, normed, "normed", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, fused_residual_forward, kernel_num,
                                              d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                                              B*T, C, block_size
                                              );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 reads and 2 writes, plus 2 BT writes for mean/rstd
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * (C * 4 + 2) * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        float toks_per_msec = B * T / elapsed_time / 1e3;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s | elements: %.2f ktok/ms\n",
               block_size, elapsed_time, memory_bandwidth, toks_per_msec);
    }

    // free memory
    free(residual);
    free(normed);
    free(mean);
    free(rstd);
    free(weight);
    free(bias);
    free(inp1);
    free(inp2);
    sycl::free(d_residual, *DefaultQueue);
    sycl::free(d_normed, *DefaultQueue);
    sycl::free(d_mean, *DefaultQueue);
    sycl::free(d_rstd, *DefaultQueue);
    sycl::free(d_weight, *DefaultQueue);
    sycl::free(d_bias, *DefaultQueue);
    sycl::free(d_inp1, *DefaultQueue);
    sycl::free(d_inp2, *DefaultQueue);

    return 0;
}
