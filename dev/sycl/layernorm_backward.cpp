/*
Kernels for layernorm backward pass.

Compile example:
nvcc -O3 --use_fast_math layernorm_backward.cu -o layernorm_backward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_backward 1

version 2 moves a lot of reduction to shared memory over global memory
./layernorm_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sycl/sycl.hpp>
#define ENABLE_BF16
#include "common.h"

namespace syclx = sycl::ext::oneapi;
using bfloat162 = sycl::marray<syclx::bfloat16, 2>; 

#define WARP16 16
#define WARP32 32

// ----------------------------------------------------------------------------
// CPU code reference

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
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
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// GPU helper functions for atomicAdd on smaller than 32-bit types
#ifdef ENABLE_BF16
void atomicAddX(syclx::bfloat16* addr, syclx::bfloat16 val) {
    /*
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    __nv_bfloat162* ptr_bf16 = reinterpret_cast<__nv_bfloat162*>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    __nv_bfloat162 add_val = (ptr_val & 0x3) ? __halves2bfloat162(__ushort_as_bfloat16(0), val)
                                             : __halves2bfloat162(val, __ushort_as_bfloat16(0));
    atomicAdd(ptr_bf16, add_val);
    */
    // I think the best thing to do here for now is to compare and swap
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    uint32_t* ptr_32bits = reinterpret_cast<uint32_t*>(ptr_val & ~uintptr_t(0x3));

    sycl::atomic_ref<uint32_t, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*ptr_32bits);
    uint32_t old_val = ref.load(); 
    uint32_t new_val = old_val;
    do {
        sycl::marray<syclx::bfloat16, 2> h2 = *reinterpret_cast<sycl::marray<syclx::bfloat16, 2>*>(&old_val);
        h2[0] += (ptr_val & 0x3) ? syclx::bfloat16(0.0f) : val;
        h2[1] += (ptr_val & 0x3) ? val : syclx::bfloat16(0.0f);
        new_val = *reinterpret_cast<uint32_t*>(&h2);
    }
    while (!ref.compare_exchange_weak(old_val, new_val));
}
#endif
#ifdef ENABLE_FP16
void atomicAddX(sycl::half* addr, sycl::half val) {
    // Same thing as bfloat16
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    uint32_t* ptr_32bits = reinterpret_cast<uint32_t*>(ptr_val & ~uintptr_t(0x3));

    sycl::atomic_ref<uint32_t, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*ptr_32bits);
    uint32_t old_val = ref.load(); 
    uint32_t new_val = old_val;
    do {
        sycl::marray<sycl::half, 2> h2 = *reinterpret_cast<sycl::marray<sycl::half, 2>*>(&old_val);
        h2[0] += (ptr_val & 0x3) ? sycl::half(0.0f) : val;
        h2[1] += (ptr_val & 0x3) ? val : sycl::half(0.0f);
        new_val = *reinterpret_cast<uint32_t*>(&h2);
    }
    while (!ref.compare_exchange_weak(old_val, new_val));
}
#endif
void atomicAddX(float* addr, float val) {
    sycl::atomic_ref<float, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*addr);
    ref += val;
}

// super naive kernel that just parallelizes over B,T and loops over C
void layernorm_backward_kernel1(sycl::nd_item<1> id, float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    int idx = id.get_global_id(0);
    if (idx >= B*T) return;
    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
}

// uses shared memory instead for the reduces
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward_kernel2(sycl::nd_item<1> id, Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, float* dweight_tmp, float* dbias_tmp, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    const Tdout* dout_bt = dout + b * T * C + t * C;
    const Trest* inp_bt = inp + b * T * C + t * C;
    Tdinp* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = (float)mean[b * T + t];
    const float rstd_bt = (float)rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    sycl::group_barrier(block);

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = thread_rank(warp); i < C; i  += size(warp)) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = sycl::reduce_over_group(warp, dnorm_mean, sycl::plus<float>{});
    dnorm_norm_mean = sycl::reduce_over_group(warp, dnorm_norm_mean, sycl::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = thread_rank(warp); i < C; i += size(warp)) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], (float)dout_bt[i]);             
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * (float)dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
    }
    sycl::group_barrier(block);

    // write to global memory
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
        atomicAdd(&dbias_tmp[i], dbias_shared[i]);
        atomicAdd(&dweight_tmp[i], dweight_shared[i]);
    }
}

template <typename Tparams>
void copy_to_dweight_dbias(sycl::nd_item<1> id, int C, Tparams* dbias, Tparams* dweight, float* dbias_tmp, float* dweight_tmp) {
    for (int i = id.get_global_id(0); i < C; i += blockDim_x(id) * gridDim_x(id)) {
        dbias[i] = (Tparams)dbias_tmp[i];
        dweight[i] = (Tparams)dweight_tmp[i];
    }
}

// kernel2 is 1 threadblock for all Cs on 32 BTs (assuming threadblock size of 1024 threads = 32 warps)
// To minimise the amount of atomicAdds, we will aim for 1 threadblock per SM, processing (total BTs / threadblocks) BTs
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward_kernel3(sycl::nd_item<1> id, Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int base_idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    sycl::group_barrier(block);

    int warps_in_grid = gridDim_x(id) * meta_group_size(warp);
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = thread_rank(warp); i < C; i  += size(warp)) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(warp, dnorm_mean, sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(warp, dnorm_norm_mean, sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = thread_rank(warp); i < C; i += size(warp)) {
            // Fix this later
            // float dout_i = (float)__ldcs(&dout_bt[i]);
            // float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dout_i = (float)dout_bt[i];
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    sycl::group_barrier(block);

    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
        atomicAddX(&dbias[i], (Tparams)dbias_shared[i]);
        atomicAddX(&dweight[i], (Tparams)dweight_shared[i]);
    }
}

// atomicCAS version of kernel3
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward_kernel4(sycl::nd_item<1> id, Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int base_idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    sycl::group_barrier(block);

    int warps_in_grid = gridDim_x(id) * meta_group_size(warp);
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = thread_rank(warp); i < C; i  += size(warp)) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(warp, dnorm_mean, sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(warp, dnorm_norm_mean, sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = thread_rank(warp); i < C; i += size(warp)) {
            // Fix this later
            // float dout_i = (float)__ldcs(&dout_bt[i]);
            // float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dout_i = (float)dout_bt[i];
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    sycl::group_barrier(block);   

    bfloat162* dbiasVec2 = reinterpret_cast<bfloat162*>(dbias);
    bfloat162* dweightVec2 = reinterpret_cast<bfloat162*>(dweight);

    // write to global memory
    for(int i = threadIdx_x(id); i < C/2; i+= blockDim_x(id)) {
        bfloat162 add_dbias = bfloat162(dbias_shared[i*2], dbias_shared[i*2+1]);
        bfloat162 add_dweight = bfloat162(dweight_shared[i*2], dweight_shared[i*2+1]);

        // Get the current value from L2 cache
        // Fix this later
        bfloat162 current_dbias = dbiasVec2[i];
        bfloat162 current_dweight = dweightVec2[i];

        // Add the two values
        bfloat162 new_dbias = add_dbias + current_dbias;
        bfloat162 new_dweight = add_dweight + current_dweight;

        // Write the result back to L2 cache using 32-bit integer atomic compare and exchange
        unsigned int current_dbias32b = *reinterpret_cast<unsigned int*>(&current_dbias);
        unsigned int current_dweight32b = *reinterpret_cast<unsigned int*>(&current_dweight);

        unsigned int new_dbias32b = *reinterpret_cast<unsigned int*>(&new_dbias);
        unsigned int new_dweight32b = *reinterpret_cast<unsigned int*>(&new_dweight);
        
        unsigned int old_dbias32b = atomicCAS((unsigned int*)&dbiasVec2[i], current_dbias32b, new_dbias32b);
        unsigned int old_dweight32b = atomicCAS((unsigned int*)&dweightVec2[i], current_dweight32b, new_dweight32b);

        // If the value has changed between read and atomic, we need to try again
        while (old_dbias32b != current_dbias32b) {
            current_dbias32b = old_dbias32b;
            new_dbias = *reinterpret_cast<bfloat162*>(&current_dbias32b) + add_dbias;
            new_dbias32b = *reinterpret_cast<unsigned int*>(&new_dbias);
            old_dbias32b = atomicCAS((unsigned int*)&dbiasVec2[i], current_dbias32b, new_dbias32b);
        }

        while (old_dweight32b != current_dweight32b) {
            current_dweight32b = old_dweight32b;
            new_dweight = *reinterpret_cast<bfloat162*>(&current_dweight32b) + add_dweight;
            new_dweight32b = *reinterpret_cast<unsigned int*>(&new_dweight);
            old_dweight32b = atomicCAS((unsigned int*)&dweightVec2[i], current_dweight32b, new_dweight32b);
        }
    }
}

// FP32 scratchpad per threadgroup, zero atomics except atomicAdd on uint for the flag (based on kernel3)
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward_kernel5(sycl::nd_item<1> id, Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C  

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int base_idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    uint *tmp_flag = (uint*)(shared + C*2);
    sycl::group_barrier(block);

    int warps_in_grid = gridDim_x(id) * meta_group_size(warp);
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = thread_rank(warp); i < C; i  += size(warp)) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(warp, dnorm_mean, sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(warp, dnorm_norm_mean, sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = thread_rank(warp); i < C; i += size(warp)) {
            // Fix this later
            // float dout_i = (float)__ldcs(&dout_bt[i]);
            // float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dout_i = (float)dout_bt[i];
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    sycl::group_barrier(block);

    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C * gridDim_x(id);
    uint* scratchFlag = (uint*)(scratch + (2 * C * gridDim_x(id)));

    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
        scratch_dbias[i + C*blockIdx_x(id)] = dbias_shared[i];
        scratch_dweight[i + C*blockIdx_x(id)] = dweight_shared[i];
    }
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
    sycl::group_barrier(block);
    if (threadIdx_x(id) == 0) {
        *tmp_flag = atomicAdd(scratchFlag, 1);
    }
    sycl::group_barrier(block);
    if (*tmp_flag == gridDim_x(id)-1) {
        // last block to finish, accumulate the scratchpad
        for (int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
            float dbias_sum = 0.0f;
            float dweight_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < gridDim_x(id); j++) {
                dbias_sum += scratch_dbias[i + j*C];
                dweight_sum += scratch_dweight[i + j*C];
            }
            dbias[i] = (Tparams)((float)dbias[i] + dbias_sum);
            dweight[i] = (Tparams)((float)dweight[i] + dweight_sum);
        }
    }
}

// single FP32 scratchpad shared by all the threadblocks (based on kernels 3 & 5)
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward_kernel6(sycl::nd_item<1> id, Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C + 1

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int base_idx = blockIdx_x(id) * meta_group_size(warp) + meta_group_rank(warp);

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = threadIdx_x(id); i < C; i  += blockDim_x(id)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    uint *tmp_flag = (uint*)(shared + C*2);
    sycl::group_barrier(block);  

    int warps_in_grid = gridDim_x(id) * meta_group_size(warp);
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = thread_rank(warp); i < C; i  += size(warp)) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(warp, dnorm_mean, sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(warp, dnorm_norm_mean, sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
            // Fix this later
            // float dout_i = (float)__ldcs(&dout_bt[i]);
            // float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dout_i = (float)dout_bt[i];
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }

    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    sycl::group_barrier(block);
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    uint* scratchFlag = (uint*)(scratch + (2 * C));
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
        atomicAdd(&scratch_dbias[i], dbias_shared[i]);
        atomicAdd(&scratch_dweight[i], dweight_shared[i]);
    }
    sycl::group_barrier(block);
    if (block.leader()) {
        *tmp_flag = atomicAdd(scratchFlag, 1);
    }
    sycl::group_barrier(block);
    if (*tmp_flag == gridDim_x(id)-1) {
        for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (Tparams)scratch_dbias[i];
            dweight[i] = (Tparams)scratch_dweight[i];
        }
    }
}

// Same as kernel 6 but without cooperative groups or templates
void layernorm_backward_kernel7(sycl::nd_item<1> id, floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C + 1
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int warpSize = size(warp);
    int warpId = threadIdx_x(id) / warpSize; // warp index within a block
    int warpsInBlock = blockDim_x(id) / warpSize;
    int base_idx = blockIdx_x(id) * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx_x(id) % warpSize; // Thread index within the warp
    int warps_in_grid = gridDim_x(id) * warpsInBlock;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    sycl::group_barrier(block);

    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx; i < C; i  += warpSize) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = warpReduceSum(warp, dnorm_mean);
        dnorm_norm_mean = warpReduceSum(warp, dnorm_norm_mean);

        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warpThreadIdx; i < C; i += warpSize) {
            // Fix this later
            float dout_i = (float)dout_bt[i];
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (floatX)((float)dinp_bt[i] + dval);
        }
    }

    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    sycl::group_barrier(block);
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
        atomicAdd(&scratch_dbias[i], dbias_shared[i]);
        atomicAdd(&scratch_dweight[i], dweight_shared[i]);
    }
    sycl::group_barrier(block);
    if (threadIdx_x(id) == 0) {
        *tmp_flag = atomicAdd(scratchFlag, 1);
    }
    sycl::group_barrier(block);
    if (*tmp_flag == gridDim_x(id)-1) {
        for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (floatX)scratch_dbias[i];
            dweight[i] = (floatX)scratch_dweight[i];
        }
    }
}

void layernorm_backward_kernel8(sycl::nd_item<1> id, floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                                            const floatX* dout, const floatX* inp, const floatX* weight,
                                            const floatX* mean, const floatX* rstd,
                                            int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C + 1
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int warpSize = size(warp);
    int warpId = threadIdx_x(id) / warpSize; // warp index within a block
    int warpsInBlock = blockDim_x(id) / warpSize; //number of warps in block
    int baseIdx = blockIdx_x(id) * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx_x(id) % warpSize; // Thread index within the warp
    int warpsInGrid = gridDim_x(id) * warpsInBlock;
    int C_per_iteration = warpSize * x128::size;
    int iterations_C = C / C_per_iteration;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    sycl::group_barrier(block);

    for (int idx = baseIdx; idx < B * T; idx += warpsInGrid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += warpSize * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float norm_bti = ((float)inp128_i[k] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
        }
        dnorm_mean = warpReduceSum(warp, dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(warp, dnorm_norm_mean) / C;

        // now iterate again and accumulate all the gradients
        // unfortunately we cannot use the same index for x128 arrays and shared memory
        // as atomics can only be 32-bit rather than 128-bit (at least pre-SM90/Hopper)
        // so this would result in an 8-way bank conflict, and kill performance
        // so instead, we use a shared memory friendly index, and reorder before the final write
        for (int i = 0; i < iterations_C; i++) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            x128 dout128   = load128cs(dout_bt + global_index);
            x128 inp128    = load128cs(inp_bt  + global_index);
            x128 dinp128   = load128(dinp_bt   + global_index);
            x128 weight128 = load128(weight    + global_index);

            for (int x = 0; x < x128::size; x++) {
                float dout_i = (float)dout128[x];
                float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128[x] * dout_i;
                // gradient contribution to bias (using shared memory friendly index)
                atomicAdd(&dbias_shared[shared_index + x*warpSize], dout_i);
                // gradient contribution to weight (using shared memory friendly index)
                atomicAdd(&dweight_shared[shared_index + x*warpSize], norm_bti * dout_i);
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp128[x] = (floatX)((float)dinp128[x] + dval);
            }
            // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
            store128cg(dinp_bt + global_index, dinp128);
        }
    }
    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    // todo - could potentially avoid the extra copy if floatX is FP32, fairly negligible though
    sycl::group_barrier(block);
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for(int i = threadIdx_x(id); i < C; i+= blockDim_x(id)) {
        // global atomics in the same "shared memory banking friendly" order
        atomicAdd(&scratch_dbias[i], dbias_shared[i]);
        atomicAdd(&scratch_dweight[i], dweight_shared[i]);
    }
    sycl::group_barrier(block);
    if (threadIdx_x(id) == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim_x(id));
    }
    sycl::group_barrier(block);
    if (*tmp_flag == gridDim_x(id)-1) {
        for (int i = warpId; i < iterations_C; i += warpsInBlock) {
            // reorder from atomic/shared memory-friendly index to real global memory index
            // and convert from float/FP32 to floatX/BF16 for the final write
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int x = 0; x < x128::size; x++) {
                float s_db = scratch_dbias[shared_index + x*warpSize];
                float s_dw = scratch_dweight[shared_index + x*warpSize];
                dbias128[x] = (floatX)(s_db + (float)dbias128[x]);
                dweight128[x] = (floatX)(s_dw + (float)dweight128[x]);
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

void layernorm_backward_kernel9(sycl::nd_item<1> id, floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                                            const floatX* dout, const floatX* inp, const floatX* weight,
                                            const floatX* mean, const floatX* rstd,
                                            int B, int T, int C, sycl::local_accessor<float> lmem) {
    //constexpr int WARP_SIZE = 8;
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int WARP_SIZE = size(warp);
    int BLOCK_SIZE = blockDim_x(id);
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C + 1

    int warpId = threadIdx_x(id) / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx_x(id) * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx_x(id) % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim_x(id) * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = ceil_div(C, C_per_iteration) + 2;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;
    float* dbias_tmp_shared = shared + 2 * C;
    float* dweight_tmp_shared = shared + 2 * C + BLOCK_SIZE;

    // init shared memory to zero
    for(int i = threadIdx_x(id); i < C; i+= BLOCK_SIZE){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*C + 2*BLOCK_SIZE);
    sycl::group_barrier(block);

    for (int idx = baseIdx; idx < B * T; idx += warpsInGrid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float norm_bti = ((float)inp128_i[k] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
        }
        dnorm_mean = warpReduceSum(warp, dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(warp, dnorm_norm_mean) / C;

        // now iterate again and accumulate all the gradients
        // unfortunately we cannot use the same index for x128 arrays and shared memory
        // as atomics can only be 32-bit rather than 128-bit (at least pre-SM90/Hopper)
        // so this would result in an 8-way bank conflict, and kill performance
        // so instead, we use a shared memory friendly index, and reorder before the final write
        for (int i = 0; i < iterations_C; i++) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dout128   = load128cs(dout_bt + global_index);
            x128 inp128    = load128cs(inp_bt  + global_index);
            x128 dinp128   = load128(dinp_bt   + global_index);
            x128 weight128 = load128(weight    + global_index);

            for (int x = 0; x < x128::size; x++) {
                float dout_i = (float)dout128[x];
                float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128[x] * dout_i;

                // sum up the gradients for bias and weight across the entire block
                // this is basically a reduction (but only inter-warp, not intra-warp)
                // doing it this way allows us to avoid using atomics while using many warps
                if (warpId != 0) {
                    dbias_tmp_shared[threadIdx_x(id)] = dout_i;
                    dweight_tmp_shared[threadIdx_x(id)] = norm_bti * dout_i;
                }
                sycl::group_barrier(block);
                if (warpId == 0) {
                    float dbias_tmp = dout_i;
                    float dweight_tmp = norm_bti * dout_i;
                    for (int j = 1; j < warpsInBlock; j++) {
                        dbias_tmp += dbias_tmp_shared[threadIdx_x(id) + j * WARP_SIZE];
                        dweight_tmp += dweight_tmp_shared[threadIdx_x(id) + j * WARP_SIZE];
                    }
                    // gradient contribution to bias (using shared memory friendly index)
                    dbias_shared[shared_index + x*WARP_SIZE] += dbias_tmp;
                    // gradient contribution to weight (using shared memory friendly index)
                    dweight_shared[shared_index + x*WARP_SIZE] += dweight_tmp;
                }
                sycl::group_barrier(block);

                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp128[x] = (floatX)((float)dinp128[x] + dval);
            }
            // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
            store128cg(dinp_bt + global_index, dinp128);
        }
    }
    sycl::group_barrier(block);
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for(int i = threadIdx_x(id); i < C; i+= BLOCK_SIZE) {
        // Write to global memory in the same "shared memory banking friendly" order
        scratch_dbias[i + 2*C*blockIdx_x(id)] = dbias_shared[i];
        scratch_dweight[i + 2*C*blockIdx_x(id)] = dweight_shared[i];
    }
    sycl::group_barrier(block);
    if (threadIdx_x(id) == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim_x(id));
    }
    sycl::group_barrier(block);
    if (*tmp_flag == gridDim_x(id)-1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for(int i = threadIdx_x(id) * f128::size; i < C; i+= BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim_x(id); read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        sycl::group_barrier(block);

        // reorder from atomic/shared memory-friendly index to real global memory index
        // and convert from float/FP32 to floatX/BF16 for the final write
        // this is separate also because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int i = warpId; i < iterations_C; i += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int x = 0; x < x128::size; x++) {
                float s_db = dbias_shared[shared_index + x*WARP_SIZE];
                float s_dw = dweight_shared[shared_index + x*WARP_SIZE];
                dbias128[x] = (floatX)(s_db + (float)dbias128[x]);
                dweight128[x] = (floatX)(s_dw + (float)dweight128[x]);
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}


// similar to kernel 9, but uses vectors to access shared memory, which also avoids the bank conflict problems,
// and makes use require fewer barriers, at the cost of increased shared memory consumption.
// warning: this kernel is _extremely_ close to getting register spills, so many "optimizations" turn out to be unhelpful
// or need to be implemented in a very specific way.
void layernorm_backward_kernel10(sycl::nd_item<1> id, floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                            const floatX* dout, const floatX* inp, const floatX* weight,
                            const floatX* mean, const floatX* rstd,
                            int B, int T, int C, sycl::local_accessor<float> lmem) {
    //constexpr int WARP_SIZE = 8;
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int WARP_SIZE = size(warp);

    int BLOCK_SIZE = blockDim_x(id);
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C + 1

    int warpId = threadIdx_x(id) / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx_x(id) * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx_x(id) % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim_x(id) * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = ceil_div(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = ceil_div(C, (WARP16 * x128::size)) * (WARP16 * x128::size);
    float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for(int i = threadIdx_x(id) * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    sycl::group_barrier(block);

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt = inp +bt * C;
        floatX* dinp_bt = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = (float)mean[bt];
        const float rstd_bt = (float)rstd[bt];
        dnorm_mean = warpReduceSum(warp, dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(warp, dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if(global_index < C) {
                dout128 = load128cs(dout_bt + global_index);
                inp128 = load128cs(inp_bt + global_index);
                dinp128 = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    dbias_f[i] = dout_i;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float) weight128[x] * (float)dout128[x]; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp128[x] = (floatX) ((float) dinp128[x] + dval);
                }

                if (warpId != 0) {
                    store128(dbias_tmp_shared + threadIdx_x(id) * f128::size, dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared + threadIdx_x(id) * f128::size, dweight_f);
                }
                sycl::group_barrier(block);
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp = load128(dbias_tmp_shared + f128::size * (threadIdx_x(id) + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx_x(id) + j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                sycl::group_barrier(block);
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if(global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    sycl::group_barrier(block);
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32; // this may be different?
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for(int i = threadIdx_x(id) * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2*C*blockIdx_x(id), load128(dbias_shared + i));
        store128(scratch_dweight + i + 2*C*blockIdx_x(id), load128(dweight_shared + i));
    }
    sycl::group_barrier(block);
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);
    if (threadIdx_x(id) == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim_x(id));
    }
    sycl::group_barrier(block);
    if (*tmp_flag == gridDim_x(id)-1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for(int i = threadIdx_x(id) * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim_x(id); read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        sycl::group_barrier(block);

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void layernorm_backward1(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C, const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_backward_kernel1(id, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward2(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(32*N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    float* dweight_tmp;
    float* dbias_tmp;
    syclMallocCheck(dweight_tmp = sycl::malloc_device<float>(C, *DefaultQueue));
    syclMallocCheck(dbias_tmp = sycl::malloc_device<float>(C, *DefaultQueue));
    DefaultQueue->memset(dweight_tmp, 0, C * sizeof(float));
    DefaultQueue->memset(dbias_tmp, 0, C * sizeof(float));
    DefaultQueue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> lmem(shared_mem_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) [[sycl::reqd_sub_group_size(32)]] {
            layernorm_backward_kernel2(id, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, dweight_tmp, dbias_tmp, lmem);
        });
    });
    DefaultQueue->parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> id) {
        copy_to_dweight_dbias(id, C, dweight, dbias, dweight_tmp, dbias_tmp);
    });
    DefaultQueue->wait();
    sycl::free(dweight_tmp, *DefaultQueue);
    sycl::free(dbias_tmp, *DefaultQueue);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward3(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
    const int grid_size = (1024/block_size) * get_num_CUs();
    size_t shared_mem_size = 2 * C * sizeof(float);
    DefaultQueue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> lmem(shared_mem_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            layernorm_backward_kernel3(id, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, lmem);
        });
    });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward4(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * get_num_CUs();
        size_t shared_mem_size = 2 * C * sizeof(float);
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(shared_mem_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
                layernorm_backward_kernel4(id, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, lmem);
            });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward5(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = 1 * get_num_CUs(); // only support 1 block per SM for simplicity, 1024 threads is best anyway
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);
        DefaultQueue->memset(scratch, 0, (grid_size * 2 * C + 1) * sizeof(float));
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(shared_mem_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
                layernorm_backward_kernel5(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, lmem);
            });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward6(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * get_num_CUs();
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        DefaultQueue->memset(scratch, 0, (1 + 2 * C) * sizeof(float));
        
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(shared_mem_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
                layernorm_backward_kernel6(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, lmem);
            });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward7(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * get_num_CUs();
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        DefaultQueue->memset(scratch, 0, (1 + 2 * C) * sizeof(float));

        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(shared_mem_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
                layernorm_backward_kernel7(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, lmem);
            });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward8(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * get_num_CUs();
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        DefaultQueue->memset(scratch, 0, (1 + 2 * C) * sizeof(float));

        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(shared_mem_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
                layernorm_backward_kernel8(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, lmem);
            });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward9(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {

        assert(C % (32 * x128::size) == 0  && "Channels must be divisible by (32 * x128::size)");
        const int grid_size = (1024/block_size) * get_num_CUs(); // todo - heuristics for other GPUs?
        size_t shared_mem_size = (2 * C + 2 * block_size + 1) * sizeof(float);

        DefaultQueue->memset(scratch, 0, 1 * sizeof(float));
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(shared_mem_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
                layernorm_backward_kernel9(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, lmem);
            });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward10(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                         const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                         int B, int T, int C, int block_size) {
        if(block_size == 1024) {
            block_size = 512;
        }
        //assert(C % (32 * x128::size) == 0  && "Channels must be divisible by (32 * x128::size)");
        const int grid_size = (1024/block_size) * get_num_CUs(); // todo - heuristics for other GPUs?
        size_t rounded_C = ceil_div(C, (WARP16 * x128::size)) * (WARP16 * x128::size);
        size_t shared_mem_size = (2 * rounded_C + 2 * (block_size - WARP16) * f128::size) * sizeof(float);
        printf("shared_mem_size: %lu\n", shared_mem_size);
        printf("local mem max size %lu\n", DefaultQueue->get_device().get_info<sycl::info::device::local_mem_size>());

        DefaultQueue->memset(scratch, 0, 1 * sizeof(float));
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(shared_mem_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) [[sycl::reqd_sub_group_size(WARP16)]] {
                layernorm_backward_kernel10(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, lmem);
            });
        });
}

// kernel version dispatch
void layernorm_backward(int kernel_num,
                        floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C,
                        const int block_size) {
    switch (kernel_num) {
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        case 1:
            layernorm_backward1(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#endif
        case 2:
            layernorm_backward2(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 3:
            layernorm_backward3(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#if defined(ENABLE_BF16)
        case 4:
            layernorm_backward4(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#endif
        case 5:
            layernorm_backward5(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 6:
            layernorm_backward6(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 7:
            layernorm_backward7(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 8:
            layernorm_backward8(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 9:
            layernorm_backward9(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 10:
            layernorm_backward10(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
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
    int C = 1600;

    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }
    DefaultQueue = &defaultQueue;

    // first do the forward pass in CPU
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    float *dout = make_random_float(B * T * C);
    float *dinp = make_zeros_float(B * T * C);
    float *dweight = make_zeros_float(C);
    float *dbias = make_zeros_float(C);
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move all the variables we need for backward pass onto the GPU
    floatX* d_dinp;
    floatX* d_dweight;
    floatX* d_dbias;
    floatX* d_dout;
    floatX* d_inp;
    floatX* d_weight;
    floatX* d_mean;
    floatX* d_rstd;
    float* d_scratch;
    syclMallocCheck(d_dinp = sycl::malloc_device<floatX>(B * T * C, defaultQueue));
    syclMallocCheck(d_dweight = sycl::malloc_device<floatX>(C, defaultQueue));
    syclMallocCheck(d_dbias = sycl::malloc_device<floatX>(C, defaultQueue));
    syclMallocCheck(d_dout = sycl::malloc_device<floatX>(B * T * C, defaultQueue));
    syclMallocCheck(d_inp = sycl::malloc_device<floatX>(B * T * C, defaultQueue));
    syclMallocCheck(d_weight = sycl::malloc_device<floatX>(C, defaultQueue));
    syclMallocCheck(d_mean = sycl::malloc_device<floatX>(B * T, defaultQueue));
    syclMallocCheck(d_rstd = sycl::malloc_device<floatX>(B * T, defaultQueue));
    syclMallocCheck(d_scratch = sycl::malloc_device<float>((1024/32) * get_num_CUs() * (2 * C + 1), defaultQueue));
    // copy over the "inputs" to the backward call
    memcpy_convert(d_dout, dout, B * T * C);
    memcpy_convert(d_inp, inp, B * T * C);
    memcpy_convert(d_weight, weight, C);
    memcpy_convert(d_mean, mean, B * T);
    memcpy_convert(d_rstd, rstd, B * T);


    // launch the kernel
    // removed 768 because it doesn't work for kernel9 despite being OK in train_gpt2.cu?!
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        // init the "outputs" of the backward call to zeros
        DefaultQueue->memset(d_dinp, 0, B * T * C * sizeof(floatX));
        DefaultQueue->memset(d_dweight, 0, C * sizeof(floatX));
        DefaultQueue->memset(d_dbias, 0, C * sizeof(floatX));

        layernorm_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd,
                           B, T, C, block_size);

        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f; // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 5e-1f; // much, much larger...
        printf("Checking correctness...\n");
        printf("dinp:\n");
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        printf("dweight:\n");
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
        printf("dbias:\n");
        validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);

        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now time the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_backward, kernel_num,
                                              d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd,
                                              B, T, C, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
    sycl::free(d_dinp, defaultQueue);
    sycl::free(d_dweight, defaultQueue);
    sycl::free(d_dbias, defaultQueue);
    sycl::free(d_dout, defaultQueue);
    sycl::free(d_inp, defaultQueue);
    sycl::free(d_weight, defaultQueue);
    sycl::free(d_mean, defaultQueue);
    sycl::free(d_rstd, defaultQueue);
    sycl::free(d_scratch, defaultQueue);
    return 0;
}
