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

#include "common.h"

// turn on bf16 as default, done up here for now
#define ENABLE_BF16

namespace syclx = sycl::ext::oneapi;
 using bfloat162 = sycl::marray<syclx::bfloat16, 2>; 

#if defined(ENABLE_BF16)
typedef syclx::bfloat16 floatX;
typedef syclx::bfloat16 floatN;
#elif defined(ENABLE_FP16)
typedef sycl::half floatX;
typedef sycl::half floatN;
#else
typedef float floatX;
typedef float floatN;
#endif

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
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dbias_ref(dbias[i]);
        dbias_ref += dout_bt[i];
        // gradient contribution to weight
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dweight_ref(dweight[i]);
        dweight_ref += norm_bti * dout_bt[i];
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
                        int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
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
    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    sycl::group_barrier(block);

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.get_local_linear_id(); i < C; i  += warp.get_max_local_range()[0]) {
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
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        // gradient contribution to bias
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dbias_ref(dbias_shared[i]);
        dbias_ref += (float)dout_bt[i];                 
        // gradient contribution to weight
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dweight_ref(dweight_shared[i]);
        dweight_ref += norm_bti * (float)dout_bt[i];
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
    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
        atomicAddX(&dbias[i], (Tparams)dbias_shared[i]);
        atomicAddX(&dweight[i], (Tparams)dweight_shared[i]);
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
    int base_idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    sycl::group_barrier(block);

    int warps_in_grid = id.get_group_range(0) * warp.get_group_linear_range();
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
        for (int i = warp.get_local_linear_id(); i < C; i  += warp.get_max_local_range()[0]) {
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
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dbias_ref(dbias_shared[i]);
            dbias_ref += dout_i;
            // gradient contribution to weight
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dweight_ref(dweight_shared[i]);
            dweight_ref += norm_bti * dout_i;
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

    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
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
    int base_idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    sycl::group_barrier(block);

    int warps_in_grid = id.get_group_range(0) * warp.get_group_linear_range();
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
        for (int i = warp.get_local_linear_id(); i < C; i  += warp.get_max_local_range()[0]) {
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
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dbias_ref(dbias_shared[i]);
            dbias_ref += dout_i;
            // gradient contribution to weight
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dweight_ref(dweight_shared[i]);
            dweight_ref += norm_bti * dout_i;
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
    for(int i = id.get_local_linear_id(); i < C/2; i+= id.get_local_range(0)) {
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
        uint current_dbias32b = *reinterpret_cast<uint*>(&current_dbias);
        uint current_dweight32b = *reinterpret_cast<uint*>(&current_dweight);

        uint new_dbias32b = *reinterpret_cast<uint*>(&new_dbias);
        uint new_dweight32b = *reinterpret_cast<uint*>(&new_dweight);
        
        uint old_bias32b = current_dbias32b;
        uint old_weight32b = current_dweight32b;

        sycl::atomic_ref<uint, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> old_dbias32b_ref(*reinterpret_cast<uint*>(&dbiasVec2[i]));
        while (!old_dbias32b_ref.compare_exchange_weak(current_dbias32b, new_dbias32b)) {
            new_dbias = *reinterpret_cast<bfloat162*>(&current_dbias32b) + add_dbias;
            new_dbias32b = *reinterpret_cast<uint*>(&new_dbias);
        };
        sycl::atomic_ref<uint, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> old_dweight32b_ref(*reinterpret_cast<uint*>(&dweightVec2[i]));
        while (!old_dweight32b_ref.compare_exchange_weak(current_dweight32b, new_dweight32b)) {
            new_dweight = *reinterpret_cast<bfloat162*>(&current_dweight32b) + add_dweight;
            new_dweight32b = *reinterpret_cast<uint*>(&new_dweight);
        };
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
    int base_idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    uint *tmp_flag = (uint*)(shared + C*2);
    sycl::group_barrier(block);

    int warps_in_grid = id.get_group_range(0) * warp.get_group_linear_range();
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
        for (int i = warp.get_local_linear_id(); i < C; i  += warp.get_max_local_range()[0]) {
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
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dbias_ref(dbias_shared[i]);
            dbias_ref += dout_i;
            // gradient contribution to weight
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dweight_ref(dweight_shared[i]);
            dweight_ref += norm_bti * dout_i;
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
    float* scratch_dweight = scratch + C * id.get_group_range(0);
    uint* scratchFlag = (uint*)(scratch + (2 * C * id.get_group_range(0)));

    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
        scratch_dbias[i + C*id.get_group(0)] = dbias_shared[i];
        scratch_dweight[i + C*id.get_group(0)] = dweight_shared[i];
    }
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
    sycl::group_barrier(block);
    if (block.leader()) {
        sycl::atomic_ref<uint, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> flag_ref(*scratchFlag);
        *tmp_flag = flag_ref.fetch_add(1);
    }
    sycl::group_barrier(block);
    if (*tmp_flag == id.get_group_range(0)-1) {
        // last block to finish, accumulate the scratchpad
        for (int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
            float dbias_sum = 0.0f;
            float dweight_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < id.get_group_range(0); j++) {
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
    int base_idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = id.get_local_linear_id(); i < C; i  += id.get_local_range(0)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    uint *tmp_flag = (uint*)(shared + C*2);
    sycl::group_barrier(block);  

    int warps_in_grid = id.get_group_range(0) * warp.get_group_linear_range();
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
        for (int i = warp.get_local_linear_id(); i < C; i  += warp.get_max_local_range()[0]) {
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
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dbias_ref(dbias_shared[i]);
            dbias_ref += dout_i;
            // gradient contribution to weight
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dweight_ref(dweight_shared[i]);
            dweight_ref += norm_bti * dout_i;
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
    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dbias_ref(scratch_dbias[i]);
        dbias_ref += dbias_shared[i];
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dweight_ref(scratch_dweight[i]);
        dweight_ref += dweight_shared[i];
    }
    sycl::group_barrier(block);
    if (block.leader()) {
        sycl::atomic_ref<uint, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> flag_ref(*scratchFlag);
        *tmp_flag = flag_ref.fetch_add(1);
    }
    sycl::group_barrier(block);
    if (*tmp_flag == id.get_group_range(0)-1) {
        for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (Tparams)scratch_dbias[i];
            dweight[i] = (Tparams)scratch_dweight[i];
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
    DefaultQueue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> lmem(shared_mem_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            layernorm_backward_kernel2(id, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, lmem);
        });
    });
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

    // convert all the necessary cpu data to floatX (e.g. bfloat16)
    floatX* meanX = (floatX*)malloc(B * T * sizeof(floatX));
    floatX* rstdX = (floatX*)malloc(B * T * sizeof(floatX));
    floatX* doutX = (floatX*)malloc(B * T * C * sizeof(floatX));
    floatX* inpX = (floatX*)malloc(B * T * C * sizeof(floatX));
    floatX* weightX = (floatX*)malloc(C * sizeof(floatX));

    for (int i = 0; i < B * T; i++) {
        meanX[i] = (floatX)mean[i];
        rstdX[i] = (floatX)rstd[i];
    }
    for (int i = 0; i < B * T * C; i++) {
        doutX[i] = (floatX)dout[i];
        inpX[i] = (floatX)inp[i];
    }
    for (int i = 0; i < C; i++) {
        weightX[i] = (floatX)weight[i];
    }

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);
    printf("Number CUs = %d\n", get_num_CUs());

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
    syclMallocCheck(d_scratch = sycl::malloc_device<float>(get_num_CUs() * (2 * C + 1), defaultQueue));
    // copy over the "inputs" to the backward call
    defaultQueue.memcpy(d_dout, doutX, B * T * C * sizeof(floatX));
    defaultQueue.memcpy(d_inp, inpX, B * T * C * sizeof(floatX));
    defaultQueue.memcpy(d_weight, weightX, C * sizeof(floatX));
    defaultQueue.memcpy(d_mean, meanX, B * T * sizeof(floatX));
    defaultQueue.memcpy(d_rstd, rstdX, B * T * sizeof(floatX));
    // init the "outputs" of the backward call to zeros
    defaultQueue.memset(d_dinp, 0, B * T * C * sizeof(floatX));
    defaultQueue.memset(d_dweight, 0, C * sizeof(floatX));
    defaultQueue.memset(d_dbias, 0, C * sizeof(floatX));

    // launch the kernel
    const int block_size = 256;
    layernorm_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C, block_size);

    // check the correctness of the kernel
    float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f; // allow larger errors for BF16/FP16
    float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 20.0f; // much, much larger...
    printf("Checking correctness...\n");
    printf("dinp:\n");
    validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
    printf("dweight:\n");
    validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
    printf("dbias:\n");
    validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);

    // now time the kernel
    // Nix 1024 for Intel GPUs
    int block_sizes[] = {32, 64, 128, 256, 512};
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
    free(meanX);
    free(rstdX);
    free(doutX);
    free(inpX);
    free(weightX);
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
