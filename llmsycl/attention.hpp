/*
Attention, as a fallback when we do not use the Flash Attention from cuDNN
*/
#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// CUDA kernels

// inputs floatX, outputs FP32 (for current FP32-only activation path for this WIP)
void permute_kernel(sycl::nd_item<1> id, floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx_x(id) * blockDim_x(id) + threadIdx_x(id);
    if (idx >= B * NH * N * d) { return; }

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

void permute_kernel_backward(sycl::nd_item<1> id, floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx_x(id) * blockDim_x(id) + threadIdx_x(id);
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

void unpermute_kernel(sycl::nd_item<1> id, floatX* inp, floatX *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

    int idx = (blockIdx_x(id) * blockDim_x(id) + threadIdx_x(id));
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = __ldcs(&inp[idx]);
}

void unpermute_kernel_backward(sycl::nd_item<1> id, floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    int idx = blockIdx_x(id) * blockDim_x(id) + threadIdx_x(id);
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] = (floatX)dout[other_idx];
}

void softmax_forward_kernel5(sycl::nd_item<1> id, floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    sycl::sub_group warp = id.get_sub_group();
    int warp_size = size(warp);
    // Maybe change this back to a const later?
    int lane_id = threadIdx_x(id) % warp_size;
    int warp_id = threadIdx_x(id) / warp_size;
    int num_warps = blockDim_x(id) / warp_size;

    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim_x(id) - blockIdx_x(id) - 1) * num_warps + warp_id; // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    const float flt_max = 340282346638528859811704183484516925440.0f; // to avoid including float.h
    float maxval = -flt_max;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += warp_size) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, regarray[k]);
        }
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += sycl::exp(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::exp(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(warp, maxval);
    sumval *= sycl::exp(inv_temperature * (maxval - global_maxval));

    float sum = warpReduceSum(warp, sumval);
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = lane_id; i <= own_pos; i += warp_size) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = sycl::exp(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}

void softmax_autoregressive_backward_inplace_kernel(sycl::nd_item<2> id, floatX* datt, const floatX* att,
                                                       int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();

    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx_x(id);
    int idx = blockIdx_y(id);

    att += idx * T * T;
    datt += idx * T * T;

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = datt + t * T;

        float local_sum = 0;
        for (int t2 = threadIdx_x(id); t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        local_sum = sycl::reduce_over_group(block, local_sum, sycl::plus<float>());

        for (int t3 = threadIdx_x(id); t3 < T; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            if(t3 <= t) {
                float acc = (float) __ldcs(att_bth + t3) * ((float) __ldcs(datt_bth + t3) - local_sum);
                __stcs(dpreatt_bth + t3, (floatX) (scale * acc));
            } else {
                // explicitly set non-causal elements to zero
                __stcs(dpreatt_bth + t3, (floatX)0.f);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, sycl::queue* stream) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const float alpha = 1.0f, beta = 0.0f;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    const int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, T, NH, HS);
    });

    floatX* preatt = inp;
   
    dnnl::memory::data_type elt_type = dnnl::memory::data_type::f32;
    switch (PRECISION_MODE) {
        case PrecisionMode::PRECISION_FP32:
            elt_type = dnnl::memory::data_type::f32;
            break;
        case PrecisionMode::PRECISION_FP16:
            elt_type = dnnl::memory::data_type::f16;
            break;
        case PrecisionMode::PRECISION_BF16:
            elt_type = dnnl::memory::data_type::bf16;
            break;
        default:
            printf("Unsupported precision mode\n");
            exit(EXIT_FAILURE);
    }

    // Setup engine and stream
    auto &engine = *DefaultEngine;
    // See if we need to do a map lookup for this later?
    auto &dnnstream = *DefaultStream;

    // Create memory descriptors
    auto q_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto k_md = dnnl::memory::desc({B * NH, HS, T}, elt_type, dnnl::memory::format_tag::acb);
    auto preatt_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto q_mem = dnnlsycl::make_memory(q_md, engine, dnnlsycl::memory_kind::usm, q);
    auto k_mem = dnnlsycl::make_memory(k_md, engine, dnnlsycl::memory_kind::usm, k);
    auto preatt_mem = dnnlsycl::make_memory(preatt_md, engine, dnnlsycl::memory_kind::usm, preatt);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, q_md, k_md, preatt_md);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(dnnstream, {
        {DNNL_ARG_SRC, q_mem},
        {DNNL_ARG_WEIGHTS, k_mem},
        {DNNL_ARG_DST, preatt_mem}
    });

    // multiply all elements of preatt elementwise by scale
    float scale = 1.f / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    stream->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel5(id, att, scale, preatt, B * NH, T);
    });


    // new approach: first cuBLAS another batched matmul
    floatX* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    auto att_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);  
    auto v_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto vaccum_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto att_mem = dnnlsycl::make_memory(att_md, engine, dnnlsycl::memory_kind::usm, att);
    auto v_mem = dnnlsycl::make_memory(v_md, engine, dnnlsycl::memory_kind::usm, v);
    auto vaccum_mem = dnnlsycl::make_memory(vaccum_md, engine, dnnlsycl::memory_kind::usm, vaccum);

    // Create primitive descriptor
    auto matmul_pd2 = dnnl::matmul::primitive_desc(engine, att_md, v_md, vaccum_md);

    // Create primitive
    auto matmul_prim2 = dnnl::matmul(matmul_pd2);
 
    // Set arguments and execute
    matmul_prim2.execute(dnnstream, {
        {DNNL_ARG_SRC, att_mem},
        {DNNL_ARG_WEIGHTS, v_mem},
        {DNNL_ARG_DST, vaccum_mem}
    });

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel(id, vaccum, out, B, T, NH, HS);
    });
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH, sycl::queue* stream) {
    const int block_size = 256;
    const int HS = C / NH; // head size
    const float alpha = 1.0f, beta = 0.0f;
    dnnl::memory::data_type elt_type = dnnl::memory::data_type::f32;
    switch (PRECISION_MODE) {
        case PrecisionMode::PRECISION_FP32:
            elt_type = dnnl::memory::data_type::f32;
            break;
        case PrecisionMode::PRECISION_FP16:
            elt_type = dnnl::memory::data_type::f16;
            break;
        case PrecisionMode::PRECISION_BF16:
            elt_type = dnnl::memory::data_type::bf16;
            break;
        default:
            printf("Unsupported precision mode\n");
            exit(EXIT_FAILURE);
    }

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel_backward(id, scratch, dout, B, T, NH, HS);
    });
    // backward into datt
    // Batched matrix multiply with oneDNN
    // Setup engine and stream
    auto &engine = *DefaultEngine;
    // Maybe change this?
    auto &dnnstream = *DefaultStream;

    // Create memory descriptors
    auto scratch_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto v_md = dnnl::memory::desc({B * NH, HS, T}, elt_type, dnnl::memory::format_tag::acb);
    auto datt_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto scratch_mem = dnnlsycl::make_memory(scratch_md, engine, dnnlsycl::memory_kind::usm, scratch);
    auto v_mem = dnnlsycl::make_memory(v_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX *>(v));
    auto datt_mem = dnnlsycl::make_memory(datt_md, engine, dnnlsycl::memory_kind::usm, datt);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, scratch_md, v_md, datt_md);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(dnnstream, {
        {DNNL_ARG_SRC, scratch_mem},
        {DNNL_ARG_WEIGHTS, v_mem},
        {DNNL_ARG_DST, datt_mem}
    });
    // backward into dv
    // Create memory descriptors
    auto att_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::acb);
    // scratch_md is already defined
    auto dv_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto att_mem = dnnlsycl::make_memory(att_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX*>(att));
    // scratch_mem is already defined
    auto dv_mem = dnnlsycl::make_memory(dv_md, engine, dnnlsycl::memory_kind::usm, dv);

    // Create primitive descriptor
    auto matmul_pd2 = dnnl::matmul::primitive_desc(engine, att_md, scratch_md, dv_md);

    // Create primitive
    auto matmul_prim2 = dnnl::matmul(matmul_pd2);
 
    // Set arguments and execute
    matmul_prim2.execute(dnnstream, {
        {DNNL_ARG_SRC, att_mem},
        {DNNL_ARG_WEIGHTS, scratch_mem},
        {DNNL_ARG_DST, dv_mem}
    });

    const float scale = 1.0f / sqrtf((float)HS);
    // backward into preatt. this is an in-place operation; datt turns into dpreatt here
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(B * NH, (T / 4) * 256),
                                                 sycl::range<2>(1, 256)), [=](sycl::nd_item<2> id) {
        softmax_autoregressive_backward_inplace_kernel(id, datt, att, B, T, C, scale);
    });
    floatX* dpreatt = datt;
    // backward into q
    auto dpreatt_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);  
    auto k_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto dq_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto dpreatt_mem = dnnlsycl::make_memory(dpreatt_md, engine, dnnlsycl::memory_kind::usm, dpreatt);
    auto k_mem = dnnlsycl::make_memory(k_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX*>(k));
    auto dq_mem = dnnlsycl::make_memory(dq_md, engine, dnnlsycl::memory_kind::usm, dq);

    // Create primitive descriptor
    auto matmul_pd3 = dnnl::matmul::primitive_desc(engine, dpreatt_md, k_md, dq_md);

    // Create primitive
    auto matmul_prim3 = dnnl::matmul(matmul_pd3);
 
    // Set arguments and execute
    matmul_prim3.execute(dnnstream, {
        {DNNL_ARG_SRC, dpreatt_mem},
        {DNNL_ARG_WEIGHTS, k_mem},
        {DNNL_ARG_DST, dq_mem}
    });

    // backward into k
    auto dpreatt_md2 = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::acb);
    auto q_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto dk_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto dpreatt_mem2 = dnnlsycl::make_memory(dpreatt_md2, engine, dnnlsycl::memory_kind::usm, dpreatt);
    auto q_mem = dnnlsycl::make_memory(q_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX*>(q));
    auto dk_mem = dnnlsycl::make_memory(dk_md, engine, dnnlsycl::memory_kind::usm, dk);

    // Create primitive descriptor
    auto matmul_pd4 = dnnl::matmul::primitive_desc(engine, dpreatt_md2, q_md, dk_md);

    // Create primitive
    auto matmul_prim4 = dnnl::matmul(matmul_pd4);
 
    // Set arguments and execute
    matmul_prim4.execute(dnnstream, {
        {DNNL_ARG_SRC, dpreatt_mem2},
        {DNNL_ARG_WEIGHTS, q_mem},
        {DNNL_ARG_DST, dk_mem}
    });
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel_backward(id, dinp, dq, dk, dv, B, T, NH, HS);
    });
}
