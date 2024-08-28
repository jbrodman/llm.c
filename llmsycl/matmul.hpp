/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename OutFloat, bool UseAuxBuffer>
void matmul_backward_bias_kernel9(sycl::nd_item<3> id, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>, sycl::local_accessor<float> lmem) {
    sycl::sub_group warp = id.get_sub_group();
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;

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
    float (*sub_results)[WARP_SIZE][bdy] = (float (*)[WARP_SIZE][bdy])shared;
    auto Partition = syclex::get_fixed_size_group<4>(warp);

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        // These don't work - not sure why yet
        // v += shfl_down(warp, v, 1, 4);
        // v += shfl_down(warp, v, 2, 4);
        // It's just doing a reduction among groups of 4 elts anwyay...
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
            // v += shfl_down(warp, v, 1, 4);  
            // v += shfl_down(warp, v, 2, 4);
            v = sycl::reduce_over_group(Partition, v, sycl::plus<float>());
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx_y(id) * OC] = a;
            }
        }
    }
}

void reduce_add_sum_kernel(sycl::nd_item<1> id, floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx_x(id) * blockDim_x(id) + threadIdx_x(id)) * f128::size;
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
// kernel launchers

void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, sycl::queue* stream) {
    int has_bias = (bias != NULL);

    // Setup engine and stream
    auto &engine = *DefaultEngine;
    // may need to update this
    auto &dnnstream = *DefaultStream;
    
    // Create memory descriptors
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
    auto inp_md = dnnl::memory::desc({B*T, C}, elt_type, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({C, OC}, elt_type, dnnl::memory::format_tag::ba);
    auto out_md = dnnl::memory::desc({B*T, OC}, elt_type, dnnl::memory::format_tag::ab);
  
    // Create memory objects
    auto inp_mem = dnnlsycl::make_memory(inp_md, engine, dnnlsycl::memory_kind::usm, inp);
    auto weight_mem = dnnlsycl::make_memory(weight_md, engine, dnnlsycl::memory_kind::usm, weight);
    auto out_mem = dnnlsycl::make_memory(out_md, engine, dnnlsycl::memory_kind::usm, out);
        
    if (has_bias) {
        auto bias_md = dnnl::memory::desc({1, OC}, elt_type, dnnl::memory::format_tag::ab);
        auto bias_mem = dnnlsycl::make_memory(bias_md, engine, dnnlsycl::memory_kind::usm, bias);

        // Create primitive descriptor
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, bias_md, out_md);

        // Create primitive
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // Set arguments and execute
        matmul_prim.execute(dnnstream, {
            {DNNL_ARG_SRC, inp_mem},
            {DNNL_ARG_WEIGHTS, weight_mem},
            {DNNL_ARG_BIAS, bias_mem},
            {DNNL_ARG_DST, out_mem}
        });
    } else {
        // Create primitive descriptor
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, out_md);

        // Create primitive
        auto matmul_prim = dnnl::matmul(matmul_pd);
    
        // Set arguments and execute
        matmul_prim.execute(dnnstream, {
            {DNNL_ARG_SRC, inp_mem},
            {DNNL_ARG_WEIGHTS, weight_mem},
            {DNNL_ARG_DST, out_mem}
        });
    }
}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, sycl::queue* stream) {
    float one = 1.0f, zero = 0.0f;

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
       // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end
        const int block_size = 256;

        sycl::range<3> block_dim((unsigned)block_size/WARP_SIZE, 8, 4); 
        const int OC_per_warp = block_dim[1] * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        int size = stream->get_device().get_info<sycl::info::device::max_compute_units>();
        int warpSize = 32;
        size *= warpSize;
        const int grid_size_y = std::max(1, size / (block_size * grid_size_x)); // full GPU!

        sycl::range<3> grid_dim(1, grid_size_y, grid_size_x);

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            stream->submit([&](sycl::handler& h) {
                sycl::local_accessor<float> lmem(x128::size*32*8, h);
                h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) __SIMD32__ {
                    matmul_backward_bias_kernel9(id, dbias, dout, B, T, OC, std::bool_constant<false>{}, lmem);
                });
            });
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            stream->submit([&](sycl::handler& h) {
                sycl::local_accessor<float> lmem(x128::size*32*8, h);
                h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) __SIMD32__ {
                    matmul_backward_bias_kernel9(id, dbias_buffer, dout, B, T, OC, std::bool_constant<true>{}, lmem);
                });
            });
            stream->parallel_for(sycl::nd_range<1>(CEIL_DIV(OC, 256*f128::size)*256, 256), [=](sycl::nd_item<1> id) __SIMD32__ {
                reduce_add_sum_kernel(id, dbias, dbias_buffer, OC, grid_size_y);
            });
        }
    }

    // backward to input, uses = in the backward pass (set the gradient)
    // Setup engine and stream
    auto &engine = *DefaultEngine;
    // maybe change this later?
    auto &dnnstream = *DefaultStream;

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

    // Create memory descriptors
    auto dout_md = dnnl::memory::desc({B*T, OC}, elt_type, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({OC, C}, elt_type, dnnl::memory::format_tag::ab);
    auto dinp_md = dnnl::memory::desc({B*T, C}, elt_type, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto dout_mem = dnnlsycl::make_memory(dout_md, engine, dnnlsycl::memory_kind::usm, dout);
    auto weight_mem = dnnlsycl::make_memory(weight_md, engine, dnnlsycl::memory_kind::usm, weight);
    auto dinp_mem = dnnlsycl::make_memory(dinp_md, engine, dnnlsycl::memory_kind::usm, dinp);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, dout_md, weight_md, dinp_md);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(dnnstream, {
        {DNNL_ARG_SRC, dout_mem},
        {DNNL_ARG_WEIGHTS, weight_mem},
        {DNNL_ARG_DST, dinp_mem}
    });
    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    // Create memory descriptors
    auto dout_md2 = dnnl::memory::desc({OC, B*T}, elt_type, dnnl::memory::format_tag::ba);
    auto inp_md = dnnl::memory::desc({B*T, C}, elt_type, dnnl::memory::format_tag::ab);
    auto dweight_md = dnnl::memory::desc({OC, C}, elt_type, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto dout_mem2 = dnnlsycl::make_memory(dout_md2, engine, dnnlsycl::memory_kind::usm, dout);
    auto inp_mem = dnnlsycl::make_memory(inp_md, engine, dnnlsycl::memory_kind::usm, inp);
    auto dweight_mem = dnnlsycl::make_memory(dweight_md, engine, dnnlsycl::memory_kind::usm, dweight);

    dnnl::post_ops po;
    po.append_sum(one);

    dnnl::primitive_attr matmul_attr2;
    matmul_attr2.set_post_ops(po);

    // Create primitive descriptor
    auto matmul_pd2 = dnnl::matmul::primitive_desc(engine, dout_md2, inp_md, dweight_md, matmul_attr2);

    // Create primitive 
    auto matmul_prim2 = dnnl::matmul(matmul_pd2);

    // Set arguments and execute
    matmul_prim2.execute(dnnstream, {
        {DNNL_ARG_SRC, dout_mem2},
        {DNNL_ARG_WEIGHTS, inp_mem},
        {DNNL_ARG_DST, dweight_mem}
    });
}
