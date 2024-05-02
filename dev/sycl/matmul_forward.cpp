/*
Kernels for matmul forward pass.
It's advised to use OpenMP here because the CPU implementation is fairly slow otherwise

Compile example:
icpx -fsycl -fopenmp -O3 matmul_forward.cpp -o matmul_forward -ldnnl

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
OMP_NUM_THREADS=32 ./matmul_forward 1

version 2 calls cuBLAS, very fast
OMP_NUM_THREADS=32 ./matmul_forward 2

version 3 calls cuBLASLt, should be even faster
OMP_NUM_THREADS=32 ./matmul_forward 3
*/

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <omp.h>

#include "common.h"

// namespace alias
namespace dnnlsycl = dnnl::sycl_interop;

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1: naive kernel, every thread handles one output element, direct global memory access
void matmul_forward_kernel1(sycl::nd_item<2> id, float* out,
                            const float* inp, const float* weight, const float* bias,
                            int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = id.get_global_id(1);
    int oc = id.get_global_id(0);
    if (bt < BT && oc < OC) {
        int b = bt / BT;
        int t = bt % BT;
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc*C;
        const float* inp_bt = inp + b * BT * C + t * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val;
    }
}

// is there no better way other than just adding bias with a whole separate kernel?
// this is a highly memory-bound operation, should be fused into the matmul kernel
// but i can't seem to find a cuBLAS function that does this
void add_bias(sycl::nd_item<1> id, float* out, const float* bias, int B, int T, int OC) {
    int idx = id.get_global_id(0);
    int stride = id.get_global_range(0);
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(sycl::queue& queue, float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    sycl::nd_range<2> grid = sycl::nd_range<2>(sycl::range<2>(ceil_div(OC, sqrt_block_size) * sqrt_block_size, 
                                               ceil_div(B*T, sqrt_block_size) * sqrt_block_size),
                                   sycl::range<2>(sqrt_block_size, sqrt_block_size));
    queue.parallel_for(grid, [=](sycl::nd_item<2> id) {
        matmul_forward_kernel1(id, out, inp, weight, bias, B*T, C, OC);
    });
}

// kernel 2 calls cuBLAS, which should be very efficient
void matmul_forward2(sycl::queue& queue, float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // for reference API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // cuBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because cuBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
    // out.T = weight @ inp.T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out.T = weight.T @ inp
    // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call cuBLAS with A = weight, B = inp
    // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

    // Setup engine and stream
    auto engine = dnnlsycl::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnlsycl::make_stream(engine, queue);

    // Create memory descriptors
    auto inp_md = dnnl::memory::desc({B*T, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({C, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba);
    auto out_md = dnnl::memory::desc({B*T, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto inp_mem = dnnlsycl::make_memory(inp_md, engine, dnnlsycl::memory_kind::usm, const_cast<float *>(inp));
    auto weight_mem = dnnlsycl::make_memory(weight_md, engine, dnnlsycl::memory_kind::usm, const_cast<float *>(weight));
    auto out_mem = dnnlsycl::make_memory(out_md, engine, dnnlsycl::memory_kind::usm, out);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, out_md);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC, inp_mem},
        {DNNL_ARG_WEIGHTS, weight_mem},
        {DNNL_ARG_DST, out_mem}
    });

    // and now we still have to add the bias... (ew)
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);
        queue.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {    
            add_bias(id, out, bias, B, T, OC);
        });
    }
}

// uses cublasLt to fuse the bias and gelu
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
void matmul_forward3(sycl::queue& queue, float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);
    int has_gelu = 0;

    // Setup engine and stream
    auto engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(engine, queue);

    // Create memory descriptors
    auto inp_md = dnnl::memory::desc({B*T, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({C, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba);
    auto out_md = dnnl::memory::desc({B*T, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto bias_md = dnnl::memory::desc({1, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto inp_mem = dnnlsycl::make_memory(inp_md, engine, dnnlsycl::memory_kind::usm, const_cast<float *>(inp));
    auto weight_mem = dnnlsycl::make_memory(weight_md, engine, dnnlsycl::memory_kind::usm, const_cast<float *>(weight));
    auto out_mem = dnnlsycl::make_memory(out_md, engine, dnnlsycl::memory_kind::usm, out);
    auto bias_mem = dnnlsycl::make_memory(bias_md, engine, dnnlsycl::memory_kind::usm, const_cast<float *>(bias));

    // Create primitive attributes
    dnnl::primitive_attr matmul_attr;
    if (has_gelu) {
        dnnl::post_ops po;
        po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 1.0f, 0.0f);

        matmul_attr.set_post_ops(po);
    }

    // Create primitive descriptor
    dnnl::matmul::primitive_desc matmul_pd;
    if (has_bias) {
        matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, bias_md, out_md, matmul_attr);
    }
    else {
        matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, out_md, matmul_attr);
    }

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    if (has_bias) {
        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, inp_mem},
            {DNNL_ARG_WEIGHTS, weight_mem},
            {DNNL_ARG_BIAS, bias_mem},
            {DNNL_ARG_DST, out_mem}
        });
    }
    else {
        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, inp_mem},
            {DNNL_ARG_WEIGHTS, weight_mem},
            {DNNL_ARG_DST, out_mem}
        });
    }
}

// kernel version dispatch
void matmul_forward(sycl::queue& queue, int kernel_num,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(queue, out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 2:
            matmul_forward2(queue, out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 3:
            matmul_forward3(queue, out, inp, weight, bias, B, T, C, OC);
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
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    d_out = sycl::malloc_device<float>(B * T * OC, defaultQueue);
    d_inp = sycl::malloc_device<float>(B * T * C, defaultQueue);
    d_weight = sycl::malloc_device<float>(OC * C, defaultQueue);
    d_bias = sycl::malloc_device<float>(OC, defaultQueue);
   
    syclMallocCheck(d_out);
    syclMallocCheck(d_inp);
    syclMallocCheck(d_weight);
    syclMallocCheck(d_bias);
   
    defaultQueue.memcpy(d_inp, inp, B * T * C * sizeof(float));
    defaultQueue.memcpy(d_weight, weight, OC * C * sizeof(float));
    defaultQueue.memcpy(d_bias, bias, OC * sizeof(float));
   

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // time the kernel at different block sizes
    // Intel GPUs do not support 1024 block size
    int sqrt_block_sizes[] = {4, 8, 16};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        matmul_forward(defaultQueue, kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        validate_result(defaultQueue, d_out, out, "out", B * T * OC, 1e-1f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(defaultQueue, repeat_times, matmul_forward,
                                              kernel_num, d_out, d_inp, d_weight, d_bias,
                                              B, T, C, OC, sqrt_block_size);

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time, tflops);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);

    sycl::free(d_out, defaultQueue);
    sycl::free(d_inp, defaultQueue);
    sycl::free(d_weight, defaultQueue);
    sycl::free(d_bias, defaultQueue);
  
    return 0;
}