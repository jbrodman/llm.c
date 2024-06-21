/*
Kernels for matmul backward pass.

Compile example:
nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward -lcublas

OMP_NUM_THREADS=32 ./matmul_backward 1
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

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != NULL){dbias[o] = sum;}
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel to backpropagate only the bias, it's just a sum :'(
void matmul_backward_bias_kernel_naive(sycl::nd_item<1> id, float* dbias, const float* dout, int B, int T, int OC) {
    int o = id.get_global_id(0);
    if (o < OC) {
        // nix the double
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                sum += dout[b * T * OC + t * OC + o];
            }
        }
        dbias[o] = sum;
    }
}

// use shared memory and coarsening + reductions
void matmul_backward_bias_kernel_faster(sycl::nd_item<1> id, float* dbias, const float* dout, int B, int T, int OC) {
    int o = blockIdx_x(id); // range [0, OC)
    int tid = threadIdx_x(id); // range [0, block_size)
    int block_size = blockDim_x(id);
    const float* x = dout + o;
    // thread coarsening
    // nix the double
    float sum = 0.0f;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    float group_sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<>());
   
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = group_sum;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward1(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC) {
    sycl::queue& queue = *DefaultQueue;

    float alpha = 1.0f;
    float beta = 1.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // for reference the API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)

    // recall the forward pass was calculated with alpha = 1.0f, beta = 0.0f as:
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);

    // backward to input
    //cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &alpha, weight, C, dout, OC, &beta, dinp, C));
    /*
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC));
    */
    // Setup engine and stream
    auto engine = dnnlsycl::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnlsycl::make_stream(engine, queue);

    // Create memory descriptors
    auto dout_md = dnnl::memory::desc({B*T, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({OC, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto dinp_md = dnnl::memory::desc({B*T, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto dout_mem = dnnlsycl::make_memory(dout_md, engine, dnnlsycl::memory_kind::usm, dout);
    auto weight_mem = dnnlsycl::make_memory(weight_md, engine, dnnlsycl::memory_kind::usm, weight);
    auto dinp_mem = dnnlsycl::make_memory(dinp_md, engine, dnnlsycl::memory_kind::usm, dinp);

    dnnl::post_ops po;
    po.append_sum(beta);

    dnnl::primitive_attr matmul_attr;
    matmul_attr.set_post_ops(po);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, dout_md, weight_md, dinp_md, matmul_attr);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC, dout_mem},
        {DNNL_ARG_WEIGHTS, weight_mem},
        {DNNL_ARG_DST, dinp_mem}
    });

    // backward to weight
    //cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &alpha, inp, C, dout, OC, &beta, dweight, C));
    // Create memory descriptors
    auto dout_md2 = dnnl::memory::desc({OC, B*T}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba);
    auto inp_md = dnnl::memory::desc({B*T, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto dweight_md = dnnl::memory::desc({OC, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto dout_mem2 = dnnlsycl::make_memory(dout_md2, engine, dnnlsycl::memory_kind::usm, dout);
    auto inp_mem = dnnlsycl::make_memory(inp_md, engine, dnnlsycl::memory_kind::usm, inp);
    auto dweight_mem = dnnlsycl::make_memory(dweight_md, engine, dnnlsycl::memory_kind::usm, dweight);

    // Create primitive descriptor
    auto matmul_pd2 = dnnl::matmul::primitive_desc(engine, dout_md2, inp_md, dweight_md, matmul_attr);

    // Create primitive 
    auto matmul_prim2 = dnnl::matmul(matmul_pd2);

    // Set arguments and execute
    matmul_prim2.execute(stream, {
        {DNNL_ARG_SRC, dout_mem2},
        {DNNL_ARG_WEIGHTS, inp_mem},
        {DNNL_ARG_DST, dweight_mem}
    });

    // backward to bias, if given
    if (dbias != NULL) {

        // sum over B,T using matrix vector multiplication with cuBLAS
        // for reference this API is:
        // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        //                    int m, int n,
        //                    const float           *alpha,
        //                    const float           *A, int lda,
        //                    const float           *x, int incx,
        //                    const float           *beta,
        //                    float           *y, int incy)
        // dout is (B,T,OC), or in 2D terms (B*T, OC)
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_N, B*T, OC, &alpha, dout, B*T, ones, 1, &beta, dbias, 1));
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, OC, B*T, &alpha, dout, OC, ones, 1, &beta, dbias, 1));

        // ugh the above isn't working...
        // let's just do naive calculation for now, fix later
        // const int block_size=128;
        // const int grid_size=(OC + block_size - 1) / block_size;
        // matmul_backward_bias_kernel<<<grid_size, block_size>>>(dbias, dout, B, T, OC);

        // bit faster
        const int block_size=512;
        const int block_dim = block_size;
        const int grid_dim = OC;
        queue.parallel_for(sycl::nd_range<1>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<1> id) {
            matmul_backward_bias_kernel_faster(id, dbias, dout, B, T, OC);
        });
    }
}

void matmul_backward(int kernel_num,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_backward1(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC);
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
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }
    DefaultQueue = &defaultQueue;

    // create host memory of random numbers
    float* dinp = make_zeros_float(B * T * C);
    float* dweight = make_zeros_float(OC * C);
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* ones = make_ones_float(OC);

    // move to GPU
    float* d_dinp;
    float* d_dweight;
    float* d_dbias;
    float* d_dout;
    float* d_inp;
    float* d_weight;
    float* d_ones;
    syclMallocCheck(d_dinp = sycl::malloc_device<float>(B * T * C, defaultQueue));
    syclMallocCheck(d_dweight = sycl::malloc_device<float>(OC * C, defaultQueue));
    syclMallocCheck(d_dbias = sycl::malloc_device<float>(OC, defaultQueue));
    syclMallocCheck(d_dout = sycl::malloc_device<float>(B * T * OC, defaultQueue));
    syclMallocCheck(d_inp = sycl::malloc_device<float>(B * T * C, defaultQueue));
    syclMallocCheck(d_weight = sycl::malloc_device<float>(OC * C, defaultQueue));
    syclMallocCheck(d_ones = sycl::malloc_device<float>(OC, defaultQueue));
    defaultQueue.memcpy(d_dinp, dinp, B * T * C * sizeof(float));
    defaultQueue.memcpy(d_dweight, dweight, OC * C * sizeof(float));
    defaultQueue.memcpy(d_dbias, dbias, OC * sizeof(float));
    defaultQueue.memcpy(d_dout, dout, B * T * OC * sizeof(float));
    defaultQueue.memcpy(d_inp, inp, B * T * C * sizeof(float));
    defaultQueue.memcpy(d_weight, weight, OC * C * sizeof(float));
    defaultQueue.memcpy(d_ones, ones, OC * sizeof(float));
    defaultQueue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // calculate the CPU reference
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);

    // calculate the GPU version
    matmul_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);

    // compare
    printf("Checking correctness...\n");
    printf("dinp:\n");
    validate_result(d_dinp, dinp, "dinp", B * T * C, 1e-3f);
    printf("dweight:\n");
    validate_result(d_dweight, dweight, "dweight", OC * C, 1e-3f);
    printf("dbias:\n");
    validate_result(d_dbias, dbias, "dbias", OC, 1e-3f);
    printf("All results match.\n\n");

    // now benchmark the kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_backward, kernel_num,
                                          d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                          B, T, C, OC);
    printf("time %.4f ms\n", elapsed_time);

    // cleanups
    free(dinp);
    free(dweight);
    free(dbias);
    free(dout);
    free(inp);
    free(weight);
    sycl::free(d_dinp, defaultQueue);
    sycl::free(d_dweight, defaultQueue);
    sycl::free(d_dbias, defaultQueue);
    sycl::free(d_dout, defaultQueue);
    sycl::free(d_inp, defaultQueue);
    sycl::free(d_weight, defaultQueue);

    return 0;
}