/*
Kernels for the AdamW optimizer.

References:
  * https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  * https://github.com/nvidia/apex/blob/master/csrc/multi_tensor_adam.cu

Compile example:
icpx -O3 -fsycl adamw.cpp -o adamw

./adamw

TODO(general):
amsgrad=True

TODO(perf):
dtype
thread coarsening/ILP

TODO(sycl):
Investigate fp precision issues

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sycl/sycl.hpp>
#include "common.h"


// ----------------------------------------------------------------------------
// CPU code reference

void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // adapted from: train_gpt2.c

    for (int i = 0; i < num_parameters; i++) {
        float param = params_memory[i];
        float grad = grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// utility functions

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
inline float lerp(float start, float end, float weight) {
    return sycl::mad(weight, end, sycl::mad(-weight, start, start));
}

// naive fused kernel
void adamw_kernel1(sycl::nd_item<1> id, float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                   float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = id.get_global_id(0);
   if (i >= num_parameters) return;  // guard
   // update the first moment (momentum)
   m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];
   // update the second moment (RMSprop)
   v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];
   float m_hat = m_memory[i] / beta1_correction;
   float v_hat = v_memory[i] / beta2_correction;
   params_memory[i] -= learning_rate * (m_hat / (sycl::sqrt(v_hat) + eps) + weight_decay * params_memory[i]);
}

// Slightly more optimized AdamW kernel by:
// * loading data that is accessed more than once into registers,
// * using optimized linear interpolation for the moment updates.
void adamw_kernel2(sycl::nd_item<1> id, float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                   float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = id.get_global_id(0);
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sycl::sqrt(v) + eps) + weight_decay * params_memory[i]);
}


// ----------------------------------------------------------------------------
// kernel launcher

// version 1: naive dispatch to naive kernel
void adamw_dispatch1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        adamw_kernel1(id, params_memory, grads_memory, m_memory, v_memory, num_parameters,
                      learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    });
}

// version 2: naive dispatch to slightly optimized kernel
void adamw_dispatch2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        adamw_kernel2(id, params_memory, grads_memory, m_memory, v_memory, num_parameters,
                      learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    });
}

void adamw(int kernel_num,
           float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters,
           float learning_rate=1e-3f, float beta1=0.9f, float beta2=0.999f, float eps=1e-8f, float weight_decay=0.0f) {
    // calculate the m_hat and v_hat correction terms once as they are the same for every param/thread
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    switch (kernel_num) {
        case 1:
            adamw_dispatch1(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        case 2:
            adamw_dispatch2(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    const long num_parameters = 1048576;
    const int t = 10;

    const float learning_rate = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;

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

    // create random data on host (to be used for the CPU reference implementation)
    float* params_memory = make_random_float(num_parameters);
    float* grads_memory = make_random_float(num_parameters);
    float* m_memory = make_random_float_01(num_parameters);
    float* v_memory = make_random_float_01(num_parameters);

    // move to GPU
    float* d_params_memory;
    float* d_grads_memory;
    float* d_m_memory;
    float* d_v_memory;

    syclMallocCheck(d_params_memory = sycl::malloc_device<float>(num_parameters, defaultQueue));
    syclMallocCheck(d_grads_memory = sycl::malloc_device<float>(num_parameters, defaultQueue));
    syclMallocCheck(d_m_memory = sycl::malloc_device<float>(num_parameters, defaultQueue));
    syclMallocCheck(d_v_memory = sycl::malloc_device<float>(num_parameters, defaultQueue));

    defaultQueue.memcpy(d_params_memory, params_memory, num_parameters * sizeof(float));
    defaultQueue.memcpy(d_grads_memory, grads_memory, num_parameters * sizeof(float));
    defaultQueue.memcpy(d_m_memory, m_memory, num_parameters * sizeof(float));
    defaultQueue.memcpy(d_v_memory, v_memory, num_parameters * sizeof(float));

    // Make sure the CPU waits for the memcpys to finish.
    defaultQueue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // calculate the CPU reference (using default hyperparams)
    clock_t start = clock();
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_parameters);
    clock_t end = clock();
    // TODO: measure runtime with multiple runs
    double elapsed_time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // calculate the GPU version (using default hyperparams)
    adamw(kernel_num, d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters);

    // compare
    printf("Checking correctness...\n");
    printf("parameters:\n");
    validate_result(d_params_memory, params_memory, "params_memory", num_parameters);
    printf("first moment:\n");
    validate_result(d_m_memory, m_memory, "m_memory", num_parameters);
    printf("second moment:\n");
    validate_result(d_v_memory, v_memory, "v_memory", num_parameters);
    printf("All results match.\n\n");

    // now benchmark the kernel
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(repeat_times, adamw, kernel_num,
      d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters,
      learning_rate, beta1, beta2, eps, weight_decay);
    printf("time gpu %.4f ms\n", elapsed_time);
    printf("time cpu %.4f ms\n", elapsed_time_cpu);

    // cleanup
    free(params_memory);
    free(grads_memory);
    free(m_memory);
    free(v_memory);

    sycl::free(d_params_memory, defaultQueue);
    sycl::free(d_grads_memory, defaultQueue);
    sycl::free(d_m_memory, defaultQueue);
    sycl::free(d_v_memory, defaultQueue);

    return 0;
}
