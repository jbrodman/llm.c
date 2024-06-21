/*
AdamW kernel
*/

// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
inline float lerp(float start, float end, float weight) {
    return sycl::mad(weight, end, sycl::mad(-weight, start, start));
}

template <typename Tp, typename Tg>
void adamw_kernel3(sycl::nd_item<1> id, Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    int idx = id.get_global_id(0);
    if (idx >= num_parameters) { return; }  // guard

    // get the gradient, m, and v for this parameter
    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    // fetch the old value of this parameter as a float, from either source
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    // update this parameter
    float param = old_param - (learning_rate * (m / (sycl::sqrt(v) + eps) + weight_decay * old_param));
    // update our low precision version of the parameters using stochastic rounding
    // this will be used in the next forward pass
    // TODO: simply doing `params_memory[i] = (floatX)param;` breaks everything (why?)
    unsigned int random = Get2dNoiseUint(threadIdx_x(id), blockIdx_x(id), seed);
    stochastic_rounding(id, param, &params_memory[idx], random);
    // write the full, float version of the param into our master copy, if we maintain one
    // this will be used in the next update
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, sycl::queue* stream) {
    // AdamW update
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    stream->parallel_for(sycl::nd_range<1>(num_blocks*block_size, block_size), [=](sycl::nd_item<1> id) {
        adamw_kernel3(id, params_memory, master_params_memory, grads_memory, m_memory, v_memory, num_parameters,
                      learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                      grad_scale, seed);
    });
}