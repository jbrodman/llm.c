/*  Kernels for fused forward/backward classifier part
This fuses softmax, crossentropy, and logit gradients into a single pass, so we don't have to write unnecessary
(B, T, V) tensors. Such an operation is only possible if `dloss` can be known beforehand, which doesn't seem like
much of a restriction: In pretraining, it is just a constant 1/batch_size tensor, for fine-tuning we might zero
out the input prompt, but that is known in advance.

Compile example:
icpx -fsycl -O3 classifier_fused.cpp -o classifier_fused
*/

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}


void crossentropy_forward_cpu(float* losses,
                              const float* probs, const int* targets,
                              int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

SoftmaxParams prepare_softmax(sycl::sub_group warp,
                                         int idx, const float* inp, int V) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    const float* x = inp + idx * V;

    float maxval = -INFINITY;
    float sumval = 0.0f;

    for (int i = warp.get_local_linear_id(); i < V; i += warp.get_local_linear_range()) {
        float v = x[i];
        float old_maxval = maxval;
        maxval = fmaxf(maxval, v);
        sumval *= expf((old_maxval - maxval));
        sumval += expf(v - maxval);
    }


    //float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    float global_maxval = sycl::reduce_over_group(warp, maxval, sycl::maximum<float>{});
    sumval *= expf((maxval - global_maxval));

    float sum = sycl::reduce_over_group(warp, sumval, sycl::plus<float>{});
    float norm = 1.f / sum;

    return SoftmaxParams{norm, global_maxval};
}


void fused_classifier_kernel(sycl::nd_item<1> id, float* dlogits, float* losses,
                             const float* logits, const float* dlosses, const int* targets,
                             int B, int T, int V) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if (idx >= B * T) {
        return;
    }

    // local indices
    int b = idx / T;
    int t = idx % T;

    auto sp = prepare_softmax(warp, idx, logits, V);

    // calculate the probability needed for the loss and update.
    // single-threaded
    if(warp.leader()) {
        int ix = targets[b * T + t];
        float prob = expf(logits[idx * V + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -logf(prob);
    }

    // calculate all the gradients
    for (int i = warp.get_local_linear_id(); i < V; i += warp.get_max_local_range()[0]) {
        float prob = expf(logits[i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = prob;
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] += (p - indicator) * dloss;
    }

}

// ----------------------------------------------------------------------------
// kernel launcher

void fused_classifier1(sycl::queue &queue, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    queue.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        fused_classifier_kernel(id, dlogits, losses, logits, dlosses, targets, B, T, V);
    });
}

void fused_classifier(sycl::queue &queue, int kernel_num, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int block_size) {
    switch (kernel_num) {
        case 1:
            fused_classifier1(queue, dlogits, losses, logits, dlosses, targets, B, T, V, block_size);
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
    int V = 50257;

    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});

    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }

    // create host memory of random numbers
    const float* logits = make_random_float_01(B * T * V);
    float* probs = (float*)malloc(B * T * V * sizeof(float));
    float* dlogits = (float*)malloc(B * T * V * sizeof(float));
    float* losses = (float*)malloc(B * T * sizeof(float));
    const float* dlosses = make_random_float(B * T);
    const int* targets = make_random_int(B * T, V);

    // move to GPU
    float* d_logits = sycl::malloc_device<float>(B * T * V, defaultQueue);
    float* d_dlogits = sycl::malloc_device<float>(B * T * V, defaultQueue);
    float* d_losses = sycl::malloc_device<float>(B * T, defaultQueue);
    float* d_dlosses = sycl::malloc_device<float>(B * T, defaultQueue);
    int* d_targets = sycl::malloc_device<int>(B * T, defaultQueue);
    
    syclMallocCheck(d_logits);
    syclMallocCheck(d_dlogits);
    syclMallocCheck(d_losses);
    syclMallocCheck(d_dlosses);
    syclMallocCheck(d_targets);

    defaultQueue.memcpy(d_logits, logits, B * T * V * sizeof(float));
    defaultQueue.memcpy(d_dlosses, dlosses, B * T * sizeof(float));
    defaultQueue.memcpy(d_targets, targets, B * T * sizeof(int));
  
    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    softmax_forward_cpu(probs, logits, B * T, V);
    crossentropy_forward_cpu(losses, probs, targets, B, T, V);
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    // time the kernel at different block sizes
    // 1024 is not supported on Intel GPUs
    int block_sizes[] = {32, 64, 128, 256, 512}; 

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        fused_classifier(defaultQueue, kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, block_size);
        validate_result(defaultQueue, d_losses, losses, "losses", B * T, 1e-4f);
        validate_result(defaultQueue, d_dlogits, dlogits, "dlogits", B * T * V, 1e-4f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(defaultQueue, repeat_times, fused_classifier,
                                              kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets,
                                              B, T, V, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free((void*)logits);
    free(probs);
    free(dlogits);
    free(losses);
    free((void*)dlosses);
    free((void*)targets);

    sycl::free(d_logits, defaultQueue);
    sycl::free(d_dlogits, defaultQueue);
    sycl::free(d_losses, defaultQueue);
    sycl::free(d_dlosses, defaultQueue);
    sycl::free(d_targets, defaultQueue);
  
    return 0;
}