/*
Kernels for softmax forward pass.

Compile example:
icpx -fsycl -O3 softmax_forward.cpp -o softmax_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./softmax_forward 1

version 2 is a fused kernel that parallelizes over all of B,T,C
./softmax_forward 2

version 3 uses intra-warp reductions for maxval and sumval, must use block_size=32
./softmax_forward 3

version 4 uses both intra-warp reductions and shared memory for inter-warp reductions
so it can tolerate any block_size % 32 == 0. this is hopefully the most efficient version
./softmax_forward 4

version 5 is naive port from CPU code (softmax_online) to kernel: parallelizes over B,T, loops over C
./softmax_forward 5

version 6 is softmax_online that parallelizes over all of B,T,C
./softmax_forward 6

version 7 is softmax optimized for very large C.
./softmax_forward 7

Note: DPC++ requires this file to be compiled with -fp-model=precise
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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
        // Note: since we want to ensure that the CUDA-kernels are accurate,
        // we do this accumulation in higher precision, so we can be assured
        // that our ground-truth is of high quality.
        // Note 2: Intel client GPUs do not currently support fp64, so back to fp32 for now
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}


// online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        float sum = 0.0f;
		for (int j = 0; j < C; j++) {
			float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
			} else {
				sum += expf(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// validates
void softmax_forward_kernel1(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = id.get_global_linear_id();
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        // Note: double changed to float since client GPUs do not support fp64
        float sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = sycl::exp(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

// doesn't validate
void softmax_forward_kernel2(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)
    int idx = id.get_group(0); // ranges [0, N)
    int tid = id.get_local_linear_id(); // ranges [0, block_size)
    int block_size = id.get_local_range(0);
    const float* x = inp + idx * C; // idx-th row of inp
    // thread coarsening
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = sycl::fmax(maxval, x[i]);
    }
    maxval = sycl::reduce_over_group(id.get_group(), maxval, sycl::maximum<float>());
   
    float offset = maxval;
    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = sycl::exp(x[i] - offset);
    }

    // I think we don't need this one?
    id.barrier();

    // thread coarsening again, for the sum
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    sumval = sycl::reduce_over_group(id.get_group(), sumval, sycl::plus<float>());
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sumval;
    }
}

// doesn't validate
void softmax_forward_kernel3(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
    // kernel must use block size of 32
    int idx = id.get_group(0);
    int tid = id.get_local_linear_id();
    const float* x = inp + idx * C;

    // Thread coarsening and within-warp reduction for maxval
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        maxval = sycl::fmax(maxval, x[i]);
    }
    maxval = sycl::reduce_over_group(id.get_sub_group(), maxval, sycl::maximum<float>());
 
    // Broadcast maxval within the warp
    float offset = maxval; 
    
    // Compute expf and write the result to global memory
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        out[idx * C + i] = sycl::exp(x[i] - offset);
    }

    // Thread coarsening and within-warp reduction for sumval
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        sumval += x[i];
    }
    sumval = sycl::reduce_over_group(id.get_sub_group(), sumval, sycl::plus<float>());
 
    // Broadcast sumval within the warp
    float sum = sumval; 
  
    // Divide the input values by the sum
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        out[idx * C + i] = x[i] / sum;
    }
}

// doesn't validate
void softmax_forward_kernel4(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    int idx = id.get_group(0);
    int tid = id.get_local_linear_id();
    sycl::sub_group sg = id.get_sub_group();
  
    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        maxval = sycl::fmax(maxval, x[i]);
    }
    maxval = sycl::reduce_over_group(id.get_group(), maxval, sycl::maximum<float>());

    // broadcast the max to all threads
    float offset = maxval; 

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        out[idx * C + i] = sycl::exp(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        sumval += x[i];
    }
    sumval = sycl::reduce_over_group(id.get_group(), sumval, sycl::plus<float>());
   
    // broadcast the sum to all threads
    float sum = sumval;

    // divide the whole row by the sum
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        out[idx * C + i] = x[i] / sum;
    }
}

// doesn't validate
void softmax_forward_online_kernel1(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = id.get_global_linear_id();
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        // Note: double changed to float since client GPUs do not support fp64
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * sycl::exp(maxval_prev - maxval) + sycl::exp(inp_row[j] - maxval);
			}
			else {
				sum += sycl::exp(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = sycl::exp(inp_row[j] - maxval) / sum;
        }
    }
}

// struct for the reduction operation, guarantees 8-byte alignment
// struct __align__(8) SumMax
struct SumMax
{
    float maxval;
    float sum;
    
};

struct SumMaxReduce {
    SumMax operator()(const SumMax& a, const SumMax& b) const {
        bool a_bigger = (a.maxval > b.maxval);
        SumMax bigger_m = a_bigger ? a : b;
        SumMax smaller_m = a_bigger ? b : a;
        SumMax res;
        res.maxval = bigger_m.maxval;
        res.sum = bigger_m.sum + smaller_m.sum * sycl::exp(smaller_m.maxval - bigger_m.maxval);
        
        return res;
    }
};

// doesn't validate
void softmax_forward_online_kernel2(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
	sycl::sub_group warp = id.get_sub_group();
	int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
	if (idx >= N) {
		return;
	}

	// one row of inp, i.e. inp[idx, :] of shape (C,)
	const float* x = inp + idx * C;

    // base case for the reduction
    SumMax sm_partial;
	sm_partial.maxval = -INFINITY;
	sm_partial.sum = 0.0f;

	// first, thread coarsening by directly accessing global memory in series
	for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
		sm_partial = SumMaxReduce()(sm_partial, { x[i], 1.0f });
	}

    // second, the reduction
    SumMax sm_total = sycl::reduce_over_group(warp, sm_partial, SumMaxReduce());

	// divide the whole row by the sum
	for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        // the below is equivalent to
        // out[idx * C + i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
        // but uses special instruction that bypasses the cache
        // Fix this later
        out[idx * C + i] = sycl::exp(x[i] - sm_total.maxval) / sm_total.sum;
        //__stcs(out + idx * C + i, expf(x[i] - sm_total.maxval) / sm_total.sum);
	}
}

// doesn't validate
void softmax_forward_kernel7(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    
    int idx = id.get_group(0);
    int tid = id.get_local_linear_id();
  
    if (tid >= C) {
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += id.get_local_range(0) * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = sycl::fmax(maxval, x[sycl::min(C - 1, static_cast<int>(i + u*id.get_local_range(0)))]);
        }
    }
    maxval = sycl::reduce_over_group(id.get_group(), maxval, sycl::maximum<float>());

    // broadcast the max to all threads
    float offset = maxval;

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += id.get_local_range(0) * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            // Fix this later
            // reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
            reg_array[u] = x[sycl::min(C - 1, static_cast<int>(i + u*id.get_local_range(0)))];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*id.get_local_range(0) < C) {
                float output = sycl::exp(reg_array[u] - offset);
                y[sycl::min(C - 1, static_cast<int>(i + u*id.get_local_range(0)))] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = sycl::reduce_over_group(id.get_group(), sumval, sycl::plus<float>());
    // broadcast the sum to all threads
    float sum = sumval;

    // divide the whole row by the sum
    for (int i = tid; i < C; i += id.get_local_range(0) * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[sycl::min(C - 1, static_cast<int>(i + u*id.get_local_range(0)))];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*id.get_local_range(0) < C) {
                y[i + u*id.get_local_range(0)] = reg_array[u] / sum;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void softmax_forward1(float* out, const float* inp, int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel1(id, out, inp, N, C);
    });
}

void softmax_forward2(float* out, const float* inp, int N, int C, const int block_size) {
    int grid_size = N;
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel2(id, out, inp, N, C);
    });
}

void softmax_forward3(float* out, const float* inp, int N, int C, int block_size) {
    block_size = 32; // awkward but ok. this one only works with block size 32
    int grid_size = N;
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel3(id, out, inp, N, C);
    });
}

void softmax_forward4(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel4(id, out, inp, N, C);
    });
}

void softmax_forward_online1(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_online_kernel1(id, out, inp, N, C);
    });
}

void softmax_forward_online2(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N * 32, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_online_kernel2(id, out, inp, N, C);
    });
}

void softmax_forward7(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel7(id, out, inp, N, C);
    });
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            softmax_forward1(out, inp, N, C, block_size);
            break;
        case 2:
            softmax_forward2(out, inp, N, C, block_size);
            break;
        case 3:
            softmax_forward3(out, inp, N, C, block_size);
            break;
        case 4:
            softmax_forward4(out, inp, N, C, block_size);
            break;
        case 5:
            softmax_forward_online1(out, inp, N, C, block_size);
            break;
        case 6:
            softmax_forward_online2(out, inp, N, C, block_size);
            break;
        case 7:
            softmax_forward7(out, inp, N, C, block_size);
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
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    printf("Using Platform: %s\n", defaultQueue.get_device().get_platform().get_info<sycl::info::platform::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }
    DefaultQueue = &defaultQueue;

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * V * sizeof(float));
    float* inp = make_random_float(B * T * V);

    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful.
    const int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k) {
        for(int j = 0; j < B * T; ++j) {
            inp[j * V +  outliers[j*3 + k]] *= 20;
        }
    }

    // move to GPU
    float* d_out;
    float* d_inp;
    syclMallocCheck(d_out = sycl::malloc_device<float>(B * T * V, defaultQueue));
    syclMallocCheck(d_inp = sycl::malloc_device<float>(B * T * V, defaultQueue));

    defaultQueue.memcpy(d_inp, inp, B * T * V * sizeof(float));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // 1024 is not supported on Intel GPUs
    int block_sizes[] = {32, 64, 128, 256, 512};

    softmax_forward_cpu(out, inp, B * T, V);
    {
        float max_el = -INFINITY;
        for(int i = 0; i <  B * T * V; ++i) {
            max_el = std::fmax(max_el, out[i]);
        }
        assert(max_el > 1e-4);
        printf("Largest output is: %f\n", max_el);
    }

    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_forward(kernel_num, d_out, d_inp, B * T, V, block_size);
        // Tweak tolerance for now b/c fp32
        validate_result(d_out, out, "out", B * T * V, 1e-3f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                              kernel_num, d_out, d_inp, B * T, V, block_size);

        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
    }

    // free memory
    free(out);
    free(inp);
  
    sycl::free(d_out, defaultQueue);
    sycl::free(d_inp, defaultQueue);

    return 0;
}