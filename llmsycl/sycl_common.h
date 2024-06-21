/*
Common utilities for SYCL code.
*/
#ifndef SYCL_COMMON_H
#define SYCL_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sycl/sycl.hpp>

namespace syclx = sycl::ext::oneapi;
namespace syclex = sycl::ext::oneapi::experimental;

// ----------------------------------------------------------------------------
// Global defines and settings

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#define WARP_SIZE 16U
#define __SIMD16__ [[sycl::reqd_sub_group_size(16)]]
#define __SIMD32__ [[sycl::reqd_sub_group_size(32)]]

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


// ----------------------------------------------------------------------------
// Error checking
// SYCL check allocation
void sycl_malloc_check(void* ptr, const char *file, int line) {
    if (ptr == nullptr) {
        printf("[SYCL ERROR] at file %s:%d:\nFailed to allocate memory\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define syclMallocCheck(ptr) (sycl_malloc_check(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
typedef sycl::half floatX;
#define PRECISION_MODE PRECISION_FP16
#else // Default to bfloat16
typedef syclx::bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}

__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif

// Helpers for this until I figure out how to deal with the cache hints
floatX __ldcs(const floatX* address) {
    return *address;
}

void __stcs(floatX* address, floatX value) {
    *address = value;
}

// ----------------------------------------------------------------------------
// Profiler utils


#endif // SYCL_COMMON_H