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

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, sycl::queue* stream) {
    // allocate pinned buffer for faster, async transfer
    char* buffer_space;
    syclMallocCheck(buffer_space = sycl::malloc_host<char>(2*buffer_size, *stream));
    // split allocation in two
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // prime the read buffer; first copy means we have to wait
    char* gpu_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    stream->memcpy(read_buffer, gpu_read_ptr, copy_amount);
    stream->wait();
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    // now the main loop; as long as there are bytes left
    while(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        stream->memcpy(read_buffer, gpu_read_ptr, copy_amount);
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        stream->wait();    // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    sycl::free(buffer_space, *stream);
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, sycl::queue* stream) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    char* buffer_space;
    syclMallocCheck(buffer_space = sycl::malloc_host<char>(2*buffer_size, *stream));
    // split allocation in two
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // prime the read buffer;
    char* gpu_write_ptr = (char*)dest;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    // now the main loop; as long as there are bytes left
    while(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        stream->memcpy(gpu_write_ptr, write_buffer, write_buffer_size);
        gpu_write_ptr += write_buffer_size;
        // while this is going on, read from disk
        freadCheck(read_buffer, 1, copy_amount, src);
        stream->wait();    // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    stream->memcpy(gpu_write_ptr, write_buffer, write_buffer_size);
    stream->wait();
    sycl::free(buffer_space, *stream);
}

#endif // SYCL_COMMON_H