#include <stdlib.h>
#include <stdio.h>

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// ----------------------------------------------------------------------------
// checking utils

// SYCL check allocation
void sycl_malloc_check(void* ptr, const char *file, int line) {
    if (ptr == nullptr) {
        printf("[SYCL ERROR] at file %s:%d:\nFailed to allocate memory\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define syclMallocCheck(ptr) (sycl_malloc_check(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(int N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

/*
float* make_zeros_float(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}
*/

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class T>
void validate_result(sycl::queue &queue, T* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    T* out_gpu = (T*)malloc(num_elements * sizeof(T));
    queue.memcpy(out_gpu, device_result, num_elements * sizeof(T)).wait();
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], out_gpu[i]);
        }
        // ensure correctness for all elements
        if (fabs(cpu_reference[i] - out_gpu[i]) > tolerance) {
            printf("Mismatch of %s at %d: %f vs %f\n", name, i, cpu_reference[i], out_gpu[i]);
            free(out_gpu);
            exit(EXIT_FAILURE);
        }
    }

    // reset the result pointer, so we can chain multiple tests and don't miss trivial errors,
    // like the kernel not writing to part of the result.
    queue.memset(device_result, 0, num_elements * sizeof(T)).wait();
    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(sycl::queue& queue, int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    // TODO - get SYCL analog of cudaEventElapsedTime and cudaEventRecord
    //cudaEvent_t start, stop;
    //cudaCheck(cudaEventCreate(&start));
    //cudaCheck(cudaEventCreate(&stop));
    //cudaCheck(cudaEventRecord(start, nullptr));
    sycl::event start = queue.ext_oneapi_submit_barrier();
    for (int i = 0; i < repeats; i++) {
        kernel(queue, std::forward<KernelArgs>(kernel_args)...);
    }
    sycl::event stop = queue.ext_oneapi_submit_barrier();
    //cudaCheck(cudaEventRecord(stop, nullptr));
    //cudaCheck(cudaEventSynchronize(start));
    //cudaCheck(cudaEventSynchronize(stop));

    float elapsed_time = 0.0;
    elapsed_time = stop.get_profiling_info<sycl::info::event_profiling::command_end>() 
        - start.get_profiling_info<sycl::info::event_profiling::command_end>();
    //cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

    return elapsed_time / repeats;
}

// Hack this in to restore John's group algorithm support for non-native types
// when using sycl::sub_groups
namespace sycl {
template <typename T, typename BinaryOperation>
using EnableIfIsNonNativeOp = sycl::detail::enable_if_t<
    (!sycl::detail::is_scalar_arithmetic<T>::value &&
     !sycl::detail::is_vector_arithmetic<T>::value &&
     std::is_trivially_copyable<T>::value) ||
        !sycl::detail::is_native_op<T, BinaryOperation>::value,
    T>;

template <typename Group, typename T, class BinaryOperation>
EnableIfIsNonNativeOp<T, BinaryOperation> reduce_over_group(Group g, T x,
                                                 BinaryOperation op) {
  static_assert(sycl::detail::is_sub_group<Group>::value,
                "reduce algorithm with user-defined types and operators"
                "only supports intel::sub_group class.");
  T result = x;
  for (int mask = 1; mask < g.get_max_local_range()[0]; mask *= 2) {
    T tmp = g.shuffle_xor(result, id<1>(mask));
    if ((g.get_local_id()[0] ^ mask) < g.get_local_range()[0]) {
      result = op(result, tmp);
    }
  }
  return g.shuffle(result, 0);
}
}  // namespace sycl