#include <stdlib.h>
#include <stdio.h>
#include <cmath>

namespace syclx = sycl::ext::oneapi;
namespace syclex = sycl::ext::oneapi::experimental;

// Random Globals
sycl::queue* DefaultQueue = nullptr;

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

inline float warpReduceSum(sycl::sub_group warp, float val) {
    return sycl::reduce_over_group(warp, val, sycl::plus<float>{});
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

int get_num_CUs() {
    // TODO - make this more general for diff types of devices
    int num_CUs = 0;
    if (DefaultQueue && DefaultQueue->get_device().is_gpu()) {
        int num_slices = DefaultQueue->get_device().get_info<sycl::ext::intel::info::device::gpu_slices>();
        int num_subslices_per_slice = DefaultQueue->get_device().get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
        num_CUs = num_slices * num_subslices_per_slice;
    } else {
        num_CUs = 1;
    }
    return num_CUs;
}
// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    explicit Packed128(sycl::int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        // Fix this later
        *reinterpret_cast<sycl::int4*>(payload) = bits;
        
    }

    static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }

    static Packed128 zeros() {
        return constant(0);
    }

    static Packed128 ones() {
        return constant(1);
    }

    ElementType& operator[](int index) {
        return payload[index];
    }
    const ElementType& operator[](int index) const {
        return payload[index];
    }
    sycl::int4 get_bits() const {
        sycl::int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        // Fix this later
        bits = *reinterpret_cast<const sycl::int4*>(payload);
        return bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    static constexpr const int size = sizeof(sycl::int4) / sizeof(ElementType);
    ElementType payload[size];
};

// short-form typedef
typedef Packed128<float> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const sycl::int4*>(address)};
}

// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
Packed128<ElementType> load128cs(const ElementType* address) {
    // Fix this later
    // return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
    return Packed128<ElementType>{*reinterpret_cast<const sycl::int4*>(address)};
}

// store a Packed128 to an aligned memory address
template<class ElementType>
void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::int4*>(target) = value.get_bits();
}

// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
void store128cs(ElementType* target, Packed128<ElementType> value) {
    // Fix this later
    //__stcs(reinterpret_cast<int4*>(target), value.get_bits());
    *reinterpret_cast<sycl::int4*>(target) = value.get_bits();
}

// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
void store128cg(ElementType* target, Packed128<ElementType> value) {
    // Fix this later
    *reinterpret_cast<sycl::int4*>(target) = value.get_bits();
}


// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#if defined(ENABLE_BF16)

typedef syclx::bfloat16 floatX;
typedef syclx::bfloat16 floatN;

#elif defined(ENABLE_FP16)

typedef sycl::half floatX;
typedef sycl::half floatN;

#else

typedef float floatX;
typedef float floatN;
#endif

typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
void memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }
    DefaultQueue->memcpy(d_ptr, converted, count * sizeof(TargetType)).wait();
    free(converted);

    return;
}

void setup_main() {
    srand(0);   // determinism
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    DefaultQueue->memcpy(out_gpu, device_result, num_elements * sizeof(D)).wait();
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // ensure correctness for all elements. We can set an "ignore" mask by writing NaN
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && std::isfinite(cpu_reference[i])) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    // reset the result pointer, so we can chain multiple tests and don't miss trivial errors,
    // like the kernel not writing to part of the result.
    // queue.memset(device_result, 0, num_elements * sizeof(T)).wait();
    
    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    // TODO - get SYCL analog of cudaEventElapsedTime and cudaEventRecord
    //cudaEvent_t start, stop;
    //cudaCheck(cudaEventCreate(&start));
    //cudaCheck(cudaEventCreate(&stop));
    //cudaCheck(cudaEventRecord(start, nullptr));
    // Look into something like the cache flush on the cuda side
    sycl::event start = DefaultQueue->ext_oneapi_submit_barrier();
    for (int i = 0; i < repeats; i++) {
        kernel(std::forward<KernelArgs>(kernel_args)...);
    }
    sycl::event stop = DefaultQueue->ext_oneapi_submit_barrier();
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

// ----------------------------------------------------------------------------
// CUDA indexing helper functions

// threadIdx
template <int N>
int threadIdx_x(sycl::nd_item<N> id);

template <>
int threadIdx_x<1>(sycl::nd_item<1> id) {
  return id.get_local_id(0);
}

template <>
int threadIdx_x<2>(sycl::nd_item<2> id) {
  return id.get_local_id(1);
}

template <>
int threadIdx_x<3>(sycl::nd_item<3> id) {
  return id.get_local_id(2);
}

template <int N>
int threadIdx_y(sycl::nd_item<N> id);

template <>
int threadIdx_y<1>(sycl::nd_item<1> id) {
  return 0;
}

template <>
int threadIdx_y<2>(sycl::nd_item<2> id) {
  return id.get_local_id(0);
}

template <>
int threadIdx_y<3>(sycl::nd_item<3> id) {
  return id.get_local_id(1);
}

template <int N>
int threadIdx_z(sycl::nd_item<N> id);

template <>
int threadIdx_z<1>(sycl::nd_item<1> id) {
    return 0;
}
template <>
int threadIdx_z<2>(sycl::nd_item<2> id) {
    return 0;
}
template <>
int threadIdx_z<3>(sycl::nd_item<3> id) {
    return id.get_local_id(0);
}

// blockIdx
template <int N>
int blockIdx_x(sycl::nd_item<N> id);

template <>
int blockIdx_x<1>(sycl::nd_item<1> id) {
    return id.get_group(0);
}

template <>
int blockIdx_x<2>(sycl::nd_item<2> id) {
    return id.get_group(1);
}

template <>
int blockIdx_x<3>(sycl::nd_item<3> id) {
    return id.get_group(2);
}

template <int N>
int blockIdx_y(sycl::nd_item<N> id);

template <>
int blockIdx_y<1>(sycl::nd_item<1> id) {
    return 0;
}

template <>
int blockIdx_y<2>(sycl::nd_item<2> id) {
    return id.get_group(0);
}

template <>
int blockIdx_y<3>(sycl::nd_item<3> id) {
    return id.get_group(1);
}

template <int N>
int blockIdx_z(sycl::nd_item<N> id);

template <>
int blockIdx_z<1>(sycl::nd_item<1> id) {
    return 0;
}

template <>
int blockIdx_z<2>(sycl::nd_item<2> id) {
    return 0;
}

template <>
int blockIdx_z<3>(sycl::nd_item<3> id) {
    return id.get_group(0);
}

// gridDim
template <int N>
int gridDim_x(sycl::nd_item<N> id);

template <>
int gridDim_x<1>(sycl::nd_item<1> id) {
    return id.get_group_range(0);
}

template <>
int gridDim_x<2>(sycl::nd_item<2> id) {
    return id.get_group_range(1);
}

template <>
int gridDim_x<3>(sycl::nd_item<3> id) {
    return id.get_group_range(2);
}

template <int N>
int gridDim_y(sycl::nd_item<N> id);

template <>
int gridDim_y<1>(sycl::nd_item<1> id) {
    return 1;
}

template <>
int gridDim_y<2>(sycl::nd_item<2> id) {
    return id.get_group_range(0);
}

template <>
int gridDim_y<3>(sycl::nd_item<3> id) {
    return id.get_group_range(1);
}

template <int N>
int gridDim_z(sycl::nd_item<N> id);

template <>
int gridDim_z<1>(sycl::nd_item<1> id) {
    return 1;
}

template <>
int gridDim_z<2>(sycl::nd_item<2> id) {
    return 1;
}

template <>
int gridDim_z<3>(sycl::nd_item<3> id) {
    return id.get_group_range(0);
}

// blockDim
template <int N>
int blockDim_x(sycl::nd_item<N> id);

template <>
int blockDim_x<1>(sycl::nd_item<1> id) {
    return id.get_local_range(0);
}

template <>
int blockDim_x<2>(sycl::nd_item<2> id) {
    return id.get_local_range(1);
}

template <>
int blockDim_x<3>(sycl::nd_item<3> id) {
    return id.get_local_range(2);
}

template <int N>
int blockDim_y(sycl::nd_item<N> id);

template <>
int blockDim_y<1>(sycl::nd_item<1> id) {
    return 1;
}

template <>
int blockDim_y<2>(sycl::nd_item<2> id) {
    return id.get_local_range(0);
}

template <>
int blockDim_y<3>(sycl::nd_item<3> id) {
    return id.get_local_range(1);
}

template <int N>
int blockDim_z(sycl::nd_item<N> id);

template <>
int blockDim_z<1>(sycl::nd_item<1> id) {
    return 1;
}

template <>
int blockDim_z<2>(sycl::nd_item<2> id) {
    return 1;
}

template <>
int blockDim_z<3>(sycl::nd_item<3> id) {
    return id.get_local_range(0);
}

int meta_group_size(sycl::sub_group warp) {
    return warp.get_group_linear_range();
}

int meta_group_rank(sycl::sub_group warp) {
    return warp.get_group_linear_id();
}

int thread_rank(sycl::sub_group warp) {
    return warp.get_local_linear_id();
}

int size(sycl::sub_group warp) {
    return warp.get_max_local_range()[0];
}

template <int N>
int thread_rank(sycl::group<N> block) {
    return block.get_local_linear_id();
}

// ----------------------------------------------------------------------------
// Atomic Helpers

float atomicAdd(float* addr, float val) {
    sycl::atomic_ref<float, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*addr);
    return ref.fetch_add(val);
}

unsigned int atomicAdd(unsigned int* addr, unsigned int val) {
    sycl::atomic_ref<unsigned int, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*addr);
    return ref.fetch_add(val);
}

unsigned int atomicInc(unsigned int* addr, unsigned int val) {
    sycl::atomic_ref<unsigned int, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*addr);
    unsigned int old_val = ref.load();
    unsigned int new_val = ((old_val >= val) ? 0 : (old_val+1));
    do {
        new_val = ((old_val >= val) ? 0 : (old_val+1));
    } while (!ref.compare_exchange_weak(old_val, new_val));
    
    return old_val;
}

unsigned int atomicCAS(unsigned int* addr, unsigned int compare, unsigned int val) {
    sycl::atomic_ref<unsigned int, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*addr);
    ref.compare_exchange_strong(compare, val);
    return compare;
}


