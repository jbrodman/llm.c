// Utilities for use in __device__ code

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include "sycl_common.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace dnnlsycl = dnnl::sycl_interop;

// ----------------------------------------------------------------------------

int get_num_CUs() {
    // TODO - make this more general for diff types of devices
    int num_CUs = 0;
    if (main_stream && main_stream->get_device().is_gpu()) {
        int num_slices = main_stream->get_device().get_info<sycl::ext::intel::info::device::gpu_slices>();
        int num_subslices_per_slice = main_stream->get_device().get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
        num_CUs = num_slices * num_subslices_per_slice;
    } else {
        num_CUs = 1;
    }
    return num_CUs;
}
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
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
    //__stcg(reinterpret_cast<int4*>(target), value.get_bits());
    *reinterpret_cast<sycl::int4*>(target) = value.get_bits();
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template<typename Td, typename Ts>
Td cast_value(Ts val);

template<>
float cast_value<float, float>(float val) {
    return val;
}

template<>
float cast_value<float, sycl::half>(sycl::half val) {
    return (float)val;
}

template<>
float cast_value<float, syclx::bfloat16>(syclx::bfloat16 val) {
    return (float)val;
}

template<typename Td, typename Ts>
void copy_and_cast_kernel(sycl::nd_item<2> id, Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = id.get_group(1) * id.get_local_range(1) + id.get_local_id(1);
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx + stride_dst * id.get_group(0)] = cast_value<Td, Ts>(src[idx + stride_src * id.get_group(0)]);
    }
}

// ----------------------------------------------------------------------------
// Warp/Block communication primitives

// warp-level reduction for summing values
inline float warpReduceSum(sycl::sub_group warp, float val) {
    return sycl::reduce_over_group(warp, val, sycl::plus<float>{});
}
// warp-level reduction for finding the maximum value
inline float warpReduceMax(sycl::sub_group warp, float val) {
    return sycl::reduce_over_group(warp, val, sycl::maximum<float>{});
}

float shfl_down(sycl::sub_group warp, float val, int delta, int width) {
    int id = warp.get_local_id();
    int subid = id % width;
    float result = sycl::shift_group_left(warp, val, delta);
    if (subid + delta >= width)
        result = val;
    return result;
}

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
constexpr unsigned int SquirrelNoise5(int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = (unsigned int) positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}
constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr int PRIME_NUMBER = 198491317; // Large prime number with non-boring bits
    return SquirrelNoise5(indexX + (PRIME_NUMBER * indexY), seed);
}


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

#ifdef ENABLE_BF16
void atomicAdd(syclx::bfloat16* addr, syclx::bfloat16 val) {
    /*
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    __nv_bfloat162* ptr_bf16 = reinterpret_cast<__nv_bfloat162*>(ptr_val & ~uintptr_t(0x3));
    // Prepare the value to add, setting the other half to zero
    __nv_bfloat162 add_val = (ptr_val & 0x3) ? __halves2bfloat162(__ushort_as_bfloat16(0), val)
                                             : __halves2bfloat162(val, __ushort_as_bfloat16(0));
    atomicAdd(ptr_bf16, add_val);
    */
    // I think the best thing to do here for now is to compare and swap
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    uint32_t* ptr_32bits = reinterpret_cast<uint32_t*>(ptr_val & ~uintptr_t(0x3));

    sycl::atomic_ref<uint32_t,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device> ref(*ptr_32bits);
    uint32_t old_val = ref.load();
    uint32_t new_val = old_val;
    do {
        sycl::marray<syclx::bfloat16, 2> h2 = *reinterpret_cast<sycl::marray<syclx::bfloat16, 2>*>(&old_val);
        h2[0] += (ptr_val & 0x3) ? syclx::bfloat16(0.0f) : val;
        h2[1] += (ptr_val & 0x3) ? val : syclx::bfloat16(0.0f);
        new_val = *reinterpret_cast<uint32_t*>(&h2);
    }
    while (!ref.compare_exchange_weak(old_val, new_val));
}
#endif

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

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
inline void stochastic_rounding(sycl::nd_item<2> id, float in, syclx::bfloat16 *out, unsigned int seed) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    unsigned int random = Get2dNoiseUint(threadIdx_x(id), blockIdx_x(id) * blockDim_x(id) + blockIdx_y(id), seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = sycl::bit_cast<unsigned int>(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = syclx::bfloat16(sycl::bit_cast<float>(float_bits));
}
inline void stochastic_rounding(sycl::nd_item<2> id, float in, sycl::half *out, unsigned int random) {
    *out = (float)in; // todo - implement this...
}
inline void stochastic_rounding(sycl::nd_item<2> id, float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

#endif