/*
GPT-2 Transformer Neural Net trained in raw CUDA
Non-trivial notes to be aware of:

We are being clever in the backward pass to conserve memory.
In particular, all parameters use a += in the backward pass, so we
can later do gradient accumulation. But all activations have = instead of +=
because these are faster (just read, no write). This is okay for all activations
except for those in the residual stream, where the gradients have to add. We make
sure that those parts work out ok and that we do a += as necessary. E.g.,
the layernorms are connected to the residuals so we += in layernorm backward.

In this file we are using Mixed Precision training, so different activations,
paramaters, grads and buffers may be kept at different precisions, to take
advantage of the fast low-precision hardware in the latest GPUs (bf16/fp16),
and fp8 (coming soon^TM).

Compile:
make train_gpt2cu

Example launch using bfloat16 on 1 GPU batch size 8, sample/eval every 200 steps:
Also we're using TinyStories here for example as it is a bigger dataset
./train_gpt2cu -b 8 -v 200 -s 200 -i data/TinyStories

Example launch using bfloat16 on 4 GPUs, same as above:
mpirun -np 4 ./train_gpt2cu -b 8 -v 200 -s 200 -i data/TinyStories

If you'd like to see train_gpt2.cu produce identical results to
`python train_gpt2.py`, you can run it like this:
make train_gpt2cu PRECISION=FP32
./train_gpt2cu -b 4 -t 64 -l 1e-4 -v 200 -s 200 -a 1 -x 10 -f 0
This reads & runs in fp32, B=4, T=64, LR=1e-4, val/sample never (200),
-a 1 is "overfit single batch", -x 10 is 10 iterations, and -f 0 disables tf32
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
// GPU / CUDA related
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace dnnlsycl = dnnl::sycl_interop;
namespace syclx = sycl::ext::oneapi;

// Multi-GPU related
#ifdef MULTI_GPU
#include <mpi.h>
#include <nccl.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// CUDA precision settings

enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Default Properties
typedef float floatN;
#define CUBLAS_LOWP_COMPUTE cublas_compute_type
#ifdef MULTI_GPU
const ncclDataType_t ncclFloatN = ncclFloat;
#endif

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
const char* load_filename = "gpt2_124M.bin";
const char* precision_mode_str = "fp32";
#ifdef MULTI_GPU
const ncclDataType_t ncclFloatX = ncclFloat;
#endif

// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
typedef sycl::half floatX;
#define PRECISION_MODE PRECISION_FP16
const char* load_filename = "gpt2_124M.bin";
const char* precision_mode_str = "fp16";
#ifdef MULTI_GPU
const ncclDataType_t ncclFloatX = ncclHalf;
#endif

#else // Default to bfloat16
typedef syclx::bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
const char* load_filename = "gpt2_124M_bf16.bin"; // bf16 weights specific filename
const char* precision_mode_str = "bf16";
#ifdef MULTI_GPU
const ncclDataType_t ncclFloatX = ncclBfloat16;
#endif
#endif

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// SYCL check allocation
void sycl_malloc_check(void* ptr, const char *file, int line) {
    if (ptr == nullptr) {
        printf("[SYCL ERROR] at file %s:%d:\nFailed to allocate memory\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define syclMallocCheck(ptr) (sycl_malloc_check(ptr, __FILE__, __LINE__))

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

// Make a global pointer to the SYCL queue to avoid passing it around
sycl::queue *DefaultQueue = nullptr;
// oneDNN engine and stream
dnnl::engine *DefaultEngine = nullptr;
dnnl::stream *DefaultStream = nullptr;

#ifdef MULTI_GPU
void nccl_check(ncclResult_t status, const char *file, int line) {
    if (status != ncclSuccess) {
        printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line, ncclGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

void mpi_check(int status, const char *file, int line) {
    if (status != MPI_SUCCESS) {
        char mpi_error[4096];
        int mpi_error_len = 0;
        assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) == MPI_SUCCESS);
        printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len, mpi_error);
        exit(EXIT_FAILURE);
    }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))
#endif

// GPU helper functions for atomicAdd on smaller than 32-bit types
#ifdef ENABLE_BF16
void atomicAddX(syclx::bfloat16* addr, syclx::bfloat16 val) {
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
#ifdef ENABLE_FP16
void atomicAddX(sycl::half* addr, sycl::half val) {
    // Same thing as bfloat16
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    uint32_t* ptr_32bits = reinterpret_cast<uint32_t*>(ptr_val & ~uintptr_t(0x3));

    sycl::atomic_ref<uint32_t, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*ptr_32bits);
    uint32_t old_val = ref.load(); 
    uint32_t new_val = old_val;
    do {
        sycl::marray<sycl::half, 2> h2 = *reinterpret_cast<sycl::marray<sycl::half, 2>*>(&old_val);
        h2[0] += (ptr_val & 0x3) ? sycl::half(0.0f) : val;
        h2[1] += (ptr_val & 0x3) ? val : sycl::half(0.0f);
        new_val = *reinterpret_cast<uint32_t*>(&h2);
    }
    while (!ref.compare_exchange_weak(old_val, new_val));
}
#endif
void atomicAddX(float* addr, float val) {
    sycl::atomic_ref<float, 
                     sycl::memory_order::relaxed, 
                     sycl::memory_scope::device> ref(*addr);
    ref += val;
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
typedef Packed128<floatX> x128;

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

// ----------------------------------------------------------------------------
// Random Number Generatiom

// Simple xorshift RNG
unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

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
constexpr unsigned int Get1dNoiseUint(int positionX, unsigned int seed)
{
	return SquirrelNoise5(positionX, seed);
}
constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
	constexpr int PRIME_NUMBER = 198491317; // Large prime number with non-boring bits
	return SquirrelNoise5(indexX + (PRIME_NUMBER * indexY), seed);
}
constexpr float Get1dNoiseZeroToOne(int index, unsigned int seed)
{
	constexpr double ONE_OVER_MAX_UINT = (1.0 / (double) 0xFFFFFFFF);
	return (float)(ONE_OVER_MAX_UINT * (double) SquirrelNoise5(index, seed));
}
constexpr float Get2dNoiseZeroToOne(int indexX, int indexY, unsigned int seed)
{
	constexpr double ONE_OVER_MAX_UINT = (1.0 / (double) 0xFFFFFFFF);
	return (float)(ONE_OVER_MAX_UINT * (double) Get2dNoiseUint(indexX, indexY, seed));
}

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
inline void stochastic_rounding(sycl::nd_item<1> id, float in, syclx::bfloat16 *out, unsigned int seed) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    unsigned int random = Get2dNoiseUint(id.get_local_linear_id(), id.get_group(0), seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = *reinterpret_cast<unsigned int*>(&in); //__float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = syclx::bfloat16(*reinterpret_cast<float*>(&float_bits));
}
inline void stochastic_rounding(sycl::nd_item<1> id, float in, sycl::half *out, unsigned int seed) {
    unsigned int random = Get2dNoiseUint(id.get_local_linear_id(), id.get_group(0), seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = *reinterpret_cast<unsigned int*>(&in); //__float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = sycl::half(*reinterpret_cast<float*>(&float_bits));
    // *out = (float)in; // todo - implement this...
}
inline void stochastic_rounding(sycl::nd_item<1> id, float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

// ----------------------------------------------------------------------------
// MPI / multi-processing setup

// Parameters specific to training on multiple GPUs.
typedef struct {
    int process_rank;      // Rank of this process among all MPI processes. 0 if no multi-GPU.
    int num_processes;     // Total number of processes. 1 if no multi-GPU.
    int local_device_idx;  // This process GPU index on current machine. 0 if no multi-GPU.
#ifdef MULTI_GPU
    ncclComm_t nccl_comm;  // NCCL communication primitive, used for collective multi-GPU work.
#endif
} MultiGpuConfig;

// one global variable to hold the multi-GPU configuration for this process
MultiGpuConfig multi_gpu_config;

#ifdef MULTI_GPU
// Determine which GPU this process should use.
// Processes on the same machines use different GPU indicies. Processes on other machines don't.
// Copied from NCCL examples: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread
int multi_gpu_get_local_device_idx(int process_rank, int num_processes) {
  char hostname[1024];
  hostname[1023] = '\0';
  // All processes on the same machine will share the same hostname.
  gethostname(hostname, 1023);
  for (int i=0; i < 1024; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        break;
    }
  }
  uint64_t hostname_hash = 5381;
  for (int c = 0; hostname[c] != '\0'; c++){ hostname_hash = ((hostname_hash << 5) + hostname_hash) ^ hostname[c]; }

  // Distribute all hostname hashes to all processes.
  uint64_t* all_hostsname_hashes = (uint64_t*)malloc(num_processes * sizeof(uint64_t));
  all_hostsname_hashes[process_rank] = hostname_hash;
  mpiCheck(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_hostsname_hashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

  // Identify which GPU we need to use.
  int local_device_idx = 0;
  for (int current_process = 0; current_process < num_processes; ++current_process) {
     if (current_process == process_rank) {
      // Found my gpu, local_device_idx now has my target GPU index.
      break;
     }
     if (all_hostsname_hashes[current_process] == all_hostsname_hashes[process_rank]) {
      // This process ID runs on the same machine, but it's not me, skip this GPU
      local_device_idx++;
     }
  }

  free(all_hostsname_hashes);
  return local_device_idx;
}
#endif

MultiGpuConfig multi_gpu_config_init(int *argc, char ***argv) {
#ifdef MULTI_GPU
    // Initialize MPI.
    MultiGpuConfig result;
    mpiCheck(MPI_Init(argc, argv));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &result.process_rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &result.num_processes));
    result.local_device_idx = multi_gpu_get_local_device_idx(result.process_rank, result.num_processes);
    cudaCheck(cudaSetDevice(result.local_device_idx));
    ncclUniqueId nccl_id;
    if (result.process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
    }
    mpiCheck(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclCheck(ncclCommInitRank(&result.nccl_comm, result.num_processes, nccl_id, result.process_rank));
    return result;
#else
    printf("Multi-GPU support is disabled. Using a single GPU.\n");
    MultiGpuConfig result;
    result.process_rank = 0;
    result.num_processes = 1;
    result.local_device_idx = 0;
    return result;
#endif
}

void multi_gpu_config_free(const MultiGpuConfig* multi_gpu_config) {
#ifdef MULTI_GPU
    ncclCheck(ncclCommDestroy(multi_gpu_config->nccl_comm));
    mpiCheck(MPI_Finalize());
#endif
}

// convenience function that only prints if the rank of process is zero
void printf0(const char *format, ...) {
    if (multi_gpu_config.process_rank == 0) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

// ----------------------------------------------------------------------------
// all the kernels

void encoder_forward_kernel2(sycl::nd_item<1> id, floatX* out,
                               int* inp, floatX* wte, floatX* wpe,
                               int B, int T, int C) {
    int idx = id.get_global_id(0);
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        floatX* wte_ix = wte + ix * C + c;
        floatX* wpe_tc = wpe + t * C + c;
        *out_btc = (floatX)((float)*wte_ix + (float)*wpe_tc);
    }
}

// really bad naive kernel with atomicAdd
void encoder_backward_kernel(sycl::nd_item<1> id, floatX* dwte, floatX* dwpe,
                                        const floatX* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = id.get_global_id(0);
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const floatX* dout_btc = dout + b * T * C + t * C + c;
        floatX* dwte_ix = dwte + ix * C + c;
        floatX* dwpe_tc = dwpe + t * C + c;

        atomicAddX(dwte_ix, (floatX)*dout_btc);
        atomicAddX(dwpe_tc, (floatX)*dout_btc);
    }
}

void layernorm_forward_kernel3(sycl::nd_item<1> id, floatX* __restrict__ out, floatX* __restrict__ mean, floatX* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    const floatX* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= N) { return; } // guard

    // the row of input that this group of threads is responsible for
    const floatX* x = inp + idx * C;

    // mean
    float sum = 0.0f;
   for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        sum += (float)x[i];
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{}); 
    float m = sum / C;
     if(warp.leader() && mean != nullptr) {
        // Fix later
        //__stcs(mean + idx, (floatX)m);
        mean[idx] = (floatX)m;
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        float diff = (float)x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if(warp.leader() && rstd != nullptr) {
        // Fix later
        // __stcs(rstd + idx, (floatX)s);
        rstd[idx] = (floatX)s;
    }

    // final normalization and scaling by weight/bias
    floatX* o = out + idx * C;
    for (int c = warp.get_local_linear_id(); c < C; c += warp.get_max_local_range()[0]) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        //float n = s * ((float)__ldcs(x+c) - m);
        float n = s * ((float)x[c] - m);
        //__stcs(o+c, (floatX)(n * (float)weight[c] + (float)bias[c]));
         o[c] = (floatX)(n * (float)weight[c] + (float)bias[c]);
    }
}

// inputs floatX, outputs FP32 (for current FP32-only activation path for this WIP)
void permute_kernel(sycl::nd_item<1> id, floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = id.get_global_id(0);
    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

void permute_kernel_backward(sycl::nd_item<1> id, floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    int idx = id.get_global_id(0);
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}

void unpermute_kernel(sycl::nd_item<1> id, floatX* inp, floatX *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

    int idx = id.get_global_id(0);;
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

void unpermute_kernel_backward(sycl::nd_item<1> id, floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    int idx = id.get_global_id(0);
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = (floatX)dout[other_idx];
    }
}

void softmax_forward_kernel5(sycl::nd_item<1> id, floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    sycl::sub_group warp = id.get_sub_group();
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (id.get_group_range(0) - id.get_group(0) - 1) * warp.get_group_linear_range() + warp.get_group_linear_id(); // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = warp.get_local_linear_id(); i < pos_by_4; i += warp.get_max_local_range()[0]) {
        float regarray[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, regarray[k]);
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += sycl::exp(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + warp.get_local_linear_id() <= own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(maxval, (float)x[4*pos_by_4 + warp.get_local_linear_id()]);
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::exp(inv_temperature * ((float)x[4*pos_by_4 + warp.get_local_linear_id()] - maxval));
    }

    float global_maxval = sycl::reduce_over_group(warp, maxval, sycl::maximum<float>{});
    sumval *= sycl::exp(inv_temperature * (maxval - global_maxval));

    float sum = sycl::reduce_over_group(warp, sumval, sycl::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.get_local_linear_id(); i <= own_pos; i += warp.get_max_local_range()[0]) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = sycl::exp(inv_temperature * ((float)x[i] - global_maxval));
        out[idx * T + i] = (floatX)(ev * norm);
    }
}

void residual_forward_kernel(sycl::nd_item<1> id, floatX* out, floatX* inp1, floatX* inp2, int N) {
    int idx = id.get_global_id(0);
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward_kernel2(sycl::nd_item<1> id, floatX* out, const floatX* inp, int N) {
    int i = (id.get_global_id(0)) * x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + i, packed_out);
    }
}

void gelu_backward_kernel(sycl::nd_item<1> id, floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    int i = id.get_global_id(0);
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float coshf_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
void matmul_backward_bias_kernel4(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC,
                                  sycl::local_accessor<float> lmem) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    float *smem = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // of size block_size (128)
    sycl::sub_group warp = id.get_sub_group();
    const int warpSize = warp.get_max_local_range()[0]; 
    const int warp_id = warp.get_group_linear_id(); // warp index in the block, 0,1,2,3
    const int lane_id = warp.get_local_linear_id(); // thread index in the warp, 0,1,2,...,31
    const int tl = id.get_group(0) * warpSize; // pointer to the start column for this block
    const int vstep = warp.get_group_linear_range(); // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const floatX* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += (float)dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    sycl::group_barrier(id.get_group());

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] = (floatX)dout_sum;
    }
}

// single FP32 scratchpad shared by all the threadblocks (based on kernels 3 & 5)
void layernorm_backward_kernel6(sycl::nd_item<1> id, floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C, sycl::local_accessor<float> lmem) {
    float* shared = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();; // size = 2 * C + 1

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    int base_idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = id.get_local_linear_id(); i < C; i  += id.get_local_range(0)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    sycl::group_barrier(block); 

    int warps_in_grid = id.get_group_range(0) * warp.get_group_linear_range();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warp.get_local_linear_id(); i < C; i  += warp.get_max_local_range()[0]) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(warp, dnorm_mean, sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(warp, dnorm_norm_mean, sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
            // Fix this later
            // float dout_i = (float)__ldcs(&dout_bt[i]);
            // float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dout_i = (float)dout_bt[i];
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dbias_ref(dbias_shared[i]);
            dbias_ref += dout_i;
            // gradient contribution to weight
            sycl::atomic_ref<float, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device> dweight_ref(dweight_shared[i]);
            dweight_ref += norm_bti * dout_i;
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (floatX)((float)dinp_bt[i] + dval);
        }
    }

    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    // todo - could potentially avoid the extra copy if floatX is FP32, fairly negligible though
    sycl::group_barrier(block);
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dbias_ref(scratch_dbias[i]);
        dbias_ref += dbias_shared[i];
        sycl::atomic_ref<float, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> dweight_ref(scratch_dweight[i]);
        dweight_ref += dweight_shared[i];
    }
     sycl::group_barrier(block);
    if (block.leader()) {
        sycl::atomic_ref<uint, 
                         sycl::memory_order::relaxed, 
                         sycl::memory_scope::device> flag_ref(*scratchFlag);
        *tmp_flag = flag_ref.fetch_add(1);
    }
     sycl::group_barrier(block);
    if (*tmp_flag == id.get_group_range(0)-1) {
        for(int i = id.get_local_linear_id(); i < C; i+= id.get_local_range(0)) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (floatX)scratch_dbias[i];
            dweight[i] = (floatX)scratch_dweight[i];
        }
    }
}

void softmax_autoregressive_backward_kernel(sycl::nd_item<2> id, 
                                            // sycl::multi_ptr<float[32], sycl::access::address_space::local_space> lmem,
                                            floatX* dpreatt, const floatX* datt, const floatX* att,
                                            int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    // Can nix the block_acc if we're not using it?
    // float* block_acc = (float*) lmem.get_raw();

    int idx = id.get_group(0);
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*id.get_group(1);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    /*
    if (warp.get_group_linear_id() == 0) {
        block_acc[warp.get_local_linear_id()] = 0;
    }
    */

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.get_local_linear_id(); t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        /*
        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});
        */
        local_sum = sycl::reduce_over_group(block, local_sum, sycl::plus<float>{});

        for (int t3 = block.get_local_linear_id(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            // Fix this later
            // float acc = (float)__ldcs(att_bth + t3) * ((float)__ldcs(datt_bth + t3) - local_sum);
            // __stcs(dpreatt_bth + t3, (floatX)(scale * acc));
            float acc = (float)att_bth[t3] * ((float)datt_bth[t3] - local_sum);
            dpreatt_bth[t3] = (floatX)scale * acc;
        }
    }
}

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
inline float lerp(float start, float end, float weight) {
    return sycl::mad(weight, end, sycl::mad(-weight, start, start));
}

// Termplate type T instead of floatx
template <typename Tp, typename Tg>
void adamw_kernel3(sycl::nd_item<1> id, Tp* params_memory, float* master_params, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              unsigned int seed) {
   int i = id.get_global_id(0);
   if (i >= num_parameters) return;  // guard
   float grad = (float)grads_memory[i];
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
   // update the parameters (weight/bias)
   float old_param = master_params != NULL ? master_params[i] : (float)params_memory[i];
   float param = old_param - (learning_rate * (m / (sycl::sqrt(v) + eps) + weight_decay * old_param));
   // if we have master parameters, directly update the two weight copies
    if (master_params != NULL) {
        params_memory[i] = (floatX)param; // low-precision copy, for use in the forward pass
        master_params[i] = param; // float copy, for use in the next parameter update
    } else {
        // without a master copy of params in float, do a direct update in low precision
        // and use stochastic rounding to mitigate loss of training stability
        unsigned int random = Get2dNoiseUint(id.get_local_linear_id(), id.get_group(0), seed);
        stochastic_rounding(id, param, &params_memory[i], random);
    }
}

struct SoftmaxParams {
    float Scale;
    float Offset;
};

SoftmaxParams prepare_softmax_blockwide_nofloat4(sycl::nd_item<1> id,
                                                   int idx, const floatX* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    sycl::group block = id.get_group();

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + id.get_local_linear_id() - id.get_local_range(0); i >= 0; i -= id.get_local_range(0)) {
        float v = (float)x[i];
        float old_maxval = thread_maxval;
        thread_maxval = sycl::fmax(thread_maxval, v);
        thread_sumval *= sycl::exp((old_maxval - thread_maxval));
        thread_sumval += sycl::exp(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // Using the SYCL group algorithms as it's a lot cleaner looking.

    float block_maxval = sycl::reduce_over_group(block, thread_maxval, -FLT_MAX, sycl::maximum<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::exp(thread_maxval - block_maxval);
    float block_sumval = sycl::reduce_over_group(block, thread_sumval, 0.0f, sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4 (see dev/cuda/classifier_fused.cu)
// will _update_ logits to logit gradients
void fused_classifier_kernel3(sycl::nd_item<1> id, floatX* logits, floatX* losses, floatX* probs,
                                         const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    int idx = id.get_group(0);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(id, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(id.get_group().leader()) {
        float prob = sycl::exp((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = (floatX)(-sycl::log(prob));
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = (dlosses != NULL) ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    // note that we use the padded dimension P to access data, but we only ever
    // modify the elements up to V, ignoring the padded dimensions and leaving them at 0
    for (int i = id.get_local_linear_id(); i < V; i += id.get_local_range(0)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        // Fix this later
        // float v = (float)__ldcs(&logits_vec[i]);
        float v = (float)logits_vec[i];
        float prob = sycl::exp(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = (floatX)prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        logits[idx * P + i] = (floatX)((prob - indicator) * dloss);
    }
}

void copy_and_cast_kernel(sycl::nd_item<1> id, float* dst, const floatX* src, size_t n) {
    // a small kernel to copy and cast, i.e. `dst <- (float) src`
    const size_t i = id.get_global_id(0);
    if (i < n) { dst[i] = (float)src[i]; }
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(floatX* out,
                     int* inp, floatX* wte, floatX* wpe,
                     int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_forward_kernel2(id, out, inp, wte, wpe, B, T, C);
    });
}

void encoder_backward(floatX* dwte, floatX* dwpe,
                    const floatX* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_backward_kernel(id, dwte, dwpe, dout, inp, B, T, C);
    });
}

void layernorm_forward(floatX* out, floatX* mean, floatX* rstd,
                       floatX* inp, floatX* weight, floatX* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel3(id, out, mean, rstd, inp, weight, bias, N, C);
    });
}

// uses cuBLASLt to fuse the bias and gelu. does not work with OC = 50257 (last layer)
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);

    // Setup engine and stream
    auto &engine = *DefaultEngine;
    auto &stream = *DefaultStream;

    // Create memory descriptors
    dnnl::memory::data_type elt_type = dnnl::memory::data_type::f32;
    switch (PRECISION_MODE) {
        case PrecisionMode::PRECISION_FP32:
            elt_type = dnnl::memory::data_type::f32;
            break;
        case PrecisionMode::PRECISION_FP16:
            elt_type = dnnl::memory::data_type::f16;
            break;
        case PrecisionMode::PRECISION_BF16:
            elt_type = dnnl::memory::data_type::bf16;
            break;
        default:
            printf("Unsupported precision mode\n");
            exit(EXIT_FAILURE);
    }
    auto inp_md = dnnl::memory::desc({B*T, C}, elt_type, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({C, OC}, elt_type, dnnl::memory::format_tag::ba);
    auto out_md = dnnl::memory::desc({B*T, OC}, elt_type, dnnl::memory::format_tag::ab);
 
    // Create memory objects
    auto inp_mem = dnnlsycl::make_memory(inp_md, engine, dnnlsycl::memory_kind::usm, inp);
    auto weight_mem = dnnlsycl::make_memory(weight_md, engine, dnnlsycl::memory_kind::usm, weight);
    auto out_mem = dnnlsycl::make_memory(out_md, engine, dnnlsycl::memory_kind::usm, out);
        
    if (has_bias) {
        auto bias_md = dnnl::memory::desc({1, OC}, elt_type, dnnl::memory::format_tag::ab);
        auto bias_mem = dnnlsycl::make_memory(bias_md, engine, dnnlsycl::memory_kind::usm, bias);

        // Create primitive descriptor
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, bias_md, out_md);

        // Create primitive
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // Set arguments and execute
        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, inp_mem},
            {DNNL_ARG_WEIGHTS, weight_mem},
            {DNNL_ARG_BIAS, bias_mem},
            {DNNL_ARG_DST, out_mem}
        });
    } else {
        // Create primitive descriptor
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, out_md);

        // Create primitive
        auto matmul_prim = dnnl::matmul(matmul_pd);
    
        // Set arguments and execute
        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, inp_mem},
            {DNNL_ARG_WEIGHTS, weight_mem},
            {DNNL_ARG_DST, out_mem}
        });
    }
}

void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, T, NH, HS);
    });

    floatX* preatt = inp;

    dnnl::memory::data_type elt_type = dnnl::memory::data_type::f32;
    switch (PRECISION_MODE) {
        case PrecisionMode::PRECISION_FP32:
            elt_type = dnnl::memory::data_type::f32;
            break;
        case PrecisionMode::PRECISION_FP16:
            elt_type = dnnl::memory::data_type::f16;
            break;
        case PrecisionMode::PRECISION_BF16:
            elt_type = dnnl::memory::data_type::bf16;
            break;
        default:
            printf("Unsupported precision mode\n");
            exit(EXIT_FAILURE);
    }

    // Setup engine and stream
    auto &engine = *DefaultEngine;
    auto &stream = *DefaultStream;

    // Create memory descriptors
    auto q_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto k_md = dnnl::memory::desc({B * NH, HS, T}, elt_type, dnnl::memory::format_tag::acb);
    auto preatt_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto q_mem = dnnlsycl::make_memory(q_md, engine, dnnlsycl::memory_kind::usm, q);
    auto k_mem = dnnlsycl::make_memory(k_md, engine, dnnlsycl::memory_kind::usm, k);
    auto preatt_mem = dnnlsycl::make_memory(preatt_md, engine, dnnlsycl::memory_kind::usm, preatt);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, q_md, k_md, preatt_md);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC, q_mem},
        {DNNL_ARG_WEIGHTS, k_mem},
        {DNNL_ARG_DST, preatt_mem}
    });

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*softmax_block_size, softmax_block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel5(id, att, scale, preatt, B * NH, T);
    });

    // new approach: first cuBLAS another batched matmul
    floatX* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    auto att_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);  
    auto v_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto vaccum_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto att_mem = dnnlsycl::make_memory(att_md, engine, dnnlsycl::memory_kind::usm, att);
    auto v_mem = dnnlsycl::make_memory(v_md, engine, dnnlsycl::memory_kind::usm, v);
    auto vaccum_mem = dnnlsycl::make_memory(vaccum_md, engine, dnnlsycl::memory_kind::usm, vaccum);

    // Create primitive descriptor
    auto matmul_pd2 = dnnl::matmul::primitive_desc(engine, att_md, v_md, vaccum_md);

    // Create primitive
    auto matmul_prim2 = dnnl::matmul(matmul_pd2);
 
    // Set arguments and execute
    matmul_prim2.execute(stream, {
        {DNNL_ARG_SRC, att_mem},
        {DNNL_ARG_WEIGHTS, v_mem},
        {DNNL_ARG_DST, vaccum_mem}
    });
    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel(id, vaccum, out, B, T, NH, HS);
    });
}

void residual_forward(floatX* out, floatX* inp1, floatX* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        residual_forward_kernel(id, out, inp1, inp2, N);
    });
}

void gelu_forward(floatX* out, const floatX* inp, int N) {
    const int block_size = 512;
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_forward_kernel2(id, out, inp, N);
    });
}

void gelu_backward(floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_backward_kernel(id, dinp, inp, dout, N);
    });
}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     int B, int T, int C, int OC) {
    sycl::queue& queue = *DefaultQueue;                        
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    // Setup engine and stream
    auto &engine = *DefaultEngine;
    auto &stream = *DefaultStream;

    dnnl::memory::data_type elt_type = dnnl::memory::data_type::f32;
    switch (PRECISION_MODE) {
        case PrecisionMode::PRECISION_FP32:
            elt_type = dnnl::memory::data_type::f32;
            break;
        case PrecisionMode::PRECISION_FP16:
            elt_type = dnnl::memory::data_type::f16;
            break;
        case PrecisionMode::PRECISION_BF16:
            elt_type = dnnl::memory::data_type::bf16;
            break;
        default:
            printf("Unsupported precision mode\n");
            exit(EXIT_FAILURE);
    }

    // Create memory descriptors
    auto dout_md = dnnl::memory::desc({B*T, OC}, elt_type, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({OC, C}, elt_type, dnnl::memory::format_tag::ab);
    auto dinp_md = dnnl::memory::desc({B*T, C}, elt_type, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto dout_mem = dnnlsycl::make_memory(dout_md, engine, dnnlsycl::memory_kind::usm, dout);
    auto weight_mem = dnnlsycl::make_memory(weight_md, engine, dnnlsycl::memory_kind::usm, weight);
    auto dinp_mem = dnnlsycl::make_memory(dinp_md, engine, dnnlsycl::memory_kind::usm, dinp);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, dout_md, weight_md, dinp_md);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC, dout_mem},
        {DNNL_ARG_WEIGHTS, weight_mem},
        {DNNL_ARG_DST, dinp_mem}
    });
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    // Create memory descriptors
    auto dout_md2 = dnnl::memory::desc({OC, B*T}, elt_type, dnnl::memory::format_tag::ba);
    auto inp_md = dnnl::memory::desc({B*T, C}, elt_type, dnnl::memory::format_tag::ab);
    auto dweight_md = dnnl::memory::desc({OC, C}, elt_type, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto dout_mem2 = dnnlsycl::make_memory(dout_md2, engine, dnnlsycl::memory_kind::usm, dout);
    auto inp_mem = dnnlsycl::make_memory(inp_md, engine, dnnlsycl::memory_kind::usm, inp);
    auto dweight_mem = dnnlsycl::make_memory(dweight_md, engine, dnnlsycl::memory_kind::usm, dweight);

    dnnl::post_ops po;
    po.append_sum(one);

    dnnl::primitive_attr matmul_attr2;
    matmul_attr2.set_post_ops(po);

    // Create primitive descriptor
    auto matmul_pd2 = dnnl::matmul::primitive_desc(engine, dout_md2, inp_md, dweight_md, matmul_attr2);

    // Create primitive 
    auto matmul_prim2 = dnnl::matmul(matmul_pd2);

    // Set arguments and execute
    matmul_prim2.execute(stream, {
        {DNNL_ARG_SRC, dout_mem2},
        {DNNL_ARG_WEIGHTS, inp_mem},
        {DNNL_ARG_DST, dweight_mem}
    });

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        const int block_size = 512; // can't do 1024
        const int grid_size = OC / 32; // for now, OC must be divisible by 32 for this kernel to work
        DefaultQueue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float> lmem(block_size, h);
            h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
                matmul_backward_bias_kernel4(id, dbias, dout, B, T, OC, lmem);
            });
        });
    }
}

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

void layernorm_backward(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int grid_size = 1 * get_num_CUs();
    size_t shared_mem_size = (2 * C + 1) * sizeof(float);
    DefaultQueue->memset(scratch, 0, (2 * C + 1) * sizeof(float)); // todo - memset in parallel with previous kernels using streams
    DefaultQueue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> lmem(shared_mem_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            layernorm_backward_kernel6(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, lmem);
        });
    });
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(floatX* dinp, floatX* dqkvr, floatX* dpreatt, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH) {
    sycl::queue& queue = *DefaultQueue;     
    const int block_size = 256;
    int HS = C / NH; // head size

    // FP16 alpha/beta need to be used if and only if CUBLAS_COMPUTE_16F
    const float alpha = 1.0f, beta = 0.0f;
    dnnl::memory::data_type elt_type = dnnl::memory::data_type::f32;
    switch (PRECISION_MODE) {
        case PrecisionMode::PRECISION_FP32:
            elt_type = dnnl::memory::data_type::f32;
            break;
        case PrecisionMode::PRECISION_FP16:
            elt_type = dnnl::memory::data_type::f16;
            break;
        case PrecisionMode::PRECISION_BF16:
            elt_type = dnnl::memory::data_type::bf16;
            break;
        default:
            printf("Unsupported precision mode\n");
            exit(EXIT_FAILURE);
    }

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel_backward(id, scratch, dout, B, T, NH, HS);
    });
    
    // backward into datt
    // Batched matrix multiply with oneDNN
    // Setup engine and stream
    auto &engine = *DefaultEngine;
    auto &stream = *DefaultStream;

    // Create memory descriptors
    auto scratch_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto v_md = dnnl::memory::desc({B * NH, HS, T}, elt_type, dnnl::memory::format_tag::acb);
    auto datt_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto scratch_mem = dnnlsycl::make_memory(scratch_md, engine, dnnlsycl::memory_kind::usm, scratch);
    auto v_mem = dnnlsycl::make_memory(v_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX *>(v));
    auto datt_mem = dnnlsycl::make_memory(datt_md, engine, dnnlsycl::memory_kind::usm, datt);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, scratch_md, v_md, datt_md);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC, scratch_mem},
        {DNNL_ARG_WEIGHTS, v_mem},
        {DNNL_ARG_DST, datt_mem}
    });
      
    // backward into dv
    // Create memory descriptors
    auto att_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::acb);
    // scratch_md is already defined
    auto dv_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto att_mem = dnnlsycl::make_memory(att_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX*>(att));
    // scratch_mem is already defined
    auto dv_mem = dnnlsycl::make_memory(dv_md, engine, dnnlsycl::memory_kind::usm, dv);

    // Create primitive descriptor
    auto matmul_pd2 = dnnl::matmul::primitive_desc(engine, att_md, scratch_md, dv_md);

    // Create primitive
    auto matmul_prim2 = dnnl::matmul(matmul_pd2);
 
    // Set arguments and execute
    matmul_prim2.execute(stream, {
        {DNNL_ARG_SRC, att_mem},
        {DNNL_ARG_WEIGHTS, scratch_mem},
        {DNNL_ARG_DST, dv_mem}
    });
    
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    DefaultQueue->parallel_for(sycl::nd_range<2>(sycl::range<2>(B * NH, (T / 4) * 256),
                                                 sycl::range<2>(1, 256)), [=](sycl::nd_item<2> id) {
        softmax_autoregressive_backward_kernel(id, dpreatt, datt, att, B, T, C, scale);
    });
    
    // backward into q
    auto dpreatt_md = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::abc);  
    auto k_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto dq_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto dpreatt_mem = dnnlsycl::make_memory(dpreatt_md, engine, dnnlsycl::memory_kind::usm, dpreatt);
    auto k_mem = dnnlsycl::make_memory(k_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX*>(k));
    auto dq_mem = dnnlsycl::make_memory(dq_md, engine, dnnlsycl::memory_kind::usm, dq);

    // Create primitive descriptor
    auto matmul_pd3 = dnnl::matmul::primitive_desc(engine, dpreatt_md, k_md, dq_md);

    // Create primitive
    auto matmul_prim3 = dnnl::matmul(matmul_pd3);
 
    // Set arguments and execute
    matmul_prim3.execute(stream, {
        {DNNL_ARG_SRC, dpreatt_mem},
        {DNNL_ARG_WEIGHTS, k_mem},
        {DNNL_ARG_DST, dq_mem}
    });
    
    // backward into k
    auto dpreatt_md2 = dnnl::memory::desc({B * NH, T, T}, elt_type, dnnl::memory::format_tag::acb);
    auto q_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);
    auto dk_md = dnnl::memory::desc({B * NH, T, HS}, elt_type, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto dpreatt_mem2 = dnnlsycl::make_memory(dpreatt_md2, engine, dnnlsycl::memory_kind::usm, dpreatt);
    auto q_mem = dnnlsycl::make_memory(q_md, engine, dnnlsycl::memory_kind::usm, const_cast<floatX*>(q));
    auto dk_mem = dnnlsycl::make_memory(dk_md, engine, dnnlsycl::memory_kind::usm, dk);

    // Create primitive descriptor
    auto matmul_pd4 = dnnl::matmul::primitive_desc(engine, dpreatt_md2, q_md, dk_md);

    // Create primitive
    auto matmul_prim4 = dnnl::matmul(matmul_pd4);
 
    // Set arguments and execute
    matmul_prim4.execute(stream, {
        {DNNL_ARG_SRC, dpreatt_mem2},
        {DNNL_ARG_WEIGHTS, q_mem},
        {DNNL_ARG_DST, dk_mem}
    });
    
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    DefaultQueue->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel_backward(id, dinp, dq, dk, dv, B, T, NH, HS);
    });
}

// replaces logits with logit gradients
template <typename Type>
void fused_classifier3(Type* logits, Type* losses,
                      const Type* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    // can't do 1024
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = N;
    DefaultQueue->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id) {
        fused_classifier_kernel3(id, logits, losses, (Type*)NULL, dlosses, targets, B, T, V, P);
    });
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
constexpr const int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte; // (V, C)
    floatX* wpe; // (maxT, C)
    floatX* ln1w; // (L, C)
    floatX* ln1b; // (L, C)
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w; // (L, C)
    floatX* ln2b; // (L, C)
    floatX* fcw; // (L, 4*C, C)
    floatX* fcb; // (L, 4*C)
    floatX* fcprojw; // (L, C, 4*C)
    floatX* fcprojb; // (L, C)
    floatX* lnfw; // (C)
    floatX* lnfb; // (C)
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb

    // populate the parameter sizes in bytes (all the same for now, keeping for future use)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

// allocate memory for the parameters and point the individual tensors to the right places
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    // calculate the total number of parameters and bytes across all tensors
    size_t num_parameters = 0;
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_elements[i];
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }
    // malloc all parameters all at once on the device
    void* params_memory;
    syclMallocCheck(params_memory = sycl::malloc_device(num_parameters_bytes, *DefaultQueue));
    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params->wte, &params->wpe, (floatX**)&params->ln1w, (floatX**)&params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, (floatX**)&params->ln2w, (floatX**)&params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, (floatX**)&params->lnfw, (floatX**)&params->lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    floatX* encoded; // (B, T, C)
    floatX* ln1; // (L, B, T, C)
    floatX* ln1_mean; // (L, B, T)
    floatX* ln1_rstd; // (L, B, T)
    floatX* atty; // (L, B, T, C)
    floatX* att; // (L, B, NH, T, T) (smaller with cuDNN)
    floatX* attproj; // (L, B, T, C)
    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    floatX* ln2_mean; // (L, B, T)
    floatX* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* fcproj; // (L, B, T, C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C)
    floatX* lnf_mean; // (B, T)
    floatX* lnf_rstd; // (B, T)
    floatX* losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    floatX* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    floatX* output;
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, size_t B, size_t T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * C; // atty
    #ifdef ENABLE_CUDNN
    // FP32 stats tensor for cuDNN to be passed to backward pass
    act_sizes[5] = L * B * NH * T * (sizeof(float) / sizeof(floatX));
    #else
    act_sizes[5] = L * B * NH * T * T; // att
    #endif
    act_sizes[6] = L * B * T * C; // attproj
    act_sizes[7] = L * B * T * C; // residual2
    act_sizes[8] = L * B * T * C; // ln2
    act_sizes[9] = L * B * T; // ln2_mean
    act_sizes[10] = L * B * T; // ln2_rstd
    act_sizes[11] = L * B * T * 4*C; // fch
    act_sizes[12] = L * B * T * 4*C; // fch_gelu
    act_sizes[13] = L * B * T * C; // fcproj
    act_sizes[14] = L * B * T * C; // residual3
    act_sizes[15] = B * T * C; // lnf
    act_sizes[16] = B * T; // lnf_mean
    act_sizes[17] = B * T; // lnf_rstd
    act_sizes[18] = B * T; // losses
    act_sizes[19] = L * B * T * 3*C; // qkvr
    act_sizes[20] = B * T * std::max(3*C, std::max(NH*T, Vp)); // output / scratch
}

// Backward pass is conceptually quite different from forward, because we can discard
// the activations of a layer as soon as we're done with it. This lets us aggressively
// reuse memory, so that we need far fewer tensors for backward state.
#ifdef ENABLE_CUDNN
#define NUM_BACKWARD_TENSORS 2
#else
#define NUM_BACKWARD_TENSORS 3
#endif

typedef struct {
    floatX* bt4c; // (B, T, 4*C)
    floatX* residual3; // (B, T, C)
    #ifndef ENABLE_CUDNN
    floatX* preatt; // (B, NH, T, T)
    #endif
} GradActTensors;

void fill_in_grad_act_sizes(size_t* act_sizes, size_t B, size_t T, GPT2Config config) {
    size_t C = config.channels;
    act_sizes[0] = B * T * 4 * C; // bt4c
    act_sizes[1] = B * T * C; // residual3

    #ifndef ENABLE_CUDNN
    size_t NH = config.num_heads;
    act_sizes[2] = B * NH * T * T; // preatt
    #endif
}

void* malloc_and_point(floatX** targets[], const size_t* act_sizes, size_t n) {
    size_t num_activations = 0;
    for (size_t i = 0; i < n; i++) {
        num_activations += act_sizes[i];
    }
    void* acts_memory;
    syclMallocCheck(acts_memory = sycl::malloc_device<floatX>(num_activations, *DefaultQueue));
    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = (floatX*)acts_memory_iterator;
        acts_memory_iterator += act_sizes[i] * sizeof(floatX);
    }
    return acts_memory;
}

void* malloc_and_point_activations(ActivationTensors* acts, const size_t* act_sizes) {
    floatX** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
    };
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

void* malloc_and_point_backward(GradActTensors* acts, const size_t* act_sizes) {
    floatX** ptrs[] = {
        &acts->bt4c, &acts->residual3,
        #ifndef ENABLE_CUDNN
        &acts->preatt,
        #endif
    };
    return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    // gradients of the weights
    ParameterTensors grads;
    void* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    float* master_weights;     // is NULL unless fp32 weights is enabled.
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    size_t num_activations;
    // gradients of the activations
    GradActTensors grads_acts;
    size_t num_grad_acts;
    void* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    float accumulated_mean_loss; // Mean loss after aggregating it on all GPUs
    floatX* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
    unsigned long long rng_state; // the RNG state for seeding stochastic rounding etc.
    int use_master_weights;
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    if (PRECISION_MODE == PRECISION_FP16) {
        // TODO for later perhaps, would require us dynamically converting the
        // model weights from fp32 to fp16 online, here in this function, or writing
        // the fp16 weights directly from Python, which we only do for fp32/bf16 atm.
        fprintf(stderr, "build_from_checkpoint() does not support fp16 right now.\n");
        exit(EXIT_FAILURE);
    }

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        // 3 = fp32, padded vocab
        // 5 = bf16, padded vocab, layernorms also in bf16
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);

    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);

    // read in all the parameters from file and copy them to device
    float* params_memory_cpu = (float*)mallocCheck(model->num_parameters_bytes);
    freadCheck(params_memory_cpu, 1, model->num_parameters_bytes, model_file);
    DefaultQueue->memcpy(model->params_memory, params_memory_cpu, model->num_parameters_bytes).wait();
    free(params_memory_cpu);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->master_weights = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
    model->rng_state = 13371337;
    model->use_master_weights = 1; // keep master weights copy in float for optim update?
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    // targets are optional and could be NULL
    // in this function we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf0("allocated %d MiB for activations\n", (int)round(num_activations * sizeof(floatX) / (1024 * 1024)));
        // also create memory for caching inputs and targets
        syclMallocCheck(model->inputs = sycl::malloc_device<int>(B * T, *DefaultQueue));
        syclMallocCheck(model->targets = sycl::malloc_device<int>(B * T, *DefaultQueue));
        syclMallocCheck(model->cpu_losses = sycl::malloc_host<floatX>(B * T, *DefaultQueue));
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs/targets to the model
    DefaultQueue->memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        DefaultQueue->memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    floatX* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_ln1b = params.ln1b + l * C;
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = acts.ln1 + l * B * T * C;
        floatX* l_ln1_mean = acts.ln1_mean + l * B * T;
        floatX* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_attproj = acts.attproj + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = acts.ln2 + l * B * T * C;
        floatX* l_ln2_mean = acts.ln2_mean + l * B * T;
        floatX* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        floatX* l_fcproj = acts.fcproj + l * B * T * C;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);

        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        floatX* scratch = (floatX*)acts.output;
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        #endif

        matmul_forward_cublaslt(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward_cublaslt(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward_cublaslt(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        fused_classifier3(acts.output, acts.losses, (floatX*)NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        DefaultQueue->memcpy(model->cpu_losses, acts.losses, B * T * sizeof(floatX)).wait();
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += (float)(model->cpu_losses[i]); }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;

    } else {
        // if we don't have targets, we don't have loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if (model->grads_acts_memory != NULL) { DefaultQueue->memset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(floatX)); }
    if (model->grads_memory != NULL) { DefaultQueue->memset(model->grads_memory, 0, model->num_parameters * sizeof(floatX));}
}

void gpt2_backward(GPT2 *model) {
    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(EXIT_FAILURE);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        // allocate buffers for weight gradients
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);
        printf0("allocated %d MiB for parameter gradients\n", (int)round(model->num_parameters * sizeof(floatX) / (1024 * 1024)));
        // we're going to be clever for the activations backward pass. we don't need to exactly
        // mirror the forward pass activations and we will save memory.
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len, model->config);
        // count up and allocate the space
        model->grads_acts_memory = malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
        model->num_grad_acts = 0;
        for (size_t i = 0; i < NUM_BACKWARD_TENSORS; i++) {
            model->num_grad_acts += bw_act_sizes[i];
        }
        printf0("allocated %d MiB for activation gradients\n", (int)round(model->num_grad_acts * sizeof(floatX) / (1024 * 1024)));
        // init gradients of parameters and activations to zero
        gpt2_zero_grad(model);
    }

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    GradActTensors grads_acts = model->grads_acts;

    // re-use the output buffer of the forward pass as a scratchpad during backward pass
    float*  scratchF = (float*)acts.output;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
    // backward the final layernorm
    floatX* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    floatX* dresidual = (floatX*)grads_acts.residual3; // the main buffer holding the gradient in the backward pass
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, grads_acts.bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        floatX* dl_ln1w = grads.ln1w + l * C;
        floatX* dl_ln1b = grads.ln1b + l * C;
        floatX* dl_qkvw = grads.qkvw + l * 3*C * C;
        floatX* dl_qkvb = grads.qkvb + l * 3*C;
        floatX* dl_attprojw = grads.attprojw + l * C * C;
        floatX* dl_attprojb = grads.attprojb + l * C;
        floatX* dl_ln2w = grads.ln2w + l * C;
        floatX* dl_ln2b = grads.ln2b + l * C;
        floatX* dl_fcw = grads.fcw + l * 4*C * C;
        floatX* dl_fcb = grads.fcb + l * 4*C;
        floatX* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        floatX* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        floatX* l_ln1 = acts.ln1 + l * B * T * C;
        floatX* l_ln1_mean = acts.ln1_mean + l * B * T;
        floatX* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = acts.ln2 + l * B * T * C;
        floatX* l_ln2_mean = acts.ln2_mean + l * B * T;
        floatX* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        // we need a B x T x C buffer; thankfully, the forward activation for lnf isn't needed anymore,
        // so we can co-opt it here.
        floatX* dl_btc = (floatX*)acts.lnf;
        floatX* dl_bt4c = (floatX*)grads_acts.bt4c;

        // backprop this layer
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);

        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, (float*)l_att, B, T, NH, C);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        // we need B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        floatX* buffer_a = l_atty;
        floatX* buffer_b = l_fch;        // this is B x T x 4C, so even larger than what we need
        floatX* dl_preatt = (floatX*)grads_acts.preatt; // dedicated scratchpad allocation
        floatX* scratchX =  (floatX*)acts.output;
        attention_backward(dl_bt4c, buffer_b, dl_preatt, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
        #endif

        // QKV parameter gradients
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
}

// Compute a mean of a single CPU value across all GPU processes. No-op when multi-GPU is disabled.
float multi_gpu_cpu_float_mean(float value, const MultiGpuConfig* multi_gpu_config) {
#ifdef MULTI_GPU
    // MPI doesn't support all reduce with mean, so we sum up, then divide.
    float result;
    mpiCheck(MPI_Allreduce(&value, &result, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    return result / multi_gpu_config->num_processes;
#else
    return value;
#endif
}

// Averages out the loss and gradients across all GPUs. No-op when multi-GPU is disabled.
// todo - this version only works if all the parameters are the same size (floatX)
void gpt2_multi_gpu_accumulate(GPT2* model, MultiGpuConfig* multi_gpu_config) {
    // Average all losses.
    model->accumulated_mean_loss = multi_gpu_cpu_float_mean(model->mean_loss, multi_gpu_config);
#ifdef MULTI_GPU
    // Average all gradients.
    ncclCheck(ncclAllReduce(model->grads_memory, model->grads_memory,
        model->num_parameters,
        ncclFloatX, ncclAvg,
        multi_gpu_config->nccl_comm,
        // use 0 for default stream (all other computations use this stream)
        /*stream=*/0));
#endif
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        syclMallocCheck(model->m_memory = sycl::malloc_device<float>(model->num_parameters, *DefaultQueue));
        syclMallocCheck(model->v_memory = sycl::malloc_device<float>(model->num_parameters, *DefaultQueue));
        DefaultQueue->memset(model->m_memory, 0, model->num_parameters * sizeof(float));
        DefaultQueue->memset(model->v_memory, 0, model->num_parameters * sizeof(float));
        printf0("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf0("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
        if (model->use_master_weights == 1) {
            // allocate one more buffer to keep the master copy of weights as float, and copy the weights over
            syclMallocCheck(model->master_weights = sycl::malloc_device<float>(model->num_parameters, *DefaultQueue));
            // ok this is a porting gotcha - need to make sure we're not capturing host pointers in GPU code
            float *master_weights = model->master_weights;
            floatX *params_memory = (floatX*)model->params_memory;
            int num_parameters = model->num_parameters;
            DefaultQueue->parallel_for(sycl::nd_range<1>(CEIL_DIV(model->num_parameters, 512) * 512, 512), [=](sycl::nd_item<1> id) {
                copy_and_cast_kernel(id, master_weights, (floatX*)params_memory, num_parameters);
            });
            printf0("allocated %zu MiB for master copy of params\n", (model->num_parameters * sizeof(float)) >> 20);
        }
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    unsigned int seed = random_u32(&model->rng_state);
    // ok this is a porting gotcha (again) - need to make sure we're not capturing host pointers in GPU code
    floatX *params_memory = (floatX*)model->params_memory;
    floatX *grads_memory = (floatX*)model->grads_memory;
    float *m_memory = model->m_memory;
    float *v_memory = model->v_memory;
    float *master_weights = model->master_weights;
    int num_parameters = model->num_parameters;
    DefaultQueue->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        adamw_kernel3(id, (floatX*)params_memory, master_weights,
                      (floatX*)grads_memory, m_memory, v_memory,
                      num_parameters,
                      learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, seed);
    });
}

void gpt2_free(GPT2 *model) {
    sycl::free(model->params_memory, *DefaultQueue);
    sycl::free(model->grads_memory, *DefaultQueue);
    sycl::free(model->m_memory, *DefaultQueue);
    sycl::free(model->v_memory, *DefaultQueue);
    sycl::free(model->master_weights, *DefaultQueue);
    sycl::free(model->acts_memory, *DefaultQueue);
    sycl::free(model->grads_acts_memory, *DefaultQueue);
    sycl::free(model->inputs, *DefaultQueue);
    sycl::free(model->targets, *DefaultQueue);
    sycl::free(model->cpu_losses, *DefaultQueue);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite: returns random batches of data from a file of integers

typedef struct {
    // Distributed data parallel specifics.
    // Each worker loads it's own chunk of data.
    int process_rank;
    int num_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B;
    size_t T;
    // input handling and its state
    FILE* tokens_file;
    long file_size;
    long current_position;
    // output memory
    int* batch;
    int* inputs;
    int* targets;
    // convenience variables
    size_t num_batches;
} DataLoader;

void dataloader_init(DataLoader *loader, const MultiGpuConfig* multi_gpu_config, const char* filename, size_t B, size_t T) {
    loader->process_rank = multi_gpu_config->process_rank;
    loader->num_processes = multi_gpu_config->num_processes;
    loader->B = B;
    loader->T = T;

    // open the input file for reading
    loader->tokens_file = fopenCheck(filename, "rb");

    // determine the file size
    fseekCheck(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseekCheck(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(EXIT_FAILURE);
    }
    loader->current_position = loader->process_rank * B * T * sizeof(int); // start at the beginning

    // allocate space for B*T + 1 integers to store the inputs and targets
    // Using CUDA CPU pinned memory for faster PCI Express transfers to GPU
    // See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    // Note: we may want to do something different here for Intel GPUs.
    loader->batch = sycl::malloc_host<int>(B * T + 1, *DefaultQueue);
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1; // targets are shifted by one
    // note: we definitely want to advance by B * T; That is the "stride" by which we move
    // the window of tokens. We only load B * T + 1 tokens because our targets are offset by 1
    loader->num_batches = loader->file_size / (loader->num_processes * B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (loader->num_processes * B * T + 1) * sizeof(int) > loader->file_size) {
        loader->current_position = loader->process_rank * B * T * sizeof(int);
    }
    // read the B*T+1 integers from the file into batch
    fseekCheck(loader->tokens_file, loader->current_position, SEEK_SET);
    freadCheck(loader->batch, sizeof(int), B*T+1, loader->tokens_file);
    // advance the current position by B*T*num_processes integers
    // note: the "stride" of tokens by which we move each time is definitely B * T
    loader->current_position += loader->num_processes * B * T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
    fcloseCheck(loader->tokens_file);
    sycl::free(loader->batch, *DefaultQueue);
}

// ----------------------------------------------------------------------------
// sampler: takes probabilities and samples integers from them

int sample_softmax(const float* logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Logger lite, will probably grow/change some over time

typedef struct {
    FILE *logfile;
    int flush_every; // every how many steps to flush the log
} Logger;

void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) { logger->logfile = fopenCheck(filename, "w"); }
}

void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0) { fflush(logger->logfile); }
    }
}

void logger_free(Logger *logger) {
    if (logger->logfile != NULL) { fclose(logger->logfile); }
}

// ----------------------------------------------------------------------------
// CLI, poor man's argparse

void error_usage() {
    // default run = debugging run with TinyShakespeare
    // bigger run = train on TinyStories! e.g. val/sample less often, but sample more tokens, write to logfile
    fprintf(stderr, "Usage:   ./train_gpt2cu [options]\n");
    fprintf(stderr, "Example: ./train_gpt2cu -i data/TinyStories -v 100 -s 100 -g 144 -o stories.log\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> input dataset prefix (default = data/tiny_shakespeare)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_batches, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -a <int>    overfit a single batch? 0/1. useful for debugging\n");
    fprintf(stderr, "  -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)\n");
    fprintf(stderr, "  -w <int>    keep f32 copy of weights for the optimizer? (default: 1)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {
    multi_gpu_config = multi_gpu_config_init(&argc, &argv);

    // read in the (optional) command line arguments
    const char* input_dataset_prefix = "data/tiny_shakespeare"; // or e.g. data/TinyStories
    const char* output_log_file = NULL;
    int B = 4; // batch size
    int T = 1024; // sequence length max
    float learning_rate = 3e-4f;
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_batches = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    int overfit_single_batch = 0; // useful for debugging, 1 = only load a single data batch once
    int max_steps = -1;
    int override_enable_tf32 = 1;
    int use_master_weights = 1;
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'i') { input_dataset_prefix = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); } // Per-GPU batch size
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'x') { max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_batches = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'a') { overfit_single_batch = atoi(argv[i+1]); }
        else if (argv[i][1] == 'f') { override_enable_tf32 = atoi(argv[i+1]); }
        else if (argv[i][1] == 'w') { use_master_weights = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| Parameter             | Value                                              |\n");
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| input dataset prefix  | %-50s |\n", input_dataset_prefix);
    printf0("| output log file       | %-50s |\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf0("| batch size B          | %-50d |\n", B);
    printf0("| sequence length T     | %-50d |\n", T);
    printf0("| learning rate         | %-50e |\n", learning_rate);
    printf0("| max_steps             | %-50d |\n", max_steps);
    printf0("| val_loss_every        | %-50d |\n", val_loss_every);
    printf0("| val_max_batches       | %-50d |\n", val_max_batches);
    printf0("| sample_every          | %-50d |\n", sample_every);
    printf0("| genT                  | %-50d |\n", genT);
    printf0("| overfit_single_batch  | %-50d |\n", overfit_single_batch);
    printf0("| use_master_weights    | %-50s |\n", use_master_weights ? "enabled" : "disabled");
    printf0("+-----------------------+----------------------------------------------------+\n");

    // set up the device
    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{} /*,
                             sycl::property::queue::enable_profiling{} */ });
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }
    DefaultQueue = &defaultQueue;
    // Setup oneDNN engine and stream
    auto engine = dnnl::sycl_interop::make_engine(DefaultQueue->get_device(), DefaultQueue->get_context());
    auto stream = dnnl::sycl_interop::make_stream(engine, *DefaultQueue);
    DefaultEngine = &engine;
    DefaultStream = &stream;

    // setup compute precision settings for cublas
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = 0;
    
    // set up cuDNN
    #ifdef ENABLE_CUDNN
    checkCudnnErr(cudnnCreate(&cudnn_handle));
    #endif

    printf0("| device                | %-50s |\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    printf0("| TF32                  | %-50s |\n", enable_tf32 ? "enabled" : "disabled");
    printf0("| precision             | %-50s |\n", precision_mode_str);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, load_filename);
    model.use_master_weights = use_master_weights;
    printf0("| load_filename         | %-50s |\n", load_filename);
    printf0("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf0("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf0("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf0("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf0("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf0("| channels C            | %-50d |\n", model.config.channels);
    printf0("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // build DataLoaders for both train and val
    char train_tokens_filename[128];
    char val_tokens_filename[128];
    assert(strlen(input_dataset_prefix) < 100); // being bit lazy here, make sure we don't overflow
    // if we're only overfitting a single batch for debugging, let's overfit the first batch
    // from val instead of train split, because val is smaller and a bit faster
    const char* train_split = (overfit_single_batch == 1) ? "val" : "train";
    sprintf(train_tokens_filename, "%s_%s.bin", input_dataset_prefix, train_split);
    sprintf(val_tokens_filename, "%s_val.bin", input_dataset_prefix);
    DataLoader train_loader;
    dataloader_init(&train_loader, &multi_gpu_config, train_tokens_filename, B, T);
    DataLoader val_loader;
    dataloader_init(&val_loader, &multi_gpu_config, val_tokens_filename, B, T);
    int train_num_batches = (max_steps == -1) ? train_loader.num_batches : max_steps; // default = 1 epoch
    int val_num_batches = train_loader.num_batches < val_max_batches ? train_loader.num_batches : val_max_batches;
    printf0("| train_num_batches     | %-50d |\n", train_num_batches);
    printf0("| val_num_batches       | %-50d |\n", val_num_batches);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // pretty print in a table the multi-gpu configuration as well
    printf0("| num_processes         | %-50d |\n", multi_gpu_config.num_processes);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // more prints related to allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf0("num_parameters: %zu ==> bytes: %zu\n", model.num_parameters, model.num_parameters_bytes);
    printf0("allocated %d MiB for model parameters\n", (int)round(model.num_parameters_bytes / (1024 * 1024)));

    // set up the Logger
    Logger logger;
    logger_init(&logger, output_log_file);

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    // train
    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    for (int step = 0; step <= train_num_batches; step++) {
        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss
        if (step % val_loss_every == 0 || last_step) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            val_loss = multi_gpu_cpu_float_mean(val_loss, &multi_gpu_config);
            printf0("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        if (multi_gpu_config.process_rank == 0 && (step > 0 && (step % sample_every) == 0 || last_step)) {
            // fill up gen_tokens with the <|endoftext|> token, which kicks off the generation
            int eot_token = tokenizer.eot_token;
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                DefaultQueue->memcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX)).wait();
                // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
                for (int i = 0; i < model.config.vocab_size; i++) {
                    cpu_logits[i] = (float)cpu_logits_raw[i];
                }

                float coin = random_f32(&rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        if (overfit_single_batch == 0 || (step == 0 && overfit_single_batch == 1)) {
            // if we're overfitting a single batch, we'll only call this at step = 0
            dataloader_next_batch(&train_loader);
        }
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        if (multi_gpu_config.num_processes > 1) {
            gpt2_multi_gpu_accumulate(&model, &multi_gpu_config);
        }
        gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        DefaultQueue->wait(); // finish all SYCL work to get correct precise timings
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        if (step > 0) { // consider the first batch to be a warmup (e.g. cuBLAS/cuDNN initialisation)
            total_sum_iteration_time_s += time_elapsed_s;
        }
        int tokens_per_second = multi_gpu_config.num_processes * (B * T) / time_elapsed_s;
        float accumulated_loss = multi_gpu_config.num_processes == 1 ? model.mean_loss : model.accumulated_mean_loss;
        printf0("step %4d/%d: train loss %f (acc %f) (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss, accumulated_loss, time_elapsed_s * 1000, tokens_per_second);
        logger_log_train(&logger, step, model.mean_loss);
    }
    // add a total average, for optimizations that are only mild improvements (excluding 1st batch as warmup)
    printf0("total average iteration time: %f ms\n", total_sum_iteration_time_s / (train_num_batches-1) * 1000);

    // free and destroy everything
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(cpu_logits_raw);
    free(cpu_logits);
    free(gen_tokens);
    #ifdef ENABLE_CUDNN
    if (cudnn_workspace != NULL) { cudaCheck(cudaFree(cudnn_workspace)); }
    checkCudnnErr(cudnnDestroy(cudnn_handle));
    #endif
    logger_free(&logger);
    multi_gpu_config_free(&multi_gpu_config);

    return 0;
}
#endif
