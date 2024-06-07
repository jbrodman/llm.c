/*
Triangular matrix multiplication as in autoregressive attention. A short story.
by @ngc92

Compile:
nvcc -O3 --use_fast_math trimat_forward.cu -o trimat_forward -lcublas

Run:

cuBLAS baseline kernel
./trimat_forward 0

naive
./trimat_forward 1

registers
./trimat_forward 2

tri3
./trimat_forward 3

tri4
./trimat_forward 4
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include "common.h"

// namespace alias
namespace dnnlsycl = dnnl::sycl_interop;
namespace syclx = sycl::ext::oneapi;

static float* d_qkvr;   // scratch for the cublas kernel

/*                    ** Chapter I - Introduction **
 *
 *  You are Trimul. You've always wanted to do fast matrix multiplication, but they said
 *  "Don't bother, big dumb Cublas is much faster than you!"
 *  "I don't need to be faster than Cublas", you replied, "I can be smarter. Cublas calculates
 *  the entire matrix, but I need only half. If I'm more than half as fast as Cublas, I'm
 *  going to win."
 *
 *  So to prove everyone wrong, you enter the TriMatlon, the most prestigious competition
 *  for anyone paying Attention.
 *
 *  Before you start preparing, lets have a look at the players involved
 *
 *  First, there is the Referee (`trimul_cpu`), slow and ponderous, but producing results
 *  beyond reproof.
 *  Then, there is Cublas. Cublas' mind is so inflexible, it doesn't actually comprehend
 *  what we are trying to do here, so Cublas has brought an assistant (`permute_kernel`)
 *  that translates the competition into a task that it can solve. But once it recognizes
 *  the problem, its muscle memory kicks in, and matrix products are produced faster than
 *  the eye can see. Stuck in its routine, Cublas doesn't realize the task is already
 *  finished with the lower triangle, though.
 *
 *  If you can do without an assistant, and can solve the right task, then that's your opportunity
 *  to shine!
 */


// taken from then attention forward pass
void trimul_cpu(float* out, const float* inp,
                int B, int T, int C, int NH) {
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* out_bth = out + b * NH * T * T + h * T * T + t * T;

                // pass 1: calculate query dot key and maxval
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;

                    out_bth[t2] = val;
                }
                for(int t2 = t + 1; t2 < T; ++t2) {
                    out_bth[t2] = NAN;
                }
            }
        }
    }
}

void permute_kernel(sycl::nd_item<1> id, float* q, float* k, float* v,
                               const float* inp,
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

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}


void trimul_cublas(float* preatt,
                   const float* inp,
                   int B, int T, int C, int NH) {
    sycl::queue& queue = *DefaultQueue;
                
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float* q, * k, * v;
    q = d_qkvr + 0 * B * T * C;
    k = d_qkvr + 1 * B * T * C;
    v = d_qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, 256);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * 256, 256), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, T, NH, HS);
    });

    // batched matrix multiply with oneDNN
    const float alpha = 1.0f / sqrtf(HS); // Need to handle non 1.0f alpha
    const float beta = 0.0f;
 
    // Setup engine and stream
    auto engine = dnnlsycl::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnlsycl::make_stream(engine, queue);

    // Create memory descriptors
    auto q_md = dnnl::memory::desc({B * NH, T, HS}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
    auto k_md = dnnl::memory::desc({B * NH, HS, T}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::acb);
    auto preatt_md = dnnl::memory::desc({B * NH, T, T}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);

    // Create memory objects
    auto q_mem = dnnlsycl::make_memory(q_md, engine, dnnlsycl::memory_kind::usm, q);
    auto k_mem = dnnlsycl::make_memory(k_md, engine, dnnlsycl::memory_kind::usm, k);
    auto preatt_mem = dnnlsycl::make_memory(preatt_md, engine, dnnlsycl::memory_kind::usm, preatt);

    dnnl::post_ops po; 
    po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, beta);

    dnnl::primitive_attr matmul_attr;
    matmul_attr.set_post_ops(po);

    // Create primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, q_md, k_md, preatt_md, matmul_attr);

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);
 
    // Set arguments and execute
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC, q_mem},
        {DNNL_ARG_WEIGHTS, k_mem},
        {DNNL_ARG_DST, preatt_mem}
    });
}

/*                    ** Chapter II - Getting a Team **
 *
 *  OK, you've registered for the competition, now what to do. TriMatlon is a team competition, so first, you need
 *  to figure out what kind of team you need, and how to organize it. The individual instances and heads of the
 *  problem are completely independent, so you just can send separate teams to work there completely independently.
 *
 *  To figure out how to organize each team, you take out your spyglass (`Nsight Compute`) and look how the Cublas teams
 *  are handling their work.
 *  Turns out, you need 256 athletes in each group, and those handle 128 x 128 of the tasks. They work together in
 *  a tight square formation, 16 wide and 16 deep.
 *
 *  So, you went out and got your 100 000 friends, and split them into groups (`trimul_launcher`). Each group gets
 *  informed about where they should work (`trimul_global`) and goes off to do their thing (`matmul_tri_naive`).
 *  Let's observe how we're doing.
 */

// Oof - ok we can't use function pointers here.



/*                     ** Chapter III - ... **
 *
 *  You go over to the playing field. On one end of the field, there is a huge pile of funnily shaped cookie cutters.
 *  Some in the shape of animals, some in the shape of a landscape. Each group of workers has assigned some runners,
 *  fetching the cookie cutters for them. The workers seem very relaxing, chatting with each other, lounging about.
 *  You focus in on one of them.
 *
 *  He seems to be giving an instruction to a runner, and then turns back to reading a novel. The runner, meanwhile,
 *  crosses the field and back, handing him an elephant shape. Then she's off again to pick up a savannah background.
 *  Having received the two shapes, pressed them into the dough, and makes an elephant-in-the-savannah cookie. He hands
 *  the cutters back to the runner. "Can you please fetch me an elephant and a jungle next?"
 *  While she's on her way, he takes a sip off his cocktail.
 *  This time, she's making only one trip, keeping the elephant in her pocket (_Cache_). Still, it seems to take forever.
 *  You keep observing:
 *  - Elephant and zoo
 *  - Elephant and island
 *  ...
 *  - Lion and savannah
 *  - Lion and jungle
 *  - Lion and zoo
 *  ...
 *
 *  The worker has his poor runner fetch the same things over and over again, looking like she's about to faint from exhaustion.
 *  Even though she realizes this and always keeps one of them in her pocket, there is so much running,
 *  and little actual work happening.
 *
 *  Clearly, this isn't going to be effective, so you call a team meeting.
 */

// baseline implementation: 20 ms
void matmul_tri_naive(sycl::nd_item<3> id, float* p, int ps, const float* k, int ks, const float* q, int qs, int T, int hs, float alpha) {
    // get coordinates of our block
    int i_base = 128 * id.get_group(2) + 8 * id.get_local_id(2);
    int j_base = 128 * id.get_group(1) + 8 * id.get_local_id(1);

    // one more check to skip the upper diagonal in blocks that are on the diagonal.
    if(j_base > i_base)
        return;

    // Simple nested loop that calculates 8x8 results in one thread.
    for(int io = 0; io < 8; ++io) {
        int i = i_base + io;
        for(int jo = 0; jo < 8; ++jo) {
            int j = j_base + jo;
            float val = 0;
            for (int s = 0; s < hs; ++s) {
                val += k[i * ks + s] * q[j * qs + s];
            }
            p[i * ps + j] = val * alpha;
        }
    }
}

/*                     ** Chapter IV - ... **
 *
 *  Each worker is producing 64 combined cookies from 8 animals and 8 landscapes. They send there runners of 64 times
 *  to fetch the corresponding shapes. This is terribly inefficient; The runners need a minute or so for each trip,
 *  but making a cookie can be done in just a second.
 *
 *  "Let's try something different tomorrow: Just get all 16 cookie cutters that you need, and do all 64 combinations
 *  of them! See all this free space on your workbench (_registers_), you can keep them all there for easy access."
 *
 *  The next morning, you come back to the field for another practice session. Initially, there is bustling activity
 *  with the runners, picking up 16 shapes for each worker. But then, the workers have to put down their newspapers
 *  and start making cookies. Now there are 64 combinations, so it takes them a full minute.
 *
 *  Not all groups of workers are equally fast. When the first group finishes with all animal-landscape combinations,
 *  they already start asking the runners for the next set of cookie cutters, combining plants and houses. Even though
 *  the workers are much busier than before, they are still spending most of their time just waiting.
 *
 *  Still, instead of being busy for 20 hours, your team is now done with the task in just 3h 30 minutes; already, this
 *  is five times faster.
 *
 *  You think to yourself: "Why should we stop at 8 x 8 combinations? Lets to 16 x 16, that's only twice the work for
 *  the runners, but four times as much for the actual workers."
 *  You head over to the baking area, and make that suggestion to one of your team leaders.
 *  "In theory, that sounds great", she agrees, "but see, we only have limited space on our workbenches (_registers_).
 *  There is still some room left, but we simply cannot bake 256 cookies at the same time, sorry."
 *
 *  A different strategy is needed, then.
 */

// reorganize loops to enable data reuse: 3.5 ms
void matmul_tri_registers(sycl::nd_item<3> id, float* p, int ps, const float* k, int ks, const float* q, int qs, int T, int hs, float alpha) {
    int i_base = 128 * id.get_group(2) + 8 * id.get_local_id(2);
    int j_base = 128 * id.get_group(1) + 8 * id.get_local_id(1);

    if (j_base > i_base)
        return;

    // shift our pointers to the sub-block this thread is responsible for
    k += i_base * ks;
    q += j_base * qs;
    p += i_base * ps + j_base;

    float vals[8][8] = {};
    for (int s = 0; s < hs; ++s) {
        float lhs[8];
        float rhs[8];
        for (int u = 0; u < 8; ++u) {
            lhs[u] = k[u * ks + s];
            rhs[u] = q[u * qs + s];
        }

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                vals[i][j] += lhs[i] * rhs[j];
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            p[i * ps + j] = vals[i][j] * alpha;
        }
    }
}

/*                     ** Chapter IV - By the Bucketload **
 *
 *  Despite the hectic activity, you pick out one of the runners. "Why are you always brining just one shape? Wouldn't
 *  it be much more efficient if you took more than one?"
 *  "Of course", the runner answers, "but they've asked me for an elephant, a lion, a zebra, and a goldfish. These
 *  are all over the place, I can't just pick them up at one spot (_strided acccess_).
 *  "But the lion is right next to the palm tree. You could bring those two together?", you confirm.
 *  "Yes", he says, "if the just asked for the different categories at the same time, that would make things
 *  so much easier. See, I have this bucket, I could carry lots of things in one go if I could just scoop them up
 *  from the same place (_coalesced access_).
 *
 *  OK, then lets fetch the first animal, first plant, first vehicle, and first landmark shape in one go (_vectorized load_).
 *  [Here, the metaphor breaks down a bit: Since we're accumulating all the results, getting more data at the same time
 *  depth-wise doesn't require more space on the workbench. We're stacking the cookies!]
 *
 *  You also streamline the shape combination further. Instead of picking up all animals and landscapes at one, it is
 *  more efficient, using less workbench space, to just pick up all animals. Then, you get one landscape, combine it
 *  will all animals, get the next landscape, combine, and so on.
 *
 *  In this way, instead of 2 x 8 x 4 cookie cutters that take up space, you only need (8+1) x 4 at the same time.
 *
 *  With these optimizations, you are down to 100 minutes for this task. Still slower than Cublas, but not by much.
 *
 *  In the arena, each team also has access to a small storage hut, much closer to their workbenches than the piles of
 *  cookie cutters on the other side. Cublas is using them heavily, so maybe you should, too.
 */

// convenient helper functions to make the code below more readable
sycl::float4 ld_vec(const float* address) {
    return *reinterpret_cast<const sycl::float4*>(address);
}

void st_vec(float* address, sycl::float4 val) {
    *reinterpret_cast<sycl::float4*>(address) = val;
}

// vector instructions for coalesced memory access: 1.7 ms
void matmul_tri3(sycl::nd_item<3> id, float* p, int ps, const float* k, int ks, const float* q, int qs, int T, int hs, float alpha) {
    int i_base = 128 * id.get_group(2) + 8 * id.get_local_id(2);
    int j_base = 128 * id.get_group(1) + 8 * id.get_local_id(1);

    if (j_base > i_base)
        return;

    // shift our pointers to the sub-block this thread is responsible for
    k += i_base * ks;
    q += j_base * qs;
    p += i_base * ps + j_base;

    float vals[8][8] = {};
    for (int s = 0; s < hs; s += 4) {
        // load in float4 to improve coalescing
        sycl::float4 rhs[8];
        for (int u = 0; u < 8; ++u) {
            rhs[u] = ld_vec(q + u * qs + s);
        }

        for (int i = 0; i < 8; ++i) {
            // no need to keep lhs around for the i loop, its only reused in the j loop anyway.
            sycl::float4 lhs = ld_vec(k + i * ks + s);
            for (int j = 0; j < 8; ++j) {
                vals[i][j] += lhs.x() * rhs[j].x();
                vals[i][j] += lhs.y() * rhs[j].y();
                vals[i][j] += lhs.z() * rhs[j].z();
                vals[i][j] += lhs.w() * rhs[j].w();
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            sycl::float4 result;
            result.x() = vals[i][j + 0] * alpha;
            result.y() = vals[i][j + 1] * alpha;
            result.z() = vals[i][j + 2] * alpha;
            result.w() = vals[i][j + 3] * alpha;
            st_vec(p + i * ps + j, result);
        }
    }
}

/*                     ** Chapter V - Sharing is Caring **
 *
 *  You take a look around the shed, and see that there are 32 shelves there. They are much larger than the workbenches,
 *  giving you enough space for all the cookie cutters needed by the entire team.
 *
 *  Within the team, workers have banded together in groups of 32. They are always doing the same thing, reducing the
 *  amount of effort required for coordination. However, that also means that if you send them all to pick up different
 *  cookie cutters from the same shelf, they will have to wait and queue up (_shared memory bank conflict_).
 *
 *  In order to achieve maximum efficiency, we send the runners fetching cutters with the maximum bucket size: 32 different
 *  categories at the same time.
 *
 *  [I'm having trouble getting the specifics into the story in a sensible way. For now, please read the code for more
 *  details.]
 *
 */
void matmul_tri4(sycl::nd_item<3> id, float* p, int ps, const float* k, int ks, const float* q, int qs, int T, int hs, float alpha,
                 sycl::multi_ptr<float[128][32][2], sycl::access::address_space::local_space> lmem) {
    int i_base = 128 * id.get_group(2) + 8 * id.get_local_id(2);
    int j_base = 128 * id.get_group(1) + 8 * id.get_local_id(1);

    // we need all threads for loading data, so none of them can chicken out early, even
    // if they are not responsible for any useful result.
    if (id.get_group(1) > id.get_group(2))
        return;

    k += 128 * id.get_group(2) * ks;
    q += 128 * id.get_group(1) * qs;

    float *shared = (float*) lmem.get_raw();
    float (*lhs_s)[32] = (float (*)[32]) shared;
    float (*rhs_s)[32] = (float (*)[32]) (shared + 128 * 32);

    float vals[8][8] = {};
    for (int so = 0; so < hs; so += 32) {
        // Read a large slice of the input, worked on together by all threads.
        // They are organized differently for this part. We want to ensure
        // fully coalesced loads, so we let a single warp handle consecutive
        // addresses, which means we need to combine two threadIdx.y values
        // in one read operation.
        // note: threads may read data here that they don't need themselves.
        //       this really is a block-level operation.
        sycl::group_barrier(id.get_group());
        for(int y = id.get_local_id(1) / 2; y < 128; y += 8) {
            int xo = (id.get_local_id(1) % 2) * 16;
            lhs_s[y][id.get_local_id(2) + xo] = k[y * ks + so + id.get_local_id(2) + xo];
            rhs_s[y][id.get_local_id(2) + xo] = q[y * qs + so + id.get_local_id(2) + xo];
        }
        sycl::group_barrier(id.get_group());

        for (int si = 0; si < 32; ++si) {
            float rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = rhs_s[u + 8 * id.get_local_id(1)][(si + id.get_local_id(2)) % 32];
            }

            for (int ii = 0; ii < 8; ++ii) {
                float lhs = lhs_s[ii + 8 * id.get_local_id(2)][(si + id.get_local_id(2)) % 32];
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs * rhs[ji];
                }
            }
        }
    }

    // don't write above the diagonal
    if (j_base > i_base)
        return;

    for (int ii = 0; ii < 8; ++ii) {
        for (int ji = 0; ji < 8; ji += 4) {
            int i = i_base + ii;
            int j = j_base + ji;
            sycl::float4 result;
            result.x() = vals[ii][ji + 0] * alpha;
            result.y() = vals[ii][ji + 1] * alpha;
            result.z() = vals[ii][ji + 2] * alpha;
            result.w() = vals[ii][ji + 3] * alpha;
            st_vec(p + i * ps + j, result);
        }
    }
}

void trimul_launcher_naive(float* out, const float* inp, int B, int T, int C, int NH) {
    // we assume nice shapes here. Let's not make the code a mess by supporting weird shapes that you
    // wouldn't want to use anyway.
    assert(T % 128 == 0);
    // No need to ceil_div, if it's not a multiple of 128, we would get wrong results anyway.
    sycl::range<3> grid_dim(NH * B, T / 128, T / 128);
    sycl::range<3> block_dim(1, 16, 16);
    DefaultQueue->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                               [=](sycl::nd_item<3> id) {
        // skip above the diagonal
        if(id.get_group(1) > id.get_group(2))
            return;

        // set up indices
        int C3 = C*3;
        int hs = C / NH; // head size
        float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));

        // we put the "batch x head" dimension into the z block index.
        int h = id.get_group(0) % NH;
        int b = id.get_group(0) / NH;

        // Get the base address for the current batch and head
        const float* q = inp + b * T * C3 + h * hs;
        const float* k = inp + b * T * C3 + h * hs + C;
        float* r = out + (b*NH + h)*T*T;

        matmul_tri_naive(id, r, T, q, C3, k, C3, T, hs, scale);
    });
}

void trimul_launcher_registers(float* out, const float* inp, int B, int T, int C, int NH) {
    // we assume nice shapes here. Let's not make the code a mess by supporting weird shapes that you
    // wouldn't want to use anyway.
    assert(T % 128 == 0);
    // No need to ceil_div, if it's not a multiple of 128, we would get wrong results anyway.
    sycl::range<3> grid_dim(NH * B, T / 128, T / 128);
    sycl::range<3> block_dim(1, 16, 16);
    DefaultQueue->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                               [=](sycl::nd_item<3> id) {
        // skip above the diagonal
        if(id.get_group(1) > id.get_group(2))
            return;

        // set up indices
        int C3 = C*3;
        int hs = C / NH; // head size
        float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));

        // we put the "batch x head" dimension into the z block index.
        int h = id.get_group(0) % NH;
        int b = id.get_group(0) / NH;

        // Get the base address for the current batch and head
        const float* q = inp + b * T * C3 + h * hs;
        const float* k = inp + b * T * C3 + h * hs + C;
        float* r = out + (b*NH + h)*T*T;

        matmul_tri_registers(id, r, T, q, C3, k, C3, T, hs, scale);
    });
}

void trimul_launcher_matmul_tri3(float* out, const float* inp, int B, int T, int C, int NH) {
    // we assume nice shapes here. Let's not make the code a mess by supporting weird shapes that you
    // wouldn't want to use anyway.
    assert(T % 128 == 0);
    // No need to ceil_div, if it's not a multiple of 128, we would get wrong results anyway.
    sycl::range<3> grid_dim(NH * B, T / 128, T / 128);
    sycl::range<3> block_dim(1, 16, 16);
    DefaultQueue->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                               [=](sycl::nd_item<3> id) {
        // skip above the diagonal
        if(id.get_group(1) > id.get_group(2))
            return;

        // set up indices
        int C3 = C*3;
        int hs = C / NH; // head size
        float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));

        // we put the "batch x head" dimension into the z block index.
        int h = id.get_group(0) % NH;
        int b = id.get_group(0) / NH;

        // Get the base address for the current batch and head
        const float* q = inp + b * T * C3 + h * hs;
        const float* k = inp + b * T * C3 + h * hs + C;
        float* r = out + (b*NH + h)*T*T;

        matmul_tri3(id, r, T, q, C3, k, C3, T, hs, scale);
    });
}

void trimul_launcher_matmul_tri4(float* out, const float* inp, int B, int T, int C, int NH) {
    // we assume nice shapes here. Let's not make the code a mess by supporting weird shapes that you
    // wouldn't want to use anyway.
    assert(T % 128 == 0);
    // No need to ceil_div, if it's not a multiple of 128, we would get wrong results anyway.
    sycl::range<3> grid_dim(NH * B, T / 128, T / 128);
    sycl::range<3> block_dim(1, 16, 16);
    DefaultQueue->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                               [=](sycl::nd_item<3> id) {
        // skip above the diagonal
        if(id.get_group(1) > id.get_group(2))
            return;

        // set up indices
        int C3 = C*3;
        int hs = C / NH; // head size
        float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));

        // we put the "batch x head" dimension into the z block index.
        int h = id.get_group(0) % NH;
        int b = id.get_group(0) / NH;

        // Get the base address for the current batch and head
        const float* q = inp + b * T * C3 + h * hs;
        const float* k = inp + b * T * C3 + h * hs + C;
        float* r = out + (b*NH + h)*T*T;

        sycl::multi_ptr<float[128][32][2], sycl::access::address_space::local_space> lmem = 
            syclx::group_local_memory_for_overwrite<float[128][32][2]>(id.get_group());
        matmul_tri4(id, r, T, q, C3, k, C3, T, hs, scale, lmem);
    });
}

/*                     ** Chapter VI - Competition Day **
 *
 * Finally, you feel ready to take on Cublas. You hand out tickets to the event for you friends to see.
 *
 *    ---------------------------------------------------------------------------------
 *    |           CuBLAS vs TriMul - Fight of the Century                             |
 *    |                                                                               |
 *    |   Ticket code:                                                                |
 *    |   > nvcc -O3 --use_fast_math trimat_forward.cu -o trimat_forward -lcublas     |
 *    |   > ./trimat 4                                                                |
 *    |                                                                               |
 *    ---------------------------------------------------------------------------------
 */

void trimul_gpu(int kernel_num,
                float* out,  const float* inp,
                int B, int T, int C, int NH) {
    switch (kernel_num) {
        case 0:
            trimul_cublas(out, inp, B, T, C, NH);
            break;
        case 1:
            trimul_launcher_naive(out, inp, B, T, C, NH);
            break;
        case 2:
            trimul_launcher_registers(out, inp, B, T, C, NH);
            break;
        case 3:
            trimul_launcher_matmul_tri3(out, inp, B, T, C, NH);
            break;
        case 4:
            trimul_launcher_matmul_tri4(out, inp, B, T, C, NH);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}



int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    // set up the device
    sycl::queue defaultQueue(sycl::gpu_selector_v, 
                            {sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});
    printf("Using device: %s\n", defaultQueue.get_device().get_info<sycl::info::device::name>().c_str());
    if (!defaultQueue.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cerr << "GPU does not support USM device allocations\n";
        return 1;
    }
    DefaultQueue = &defaultQueue;

    // create host memory of random numbers
    float* out = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_inp;
    syclMallocCheck(d_out = sycl::malloc_device<float>(B * NH * T * T, defaultQueue));
    syclMallocCheck(d_inp = sycl::malloc_device<float>(B * T * 3 * C, defaultQueue));
    // buffer for onednn
    syclMallocCheck(d_qkvr = sycl::malloc_device<float>(B * T * 3 * C, defaultQueue));
    defaultQueue.memcpy(d_inp, inp, B * T * 3 * C * sizeof(float));
    defaultQueue.wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    trimul_cpu(out, inp, B, T, C, NH);
    trimul_gpu(kernel_num, d_out, d_inp, B, T, C, NH);
    validate_result(d_out, out, "out", B * NH * T * T, 1e-4f);

    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    int repeat_times = 100;

    float elapsed_time = benchmark_kernel(repeat_times, trimul_gpu,
                                          kernel_num, d_out, d_inp,
                                          B, T, C, NH);


    float cublas_time = benchmark_kernel(repeat_times, trimul_gpu,
                                         0, d_out, d_inp,
                                         B, T, C, NH);

    printf("time %.2f ms vs %.2f ms for CuBLAS\n", elapsed_time, cublas_time);

    // free memory
    free(out);
    free(inp);
    sycl::free(d_qkvr, defaultQueue);
    sycl::free(d_out, defaultQueue);
    sycl::free(d_inp, defaultQueue);

    return 0;
}
