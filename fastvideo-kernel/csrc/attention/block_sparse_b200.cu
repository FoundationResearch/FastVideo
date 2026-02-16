// SM100/B200 block-sparse attention forward implementation (ThunderKittens v2).
//
// - Builds only when FASTVIDEO_KERNEL_BUILD_TK_SM100 is enabled (CMake).
// - Uses include/tk_b200 (must be isolated from include/tk).
// - Forward only is required for now; backward remains a stub and will throw
//   if called.
//
// Interface matches the existing SM90/H100 op to keep Python wrapper unchanged.

#include "kittens.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

using namespace kittens;

namespace {
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

template <int D>
struct fwd_attend_ker_tile_dims {};
template <>
struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int qo_height = 4 * 16;
    constexpr static int kv_height = 4 * 16;
};
template <>
struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int qo_height = 4 * 16;
    constexpr static int kv_height = 4 * 16;
};

template <int D>
struct fwd_globals {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height,
                         fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height,
                         fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height,
                         fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<
        st_fl<fwd_attend_ker_tile_dims<D>::qo_height,
              fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height,
                         fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N;
    const int hr;
    const int max_kv_blocks_per_q;

    int32_t* __restrict__ q2k_block_sparse_index;
    int32_t* __restrict__ q2k_block_sparse_num;
    int32_t* __restrict__ block_size;
};

template <int D>
__global__ __launch_bounds__(128, 4) void
fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile = st_bf<64, K::tile_width>;
    using k_tile = st_bf<64, K::tile_width>;
    using v_tile = st_bf<64, K::tile_width>;
    using o_tile = st_bf<64, K::tile_width>;
    using p_tile = st_bf<64, 64>;

    q_tile (&q_smem)[1] = al.allocate<q_tile, 1>();
    k_tile (&k_smem)[1] = al.allocate<k_tile, 1>();
    v_tile (&v_smem)[1] = al.allocate<v_tile, 1>();
    p_tile (&p_smem)[1] = al.allocate<p_tile, 1>();
    // Reuse Q shared memory region for O to reduce shared memory footprint.
    // This is safe because we fully materialize Q into registers before
    // writing O back to shared.
    auto (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

    const int kv_head_idx = blockIdx.y / g.hr;
    const int seq_idx = blockIdx.x;

    int32_t* q2k_block_sparse_index_ptr =
        g.q2k_block_sparse_index +
        blockIdx.z * gridDim.y * gridDim.x * g.max_kv_blocks_per_q +
        blockIdx.y * gridDim.x * g.max_kv_blocks_per_q +
        blockIdx.x * g.max_kv_blocks_per_q;
    int32_t* q2k_block_sparse_num_ptr =
        g.q2k_block_sparse_num +
        blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    const int32_t kv_blocks = q2k_block_sparse_num_ptr[0];

    __shared__ kittens::semaphore qsmem_semaphore;
    __shared__ kittens::semaphore k_smem_arrived;
    __shared__ kittens::semaphore v_smem_arrived;
    __shared__ kittens::semaphore score_done;
    __shared__ kittens::semaphore pv_done;
    __shared__ uint32_t tmem_addr;
    __shared__ kittens::semaphore tmem_provisioned;

    if (threadIdx.x == 0) {
        const int32_t kv_block_index = q2k_block_sparse_index_ptr[0];

        init_semaphore(qsmem_semaphore, 0, 1);
        init_semaphore(k_smem_arrived, 0, 1);
        init_semaphore(v_smem_arrived, 0, 1);

        // preload q block
        coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
        tma::load_async(q_smem[0], g.q, q_tile_idx, qsmem_semaphore);

        // preload the zeroth block of kv
        tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
        coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
        tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);

        tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
        coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
        tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);

        // TCGEN05 barriers (phase toggles each iteration)
        init_semaphore(score_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
        init_semaphore(tmem_provisioned, 0, 1);
    }
    __syncthreads();

    // wait for q block
    wait(qsmem_semaphore, 0);

    const int warp_id = kittens::warpid();
    const bool active_warp = (warp_id < 4);

    // Tensor memory (tmem): unmanaged explicit provision / set_addr / deprovision
    // NOTE: tensor_allocator::provision/deprovision must be called by one full warp.
    tensor_allocator<1, 1, false> tm_alloc{};
    if (warp_id == 0) {
        tm_alloc.provision(tmem_addr);
        if (laneid() == 0) warp::arrive(tmem_provisioned);
    }
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    // Allocate tensor memory tiles for tcgen05 matmul outputs.
    // - score_tm: 64x64 fp32 (QK^T) per kv block (overwritten)
    // - pv_tm:    64xD  fp32 (P*V) per kv block (overwritten)
    using score_tt = half_tt_fl<64>;
    using pv_tt = half_tt_fl<K::tile_width>;
    score_tt score_tm = tm_alloc.template allocate<score_tt>(0, 0);
    pv_tt pv_tm = tm_alloc.template allocate<pv_tt>(0, 128);
    int score_phase = 0;
    int pv_phase = 0;

    // Per-warp Q tile: (16, D)
    rt_bf<16, K::tile_width> q_reg;
    if (active_warp) {
        auto q_sub = q_smem[0].template subtile<16, K::tile_width>(
            int2{warp_id, 0});
        warp::load(q_reg, q_sub);
    }

    rt_fl<16, K::tile_width> o_reg;
    if (active_warp) {
        warp::zero(o_reg);
    }

    col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
    if (active_warp) {
        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
    }

    // main loop: pipeline loads next KV block while computing current.
    for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        // Load K/V tile for this kv_idx
        wait(k_smem_arrived, kv_idx % 2);
        wait(v_smem_arrived, kv_idx % 2);

        // ---- tcgen05: score_tm = Q * K^T (64x64 fp32) ----
        __syncthreads();
        if (warp_id == 0 && warp::elect_leader()) {
            // overwrite score_tm each iteration
            mm_ABt(score_tm, q_smem[0], k_smem[0], score_done);
        }
        wait(score_done, score_phase);
        score_phase ^= 1;
        // Load score_tm into per-warp registers (16x64 each warp)
        rt_fl<16, 64> att_block;
        warpgroup::load_async(att_block, score_tm);
        tensor_load_wait();

        if (active_warp) {
            warp::copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) {
                warp::mul(max_vec_last_scaled, max_vec_last_scaled,
                          1.44269504089f * 0.125f);
            } else {
                warp::mul(max_vec_last_scaled, max_vec_last_scaled,
                          1.44269504089f * 0.08838834764f);
            }

            warp::right_fill(att_block, att_block,
                             g.block_size[q2k_block_sparse_index_ptr[kv_idx]],
                             base_types::constants<float>::neg_infty());
            warp::row_max(max_vec, att_block, max_vec);

            if constexpr (D == 64) {
                warp::mul(att_block, att_block, 1.44269504089f * 0.125f);
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
            } else {
                warp::mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
            }

            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
            warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
            warp::row_sum(norm_vec, att_block, norm_vec);
            warp::add(att_block, att_block, 0.f);

            warp::mul_row(o_reg, o_reg, max_vec_last_scaled);

            // Write P (bf16) to shared so tcgen05 can consume it.
            rt_bf<16, 64> p_reg_bf;
            warp::copy(p_reg_bf, att_block);
            auto p_sub = p_smem[0].template subtile<16, 64>(int2{warp_id, 0});
            warp::store(p_sub, p_reg_bf);
        }

        // ---- tcgen05: pv_tm = P * V (64xD fp32) ----
        __syncthreads();
        if (warp_id == 0 && warp::elect_leader()) {
            mm_AB(pv_tm, p_smem[0], v_smem[0], pv_done);
        }
        wait(pv_done, pv_phase);
        pv_phase ^= 1;

        rt_fl<16, K::tile_width> pv_reg;
        warpgroup::load_async(pv_reg, pv_tm);
        tensor_load_wait();

        if (active_warp) {
            warp::add(o_reg, o_reg, pv_reg);
        }

        // Kick next K/V loads (single buffer) after all warps finished consuming.
        __syncthreads();
        if (threadIdx.x == 0 && (kv_idx + 1) < kv_blocks) {
            const int32_t kv_block_index =
                q2k_block_sparse_index_ptr[kv_idx + 1];
            tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
            coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index,
                                        0};
            tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);

            tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
            coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index,
                                        0};
            tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
        }
        __syncthreads();
    }

    if (active_warp) {
        warp::div_row(o_reg, o_reg, norm_vec);

        // Store (16,D) into shared output tile, then a single warp TMA-stores to GMEM.
        auto o_sub = o_smem[0].template subtile<16, K::tile_width>(
            int2{warp_id, 0});
        warp::store(o_sub, o_reg);
    }
    __syncthreads();

    if (threadIdx.x / 32 == 0) {
        coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
        tma::store_async(g.o, o_smem[0], o_tile_idx);
        tma::store_async_wait();
    }

    // Release tensor memory allocation.
    __syncthreads();
    if (warp_id == 0) {
        tm_alloc.deprovision();
    }
}

}  // namespace

#include "pyutils/torchutils.cuh"

std::vector<torch::Tensor> block_sparse_attention_forward_sm100(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q2k_block_sparse_index,
    torch::Tensor q2k_block_sparse_num,
    torch::Tensor kv_block_size) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    // q shape: (batch, qo_heads, q_seq_len, head_dim)
    // k shape: (batch, kv_heads, kv_seq_len, head_dim)
    // v shape: (batch, kv_heads, kv_seq_len, head_dim)
    // q2k_block_sparse_index shape: (batch, qo_heads, num_q_blocks, max_kv_blocks_per_q)
    // q2k_block_sparse_num shape: (batch, qo_heads, num_q_blocks)
    // kv_block_size shape: (num_kv_blocks)
    const auto batch = q.size(0);
    const auto q_seq_len = q.size(2);
    const auto kv_seq_len = k.size(2);
    const auto head_dim = q.size(3);
    const auto qo_heads = q.size(1);
    const auto kv_heads = k.size(1);

    const auto max_kv_blocks_per_q = q2k_block_sparse_index.size(3);
    const auto num_q_blocks = q2k_block_sparse_index.size(2);
    const auto num_kv_blocks = kv_block_size.size(0);

    TORCH_CHECK(batch == 1,
                "Batch size dim will be removed in the future, please set "
                "batch to 1");
    TORCH_CHECK(num_q_blocks * BLOCK_M == q_seq_len,
                "Expected Q length padded to 64-token blocks");
    TORCH_CHECK(num_kv_blocks * BLOCK_M == kv_seq_len,
                "Expected KV length padded to 64-token blocks");

    TORCH_CHECK(qo_heads >= kv_heads,
                "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0,
                "QO heads must be divisible by KV heads");
    const auto hr = qo_heads / kv_heads;

    auto* q_ptr = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    auto* k_ptr = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    auto* v_ptr = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());

    torch::Tensor o = torch::empty(
        {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
         static_cast<const uint>(q_seq_len), static_cast<const uint>(head_dim)},
        v.options());
    // Forward-only on SM100 for now: we return a correctly-shaped LSE buffer,
    // but do not populate it (backward is not supported yet).
    torch::Tensor l_vec = torch::zeros(
        {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
         static_cast<const uint>(q_seq_len), static_cast<const uint>(1)},
        torch::TensorOptions()
            .dtype(torch::kFloat)
            .device(q.device())
            .memory_format(at::MemoryFormat::Contiguous));

    auto* o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    auto* l_ptr = l_vec.data_ptr<float>();

    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int threads = 128;
    dim3 grid(q_seq_len / BLOCK_M, qo_heads, batch);

    if (head_dim == 64) {
        using globals = fwd_globals<64>;
        using q_tile = typename globals::q_tile;
        using k_tile = typename globals::k_tile;
        using v_tile = typename globals::v_tile;
        using l_col_vec = typename globals::l_col_vec;
        using o_tile = typename globals::o_tile;

        using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

        q_global qg_arg{q_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads),
                        static_cast<unsigned int>(q_seq_len), 64U};
        k_global kg_arg{k_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(kv_heads),
                        static_cast<unsigned int>(kv_seq_len), 64U};
        v_global vg_arg{v_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(kv_heads),
                        static_cast<unsigned int>(kv_seq_len), 64U};
        l_global lg_arg{l_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads), 1U,
                        static_cast<unsigned int>(q_seq_len)};
        o_global og_arg{o_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads),
                        static_cast<unsigned int>(q_seq_len), 64U};

        globals g{qg_arg,
                  kg_arg,
                  vg_arg,
                  lg_arg,
                  og_arg,
                  static_cast<int>(q_seq_len),
                  static_cast<int>(hr),
                  static_cast<int>(max_kv_blocks_per_q),
                  reinterpret_cast<int32_t*>(q2k_block_sparse_index.data_ptr()),
                  reinterpret_cast<int32_t*>(q2k_block_sparse_num.data_ptr()),
                  reinterpret_cast<int32_t*>(kv_block_size.data_ptr())};

        // Shared memory: Q + K + V (O aliases Q) + allocator overhead.
        constexpr int mem_size =
            int(sizeof(typename globals::q_tile) + sizeof(typename globals::k_tile) +
                sizeof(typename globals::v_tile) + sizeof(st_bf<64, 64>) + 4096);
        cudaFuncSetAttribute(fwd_attend_ker<64>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             mem_size);
        fwd_attend_ker<64><<<grid, threads, mem_size, stream>>>(g);
        CHECK_CUDA_ERROR(cudaGetLastError());
        return {o, l_vec};
    }

    if (head_dim == 128) {
        using globals = fwd_globals<128>;
        using q_tile = typename globals::q_tile;
        using k_tile = typename globals::k_tile;
        using v_tile = typename globals::v_tile;
        using l_col_vec = typename globals::l_col_vec;
        using o_tile = typename globals::o_tile;

        using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

        q_global qg_arg{q_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads),
                        static_cast<unsigned int>(q_seq_len), 128U};
        k_global kg_arg{k_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(kv_heads),
                        static_cast<unsigned int>(kv_seq_len), 128U};
        v_global vg_arg{v_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(kv_heads),
                        static_cast<unsigned int>(kv_seq_len), 128U};
        l_global lg_arg{l_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads), 1U,
                        static_cast<unsigned int>(q_seq_len)};
        o_global og_arg{o_ptr, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads),
                        static_cast<unsigned int>(q_seq_len), 128U};

        globals g{qg_arg,
                  kg_arg,
                  vg_arg,
                  lg_arg,
                  og_arg,
                  static_cast<int>(q_seq_len),
                  static_cast<int>(hr),
                  static_cast<int>(max_kv_blocks_per_q),
                  reinterpret_cast<int32_t*>(q2k_block_sparse_index.data_ptr()),
                  reinterpret_cast<int32_t*>(q2k_block_sparse_num.data_ptr()),
                  reinterpret_cast<int32_t*>(kv_block_size.data_ptr())};

        // Shared memory: Q + K + V (O aliases Q) + allocator overhead.
        constexpr int mem_size =
            int(sizeof(typename globals::q_tile) + sizeof(typename globals::k_tile) +
                sizeof(typename globals::v_tile) + sizeof(st_bf<64, 64>) + 4096);
        cudaFuncSetAttribute(fwd_attend_ker<128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             mem_size);
        fwd_attend_ker<128><<<grid, threads, mem_size, stream>>>(g);
        CHECK_CUDA_ERROR(cudaGetLastError());
        return {o, l_vec};
    }

    TORCH_CHECK(false,
                "Unsupported head_dim for SM100 block sparse attention: ",
                head_dim, " (expected 64 or 128)");
}

std::vector<torch::Tensor> block_sparse_attention_backward_sm100(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor l_vec,
    torch::Tensor og,
    torch::Tensor k2q_block_sparse_index,
    torch::Tensor k2q_block_sparse_num,
    torch::Tensor kv_block_size) {
    (void)q;
    (void)k;
    (void)v;
    (void)o;
    (void)l_vec;
    (void)og;
    (void)k2q_block_sparse_index;
    (void)k2q_block_sparse_num;
    (void)kv_block_size;
    TORCH_CHECK(false,
                "SM100/B200 block_sparse_attention_backward is not implemented "
                "yet. Set FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1 to use Triton "
                "fallback.");
}


