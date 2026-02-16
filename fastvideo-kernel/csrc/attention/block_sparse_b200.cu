#include <torch/extension.h>
#
#include <vector>
#
// NOTE:
// This file is the SM100/B200 counterpart of csrc/attention/block_sparse_h100.cu.
// Today it's a *stub* to validate the build + binding pipeline with a newer
// ThunderKittens checkout (include/tk_b200) without affecting the existing H100
// kernels (include/tk).
//
// Once you start real B200 bring-up, replace the TORCH_CHECK(false, ...) below
// with the actual implementation.
#
std::vector<torch::Tensor> block_sparse_attention_forward_sm100(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q2k_block_sparse_index,
    torch::Tensor q2k_block_sparse_num,
    torch::Tensor kv_block_size) {
    (void)q;
    (void)k;
    (void)v;
    (void)q2k_block_sparse_index;
    (void)q2k_block_sparse_num;
    (void)kv_block_size;
    TORCH_CHECK(
        false,
        "SM100/B200 block_sparse_attention_forward is not implemented yet. "
        "Set FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1 to use Triton fallback.");
}
#
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
    TORCH_CHECK(
        false,
        "SM100/B200 block_sparse_attention_backward is not implemented yet. "
        "Set FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1 to use Triton fallback.");
}


