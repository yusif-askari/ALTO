"""Grouped MXFP4 weight-gradient GEMM over contiguous expert segments (gfx950, Triton tl.dot_scaled).

    dW[e] = dy_eᵀ · x_e        for every expert e, contraction over the tokens of e

Operands are MXFP4, quantized along the token axis (blocks of 32 tokens), packed two tokens per byte:
    dy  [T/2, N] uint8, scales [T/32, N] uint8 (e8m0)
    x   [T/2, K] uint8, scales [T/32, K] uint8
Tokens are grouped by expert in ascending expert order, every segment padded to a multiple of 128 rows
(ALTO's layout); `expert_indices` [T] gives the expert of every row.

Differences to ALTO's `_kernel_mxfp4_grouped_gemm_backward_dw`:
  - each program reads its expert's row range [start, end) from a segment-offset array and loops over
    those rows only (ALTO's kernel scans the token groups of all experts and skips the others);
  - the token count is a runtime argument (ALTO declares it constexpr, so every new token count compiles a
    new kernel);
  - tile sizes are autotuned per (N, K) (ALTO's grouped kernels run with one fixed 128x128x128 config).
The accumulation is the same: fp32, over the expert's rows in ascending order of BLOCK_M-row chunks,
each chunk's product added to the accumulator separately.
"""
import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bm in (128, 256):
        for bn, bk in ((128, 128), (128, 256), (256, 128), (256, 256), (64, 128), (128, 64)):
            for warps in (4, 8):
                cfgs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=warps,
                                          num_stages=2))
    return cfgs


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _wgrad_seg_kernel(
    go_ptr, x_ptr, dw_ptr, seg_ptr, gos_ptr, xs_ptr,
    stride_gom, stride_gon, stride_xm, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_gosm, stride_gosn, stride_xsm, stride_xsk,
    N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    e = tl.program_id(2)
    start = tl.load(seg_ptr + e)
    end = tl.load(seg_ptr + e + 1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_k = offs_k < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m0 in range(start, end, BLOCK_M):
        offs_mp = m0 // 2 + tl.arange(0, BLOCK_M // 2)           # packed rows (2 tokens per byte)
        offs_ms = m0 // 32 + tl.arange(0, BLOCK_M // 32)         # scale rows (32 tokens per scale)
        mmp = offs_mp < end // 2
        mms = offs_ms < end // 32
        # dyᵀ tile [BLOCK_N, BLOCK_M/2], packed along the contraction (token) axis
        go = tl.load(go_ptr + offs_mp[None, :] * stride_gom + offs_n[:, None] * stride_gon,
                     mask=mmp[None, :] & mask_n[:, None], other=0)
        gs = tl.load(gos_ptr + offs_ms[None, :] * stride_gosm + offs_n[:, None] * stride_gosn,
                     mask=mms[None, :] & mask_n[:, None], other=127)
        # x tile [BLOCK_M/2, BLOCK_K], packed along the contraction axis
        xx = tl.load(x_ptr + offs_mp[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                     mask=mmp[:, None] & mask_k[None, :], other=0)
        xs = tl.load(xs_ptr + offs_ms[None, :] * stride_xsm + offs_k[:, None] * stride_xsk,
                     mask=mms[None, :] & mask_k[:, None], other=127)
        # separate add (not dot_scaled(..., acc)): same rounding order as ALTO's kernel, which reproduces
        # the emulation bit for bit; accumulating inside the MFMA changed 2 of 5.8e7 outputs by 1 bf16 ulp
        acc += tl.dot_scaled(go, gs, "e2m1", xx, xs, "e2m1")
    w_ptrs = dw_ptr + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(w_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_k[None, :])


def segment_offsets(expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Row offsets [E+1] of the expert segments from the sorted per-row expert index (no host sync)."""
    probe = torch.arange(num_experts + 1, device=expert_indices.device, dtype=expert_indices.dtype)
    return torch.searchsorted(expert_indices, probe).to(torch.int32)
