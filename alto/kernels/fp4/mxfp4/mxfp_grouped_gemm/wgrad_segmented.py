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
import math

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


# ---- load-balanced variant: long expert segments split into chunks ------------------------------------------
# With skewed routing a few experts hold many more rows than the mean; the programs of those experts finish
# last while the other compute units idle. Here an expert with more than CHUNK rows is split into
# ceil(rows / CHUNK) chunks, each its own program writing an fp32 partial tile; a second kernel adds the
# partial tiles of each split expert in chunk order (deterministic). Experts with at most CHUNK rows are
# computed exactly as in _wgrad_seg_kernel (same rows, same order: bit-identical). The work list is derived
# inside the kernel from the segment offsets (no host synchronization, no extra launches); the grid is a
# host-side upper bound and surplus programs exit.

@triton.jit
def _work_split(seg_ptr, CHUNK, NUM_EXPERTS, E_POW2: tl.constexpr):
    ei = tl.arange(0, E_POW2)
    valid = ei < NUM_EXPERTS
    st = tl.load(seg_ptr + ei, mask=valid, other=0)
    en = tl.load(seg_ptr + ei + 1, mask=valid, other=0)
    ln = en - st
    nch = tl.where(valid, tl.where(ln > CHUNK, (ln + CHUNK - 1) // CHUNK, 1), 0)
    spl = tl.where(nch > 1, nch, 0)
    return ei, st, en, nch, spl


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _wgrad_bal_kernel(
    go_ptr, x_ptr, dw_ptr, part_ptr, seg_ptr, gos_ptr, xs_ptr,
    stride_gom, stride_gon, stride_xm, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_gosm, stride_gosn, stride_xsm, stride_xsk,
    N, K, CHUNK, NUM_EXPERTS,
    E_POW2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    item = tl.program_id(2)
    ei, st, en, nch, spl = _work_split(seg_ptr, CHUNK, NUM_EXPERTS, E_POW2)
    cum = tl.cumsum(nch, axis=0)
    total = tl.sum(nch, axis=0)
    if item < total:
        e = tl.sum((cum <= item).to(tl.int32), axis=0)
        sel = ei == e
        n_e = tl.sum(tl.where(sel, nch, 0), axis=0)
        c = item - tl.sum(tl.where(sel, cum - nch, 0), axis=0)
        start = tl.sum(tl.where(sel, st, 0), axis=0)
        end = tl.sum(tl.where(sel, en, 0), axis=0)
        lo = start + c * CHUNK
        hi = tl.minimum(end, lo + CHUNK)      # unsplit experts have at most CHUNK rows: hi = end

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_n = offs_n < N
        mask_k = offs_k < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m0 in range(lo, hi, BLOCK_M):
            offs_mp = m0 // 2 + tl.arange(0, BLOCK_M // 2)
            offs_ms = m0 // 32 + tl.arange(0, BLOCK_M // 32)
            mmp = offs_mp < hi // 2
            mms = offs_ms < hi // 32
            go = tl.load(go_ptr + offs_mp[None, :] * stride_gom + offs_n[:, None] * stride_gon,
                         mask=mmp[None, :] & mask_n[:, None], other=0)
            gs = tl.load(gos_ptr + offs_ms[None, :] * stride_gosm + offs_n[:, None] * stride_gosn,
                         mask=mms[None, :] & mask_n[:, None], other=127)
            xx = tl.load(x_ptr + offs_mp[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                         mask=mmp[:, None] & mask_k[None, :], other=0)
            xs = tl.load(xs_ptr + offs_ms[None, :] * stride_xsm + offs_k[:, None] * stride_xsk,
                         mask=mms[None, :] & mask_k[:, None], other=127)
            acc += tl.dot_scaled(go, gs, "e2m1", xx, xs, "e2m1")
        mask_nk = mask_n[:, None] & mask_k[None, :]
        if n_e == 1:
            w_ptrs = dw_ptr + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            tl.store(w_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=mask_nk)
        else:
            cs = tl.cumsum(spl, axis=0)
            slot = tl.sum(tl.where(sel, cs - spl, 0), axis=0) + c
            p_ptrs = part_ptr + slot.to(tl.int64) * N * K + offs_n[:, None] * K + offs_k[None, :]
            tl.store(p_ptrs, acc, mask=mask_nk)


@triton.jit
def _wgrad_reduce_kernel(
    dw_ptr, part_ptr, seg_ptr, stride_we, stride_wn, stride_wk, N, K, CHUNK, NUM_EXPERTS,
    E_POW2: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    e = tl.program_id(2)
    ei, st, en, nch, spl = _work_split(seg_ptr, CHUNK, NUM_EXPERTS, E_POW2)
    sel = ei == e
    n_e = tl.sum(tl.where(sel, nch, 0), axis=0)
    if n_e > 1:
        cs = tl.cumsum(spl, axis=0)
        slot0 = tl.sum(tl.where(sel, cs - spl, 0), axis=0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_nk = (offs_n < N)[:, None] & (offs_k < K)[None, :]
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c in range(0, n_e):                 # chunk order: deterministic
            p_ptrs = part_ptr + (slot0 + c).to(tl.int64) * N * K + offs_n[:, None] * K + offs_k[None, :]
            acc += tl.load(p_ptrs, mask=mask_nk, other=0.0)
        w_ptrs = dw_ptr + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        tl.store(w_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=mask_nk)


def balanced_chunk(total_rows: int, num_experts: int) -> int:
    """Rows per chunk: twice the mean segment length, rounded up to 256 (a multiple of every BLOCK_M), so that
    with balanced routing no expert is split."""
    mean = total_rows / max(num_experts, 1)
    return max(256, int(math.ceil(2 * mean / 256)) * 256)
