# SPDX-License-Identifier: MIT
"""Routed-expert fc1 with SwiGLU and the fc2 input quantization in the GEMM epilogue (gfx950, MXFP4).

One kernel computes, for every 128-aligned expert segment of the padded token layout,

    y  = Q(x) · Q_tile(W1_e)ᵀ                      [Tp, 2F]  bf16, gate half | value half (as the unfused GEMM)
    h  = bf16( silu(g) · v · p )                    [Tp, F]   g, v = the bf16 gate / value of y, p = router prob
    qa, sa = Q(h) along F                           fc2 forward operand  (blocks of 32 features)
    qb, sb = Q(H·h) along tokens                    fc2 wgrad operand    (H: 32x32 rotation of each 32-token block)

with round-to-nearest quantization (the forward operands of the alto recipe are not stochastically rounded).
Each program owns a [BLOCK_M, BLOCK_N] tile of h and accumulates the two matching [BLOCK_M, BLOCK_N] tiles
of y (gate columns n.., value columns F+n..), so the activation and both quantizations of h happen in
registers: h is never written to memory, and y is written once (kept for the SwiGLU backward).

The epilogue reuses the scale rule and FP4 packing of the standalone quantizers (_calculate_scales,
_pack_fp4) and the rotation of the fused dual quantization, so qa/sa/qb/sb are bit-identical to
dual_quantize(h) of the same h.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..mxfp_quantization import _calculate_scales, _pack_fp4
from .autotune import ALIGN_SIZE_M


def _configs():
    shapes = [(64, 32, 128, 4), (64, 64, 128, 4), (128, 32, 128, 4), (128, 64, 128, 8), (64, 64, 64, 4),
              (128, 64, 64, 8), (64, 32, 256, 4), (128, 32, 256, 8)]
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=2)
            for m, n, k, w in shapes]


@triton.autotune(configs=_configs(), key=["F", "K", "TP_BUCKET", "STORE_H"])
@triton.jit
def _fc1_swiglu_quant_kernel(
    xq_ptr, xs_ptr, w_ptr, ws_ptr, idx_ptr, p_ptr, had_ptr,
    y_ptr, qa_ptr, sa_ptr, qb_ptr, sb_ptr, hout_ptr,
    stride_we, stride_wn, stride_wse, stride_wsn,
    Tp, TP_BUCKET,
    F: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SCALE_SELECTION: tl.constexpr, STORE_H: tl.constexpr,
):
    QB: tl.constexpr = 32
    KP: tl.constexpr = K // 2              # packed K
    KS: tl.constexpr = K // QB             # scales along K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask_m = offs_m < Tp
    mask_n = offs_n < F
    # one expert per tile: segments are ALIGN_SIZE_M-aligned and BLOCK_M divides ALIGN_SIZE_M
    expert = tl.load(idx_ptr + m0)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_ng_s = offs_n // QB               # 32x32 weight tiles: one scale per 32 rows and 32 columns
    offs_nv_s = (F + offs_n) // QB
    w_e = w_ptr + expert * stride_we
    ws_e = ws_ptr + expert * stride_wse
    for k0 in range(0, K, BLOCK_K):
        offs_kp = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        offs_ks = k0 // QB + tl.arange(0, BLOCK_K // QB)
        mask_kp = offs_kp < KP
        mask_ks = offs_ks < KS
        a = tl.load(xq_ptr + offs_m[:, None] * KP + offs_kp[None, :],
                    mask=mask_m[:, None] & mask_kp[None, :], other=0)
        a_s = tl.load(xs_ptr + offs_m[:, None] * KS + offs_ks[None, :],
                      mask=mask_m[:, None] & mask_ks[None, :], other=127)
        bg = tl.load(w_e + offs_n[None, :] * stride_wn + offs_kp[:, None],
                     mask=mask_kp[:, None] & mask_n[None, :], other=0)
        bv = tl.load(w_e + (F + offs_n)[None, :] * stride_wn + offs_kp[:, None],
                     mask=mask_kp[:, None] & mask_n[None, :], other=0)
        bg_s = tl.load(ws_e + offs_ng_s[:, None] * stride_wsn + offs_ks[None, :],
                       mask=mask_n[:, None] & mask_ks[None, :], other=127)
        bv_s = tl.load(ws_e + offs_nv_s[:, None] * stride_wsn + offs_ks[None, :],
                       mask=mask_n[:, None] & mask_ks[None, :], other=127)
        acc_g += tl.dot_scaled(a, a_s, "e2m1", bg, bg_s, "e2m1", lhs_k_pack=True, rhs_k_pack=True,
                               out_dtype=tl.float32)
        acc_v += tl.dot_scaled(a, a_s, "e2m1", bv, bv_s, "e2m1", lhs_k_pack=True, rhs_k_pack=True,
                               out_dtype=tl.float32)

    # ---- y (bf16, as the unfused GEMM writes it), kept for the SwiGLU backward
    g = acc_g.to(tl.bfloat16)
    v = acc_v.to(tl.bfloat16)
    mask_mn = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + offs_m[:, None] * (2 * F) + offs_n[None, :], g, mask=mask_mn)
    tl.store(y_ptr + offs_m[:, None] * (2 * F) + (F + offs_n)[None, :], v, mask=mask_mn)

    # ---- h = bf16(silu(g) * v * p), fp32 arithmetic from the bf16 g, v (one rounding at the end).
    # silu(g) = g / (1 + exp(-g)) with libdevice exp, exactly as torch.compile (inductor) generates it for
    # Megatron's weighted SwiGLU, so h matches the unfused path bit for bit (g * tl.sigmoid(g) differs in
    # the last fp32 bit for ~6 in 10^6 elements).
    p = tl.load(p_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    gf = g.to(tl.float32)
    silu = gf / (libdevice.exp(-gf) + 1.0)
    h = ((silu * v.to(tl.float32)) * p[:, None]).to(tl.bfloat16)
    if STORE_H:
        tl.store(hout_ptr + offs_m[:, None] * F + offs_n[None, :], h, mask=mask_mn)

    # ---- fc2 forward operand: Q(h) along F, round to nearest
    sa = _calculate_scales(h, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                           USE_DYNAMIC_CLIP=False, SCALE_SELECTION=SCALE_SELECTION)
    qa = _pack_fp4(h, sa, 0, 0, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                   USE_SR=False, USE_ASM=True)
    offs_n2 = n0 // 2 + tl.arange(0, BLOCK_N // 2)
    offs_ns = n0 // QB + tl.arange(0, BLOCK_N // QB)
    tl.store(qa_ptr + offs_m[:, None] * (F // 2) + offs_n2[None, :], qa,
             mask=mask_m[:, None] & (offs_n2 < F // 2)[None, :])
    tl.store(sa_ptr + offs_m[:, None] * (F // QB) + offs_ns[None, :], sa,
             mask=mask_m[:, None] & (offs_ns < F // QB)[None, :])

    # ---- fc2 wgrad operand: Q(H·h) along tokens (rotation as in the fused dual quantization)
    hr = tl.arange(0, QB)
    ht = tl.load(had_ptr + hr[None, :] * QB + hr[:, None])               # Hᵀ
    hg = tl.reshape(tl.trans(h), (BLOCK_N * BLOCK_M // QB, QB))          # rows (feature, token group)
    r = tl.dot(hg, ht, out_dtype=tl.float32)
    r = tl.reshape(r.to(tl.bfloat16), (BLOCK_N, BLOCK_M))
    sb = _calculate_scales(r, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_M, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                           USE_DYNAMIC_CLIP=False, SCALE_SELECTION=SCALE_SELECTION)
    qb = _pack_fp4(r, sb, 0, 0, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_M, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                   USE_SR=False, USE_ASM=True)
    offs_m2 = m0 // 2 + tl.arange(0, BLOCK_M // 2)
    offs_ms = m0 // QB + tl.arange(0, BLOCK_M // QB)
    tl.store(qb_ptr + offs_n[:, None] * (Tp // 2) + offs_m2[None, :], qb,
             mask=mask_n[:, None] & (offs_m2 < Tp // 2)[None, :])
    tl.store(sb_ptr + offs_n[:, None] * (Tp // QB) + offs_ms[None, :], sb,
             mask=mask_n[:, None] & (offs_ms < Tp // QB)[None, :])


def fc1_swiglu_quant(xq, xs, wq, ws, expert_indices, probs, hadamard, blockscale_selection="default",
                     store_h=False):
    """Fused routed fc1 + SwiGLU + fc2 input quantization.

    xq [Tp, K/2] uint8, xs [Tp, K/32] uint8: Q(x) along K (Tp = padded rows, segments 128-aligned)
    wq [E, 2F, K/2] uint8, ws [E, 2F/32, K/32] uint8: Q_tile(W1) (gate rows 0..F, value rows F..2F)
    expert_indices [Tp] int32, probs [Tp] (fp32 or bf16; 0 on padding rows), hadamard [32, 32]
    Returns y [Tp, 2F] bf16, qa [Tp, F/2], sa [Tp, F/32], qb [Tp/2, F] (view), sb [Tp/32, F] (view),
    and h [Tp, F] bf16 if store_h (tests only)."""
    Tp, KP = xq.shape
    K = 2 * KP
    E, F2, KPw = wq.shape
    assert KPw == KP and F2 % 2 == 0, (wq.shape, xq.shape)
    F = F2 // 2
    assert Tp % ALIGN_SIZE_M == 0 and F % 32 == 0 and K % 32 == 0, (Tp, F, K)
    assert xq.is_contiguous() and xs.is_contiguous() and expert_indices.dtype == torch.int32
    assert wq.stride(2) == 1 and ws.stride(2) == 1 and probs.shape[0] == Tp
    dev = xq.device
    h_t = hadamard.to(device=dev, dtype=torch.bfloat16).contiguous()
    y = torch.empty((Tp, F2), dtype=torch.bfloat16, device=dev)
    qa = torch.empty((Tp, F // 2), dtype=torch.uint8, device=dev)
    sa = torch.empty((Tp, F // 32), dtype=torch.uint8, device=dev)
    qb = torch.empty((F, Tp // 2), dtype=torch.uint8, device=dev)
    sb = torch.empty((F, Tp // 32), dtype=torch.uint8, device=dev)
    hout = torch.empty((Tp, F), dtype=torch.bfloat16, device=dev) if store_h else y
    grid = lambda M: (triton.cdiv(Tp, M["BLOCK_M"]), triton.cdiv(F, M["BLOCK_N"]))
    _fc1_swiglu_quant_kernel[grid](
        xq, xs, wq, ws, expert_indices, probs.contiguous(), h_t,
        y, qa, sa, qb, sb, hout,
        wq.stride(0), wq.stride(1), ws.stride(0), ws.stride(1),
        Tp, triton.next_power_of_2(Tp),
        F=F, K=K, SCALE_SELECTION=blockscale_selection, STORE_H=store_h,
    )
    out = (y, qa, sa, qb.t(), sb.t())
    return out + ((hout,) if store_h else ())
