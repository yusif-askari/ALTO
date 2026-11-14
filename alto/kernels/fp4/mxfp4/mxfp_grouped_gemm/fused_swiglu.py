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


@triton.autotune(configs=_configs(), key=["F", "K", "TP_BUCKET", "STORE_H", "ROW_MAP", "STORE_Y"])
@triton.jit
def _fc1_swiglu_quant_kernel(
    xq_ptr, xs_ptr, w_ptr, ws_ptr, idx_ptr, p_ptr, rows_ptr, had_ptr,
    y_ptr, qa_ptr, sa_ptr, qb_ptr, sb_ptr, hout_ptr,
    stride_we, stride_wn, stride_wse, stride_wsn,
    Tp, TP_BUCKET,
    F: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SCALE_SELECTION: tl.constexpr, STORE_H: tl.constexpr, ROW_MAP: tl.constexpr,
    STORE_Y: tl.constexpr = True,
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
    if STORE_Y:      # False: profiling only (cost of keeping y for the backward)
        tl.store(y_ptr + offs_m[:, None] * (2 * F) + offs_n[None, :], g, mask=mask_mn)
        tl.store(y_ptr + offs_m[:, None] * (2 * F) + (F + offs_n)[None, :], v, mask=mask_mn)

    # ---- h = bf16(silu(g) * v * p), fp32 arithmetic from the bf16 g, v (one rounding at the end).
    # silu(g) = g / (1 + exp(-g)) with libdevice exp, exactly as torch.compile (inductor) generates it for
    # Megatron's weighted SwiGLU, so h matches the unfused path bit for bit (g * tl.sigmoid(g) differs in
    # the last fp32 bit for ~6 in 10^6 elements).
    if ROW_MAP:
        # probs in the unpadded token order: grouped row m holds token rows[m] (padding: rows[m] < 0, p = 0)
        src = tl.load(rows_ptr + offs_m, mask=mask_m, other=-1)
        p = tl.load(p_ptr + src, mask=src >= 0, other=0.0).to(tl.float32)
    else:
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
                     store_h=False, src_rows=None, store_y=True):
    """Fused routed fc1 + SwiGLU + fc2 input quantization.

    xq [Tp, K/2] uint8, xs [Tp, K/32] uint8: Q(x) along K (Tp = padded rows, segments 128-aligned)
    wq [E, 2F, K/2] uint8, ws [E, 2F/32, K/32] uint8: Q_tile(W1) (gate rows 0..F, value rows F..2F)
    expert_indices [Tp] int32, probs [Tp] (fp32 or bf16; 0 on padding rows), hadamard [32, 32]
    src_rows [Tp] int32 (optional): token of each grouped row (< 0: padding); probs are then given in the
    unpadded token order, probs [T] (no padded copy of the probs is needed)
    Returns y [Tp, 2F] bf16, qa [Tp, F/2], sa [Tp, F/32], qb [Tp/2, F] (view), sb [Tp/32, F] (view),
    and h [Tp, F] bf16 if store_h (tests only)."""
    Tp, KP = xq.shape
    K = 2 * KP
    E, F2, KPw = wq.shape
    assert KPw == KP and F2 % 2 == 0, (wq.shape, xq.shape)
    F = F2 // 2
    assert Tp % ALIGN_SIZE_M == 0 and F % 32 == 0 and K % 32 == 0, (Tp, F, K)
    assert xq.is_contiguous() and xs.is_contiguous() and expert_indices.dtype == torch.int32
    assert wq.stride(2) == 1 and ws.stride(2) == 1
    assert probs.shape[0] == Tp if src_rows is None else (src_rows.shape == (Tp,) and src_rows.dtype == torch.int32)
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
        xq, xs, wq, ws, expert_indices, probs.contiguous(), src_rows if src_rows is not None else expert_indices, h_t,
        y, qa, sa, qb, sb, hout,
        wq.stride(0), wq.stride(1), ws.stride(0), ws.stride(1),
        Tp, triton.next_power_of_2(Tp),
        F=F, K=K, SCALE_SELECTION=blockscale_selection, STORE_H=store_h, ROW_MAP=src_rows is not None, STORE_Y=store_y,
    )
    out = (y, qa, sa, qb.t(), sb.t())
    return out + ((hout,) if store_h else ())


# ================= backward: fc2 dgrad + weighted SwiGLU backward + fc1 gradient quantization =================
from ..mxfp_dual_quant import _pack_fp4_sr  # noqa: E402


def _bwd_configs():
    shapes = [(64, 32, 128, 4), (64, 64, 128, 4), (128, 32, 128, 4), (128, 64, 128, 8), (64, 64, 64, 4),
              (128, 64, 64, 8), (64, 32, 256, 4), (128, 32, 256, 8)]
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_D": d}, num_warps=w, num_stages=2)
            for m, n, d, w in shapes]


@triton.jit
def _quant_pair_store(t, rand_a, rand_b, ht, col0, offs_m, mask_m, m0, n_cols, Tp,
                      q_ptr, s_ptr, qt_ptr, st_ptr, F2: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, USE_SR: tl.constexpr,
                      SCALE_SELECTION: tl.constexpr):
    """One [BLOCK_M, BLOCK_N] bf16 tile t of a [Tp, F2] gradient, columns col0..col0+BLOCK_N: Q(t) along
    the columns into q [Tp, F2/2] / s [Tp, F2/32], and Q(H·t) along tokens into qt [F2, Tp/2] / st [F2, Tp/32],
    stochastically rounded with the given random numbers if USE_SR (else round to nearest)."""
    QB: tl.constexpr = 32
    offs_c = col0 + tl.arange(0, BLOCK_N)
    mask_c = offs_c < col0 + n_cols
    sa = _calculate_scales(t, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                           USE_DYNAMIC_CLIP=False, SCALE_SELECTION=SCALE_SELECTION)
    if USE_SR:
        qa = _pack_fp4_sr(t, sa, rand_a, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QB)
    else:
        qa = _pack_fp4(t, sa, 0, 0, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                       USE_SR=False, USE_ASM=True)
    offs_c2 = col0 // 2 + tl.arange(0, BLOCK_N // 2)
    offs_cs = col0 // QB + tl.arange(0, BLOCK_N // QB)
    tl.store(q_ptr + offs_m[:, None] * (F2 // 2) + offs_c2[None, :], qa,
             mask=mask_m[:, None] & (offs_c2 < (col0 + n_cols) // 2)[None, :])
    tl.store(s_ptr + offs_m[:, None] * (F2 // QB) + offs_cs[None, :], sa,
             mask=mask_m[:, None] & (offs_cs < (col0 + n_cols) // QB)[None, :])
    tg = tl.reshape(tl.trans(t), (BLOCK_N * BLOCK_M // QB, QB))
    r = tl.dot(tg, ht, out_dtype=tl.float32)
    r = tl.reshape(r.to(tl.bfloat16), (BLOCK_N, BLOCK_M))
    sb = _calculate_scales(r, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_M, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                           USE_DYNAMIC_CLIP=False, SCALE_SELECTION=SCALE_SELECTION)
    if USE_SR:
        qb = _pack_fp4_sr(r, sb, rand_b, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_M, QUANT_BLOCK_SIZE=QB)
    else:
        qb = _pack_fp4(r, sb, 0, 0, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_M, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                       USE_SR=False, USE_ASM=True)
    offs_m2 = m0 // 2 + tl.arange(0, BLOCK_M // 2)
    offs_ms = m0 // QB + tl.arange(0, BLOCK_M // QB)
    tl.store(qt_ptr + offs_c[:, None] * (Tp // 2) + offs_m2[None, :], qb,
             mask=mask_c[:, None] & (offs_m2 < Tp // 2)[None, :])
    tl.store(st_ptr + offs_c[:, None] * (Tp // QB) + offs_ms[None, :], sb,
             mask=mask_c[:, None] & (offs_ms < Tp // QB)[None, :])


@triton.autotune(configs=_bwd_configs(), key=["F", "D", "TP_BUCKET", "USE_SR", "ROW_MAP"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def _fc2_dgrad_swiglu_bwd_kernel(
    dq_ptr, ds_ptr, w_ptr, ws_ptr, idx_ptr, y_ptr, p_ptr, rows_ptr, had_ptr,
    gq_ptr, gs_ptr, gqt_ptr, gst_ptr, dpp_ptr, dy_ptr,
    stride_we, stride_wd, stride_wse, stride_wsd, stride_wsf,
    philox_seed, philox_offset, Tp, TP_BUCKET, T_OUT,
    F: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    USE_SR: tl.constexpr, SCALE_SELECTION: tl.constexpr, STORE_DY: tl.constexpr, ROW_MAP: tl.constexpr,
):
    QB: tl.constexpr = 32
    DP: tl.constexpr = D // 2
    DS: tl.constexpr = D // QB
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_np = n0 // 2 + tl.arange(0, BLOCK_N // 2)
    mask_m = offs_m < Tp
    mask_n = offs_n < F
    expert = tl.load(idx_ptr + m0)

    # ---- dh tile = Q(dy) · Q_tile(W2): contraction over D; W2 [E, D, F/2] packed along F (output dim)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_e = w_ptr + expert * stride_we
    ws_e = ws_ptr + expert * stride_wse
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        offs_dp = d0 // 2 + tl.arange(0, BLOCK_D // 2)
        offs_ds = d0 // QB + tl.arange(0, BLOCK_D // QB)
        go = tl.load(dq_ptr + offs_m[:, None] * DP + offs_dp[None, :],
                     mask=mask_m[:, None] & (offs_dp < DP)[None, :], other=0)
        go_s = tl.load(ds_ptr + offs_m[:, None] * DS + offs_ds[None, :],
                       mask=mask_m[:, None] & (offs_ds < DS)[None, :], other=127)
        w = tl.load(w_e + offs_d[:, None] * stride_wd + offs_np[None, :],
                    mask=(offs_d < D)[:, None] & (offs_np < F // 2)[None, :], other=0)
        # scales of the 32x32 tiles: [output column // 32, contraction block]
        w_s = tl.load(ws_e + (offs_n // QB)[:, None] * stride_wsf + offs_ds[None, :] * stride_wsd,
                      mask=mask_n[:, None] & (offs_ds < DS)[None, :], other=127)
        acc += tl.dot_scaled(go, go_s, "e2m1", w, w_s, "e2m1", lhs_k_pack=True, rhs_k_pack=False,
                             out_dtype=tl.float32)

    # ---- weighted SwiGLU backward, the arithmetic torch.compile generates for Megatron's
    # weighted_swiglu_back (dh rounded to bf16 as the unfused dgrad stores it; y = [g | v] bf16; p fp32)
    dh = acc.to(tl.bfloat16).to(tl.float32)
    mask_mn = mask_m[:, None] & mask_n[None, :]
    g = tl.load(y_ptr + offs_m[:, None] * (2 * F) + offs_n[None, :], mask=mask_mn, other=0.0).to(tl.float32)
    v = tl.load(y_ptr + offs_m[:, None] * (2 * F) + (F + offs_n)[None, :], mask=mask_mn, other=0.0).to(tl.float32)
    if ROW_MAP:
        src = tl.load(rows_ptr + offs_m, mask=mask_m, other=-1)
        p = tl.load(p_ptr + src, mask=src >= 0, other=0.0)
    else:
        p = tl.load(p_ptr + offs_m, mask=mask_m, other=0.0)
    dhp = dh * p[:, None]
    s = tl.sigmoid(g)
    d_gate = ((dhp * s) * (g * (1.0 - s) + 1.0) * v).to(tl.bfloat16)
    silu = g / (libdevice.exp(-g) + 1.0)
    d_val = (dhp * silu).to(tl.bfloat16)
    # d(prob) partial sum over this tile's columns (summed over the column tiles on the host)
    dpp = tl.sum((silu * v) * dh, axis=1)
    if ROW_MAP:
        tl.store(dpp_ptr + pid_n * T_OUT + src, dpp, mask=src >= 0)       # unpadded token order
    else:
        tl.store(dpp_ptr + pid_n * Tp + offs_m, dpp, mask=mask_m)
    if STORE_DY:
        tl.store(dy_ptr + offs_m[:, None] * (2 * F) + offs_n[None, :], d_gate, mask=mask_mn)
        tl.store(dy_ptr + offs_m[:, None] * (2 * F) + (F + offs_n)[None, :], d_val, mask=mask_mn)

    # ---- fc1 gradient operands: Q_sr(dy1) along 2F (dgrad), Q_sr(H·dy1) along tokens (wgrad)
    hr = tl.arange(0, QB)
    ht = tl.load(had_ptr + hr[None, :] * QB + hr[:, None])
    if USE_SR:
        # one Philox call: 4 random numbers = one packed pair in each of the 4 quantized outputs
        tile_id = pid_m * tl.num_programs(1) + pid_n
        roffs = philox_offset + tile_id * (BLOCK_M * BLOCK_N // 2) + \
            tl.arange(0, BLOCK_M)[:, None] * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)[None, :]
        r1, r2, r3, r4 = tl.randint4x(philox_seed, roffs)
        r3 = tl.reshape(r3, (BLOCK_N, BLOCK_M // 2))
        r4 = tl.reshape(r4, (BLOCK_N, BLOCK_M // 2))
    else:
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
    _quant_pair_store(d_gate, r1, r3, ht, n0, offs_m, mask_m, m0, F, Tp, gq_ptr, gs_ptr, gqt_ptr, gst_ptr,
                      F2=2 * F, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, USE_SR=USE_SR, SCALE_SELECTION=SCALE_SELECTION)
    _quant_pair_store(d_val, r2, r4, ht, F + n0, offs_m, mask_m, m0, 2 * F - F, Tp, gq_ptr, gs_ptr, gqt_ptr,
                      gst_ptr, F2=2 * F, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, USE_SR=USE_SR,
                      SCALE_SELECTION=SCALE_SELECTION)


def fc2_dgrad_swiglu_bwd(dq, ds, w2q, w2s, expert_indices, y, probs, hadamard, use_sr=True,
                         blockscale_selection="default", store_dy=False, src_rows=None):
    """Fused routed fc2 dgrad + weighted SwiGLU backward + quantization of the fc1 gradient.

    dq [Tp, D/2], ds [Tp, D/32]: Q(dout) along D (fc2 output dim); w2q [E, D, F/2], w2s [E, D/32, F/32]:
    Q_tile(W2); y [Tp, 2F] bf16 (fc1 output saved by the forward); probs [Tp] fp32.
    Returns gq [Tp, F] (= 2F/2), gs [Tp, 2F/32], gq_t [Tp/2, 2F] (view), gs_t [Tp/32, 2F] (view):
    the fc1 output gradient dy1 quantized along 2F (fc1 dgrad) and, rotated, along tokens (fc1 wgrad);
    and d(probs) [Tp] fp32; plus dy1 [Tp, 2F] bf16 if store_dy (tests only).
    src_rows [Tp] int32 (optional): token of each grouped row (< 0: padding); probs [T] and d(probs) [T]
    are then in the unpadded token order."""
    Tp, DP = dq.shape
    D = 2 * DP
    E, Dw, FP = w2q.shape
    F = 2 * FP
    assert Dw == D and y.shape == (Tp, 2 * F) and Tp % ALIGN_SIZE_M == 0 and F % 32 == 0 and D % 32 == 0
    assert dq.is_contiguous() and ds.is_contiguous() and y.is_contiguous() and w2q.stride(2) == 1
    dev = dq.device
    h_t = hadamard.to(device=dev, dtype=torch.bfloat16).contiguous()
    gq = torch.empty((Tp, F), dtype=torch.uint8, device=dev)
    gs = torch.empty((Tp, 2 * F // 32), dtype=torch.uint8, device=dev)
    gqt = torch.empty((2 * F, Tp // 2), dtype=torch.uint8, device=dev)
    gst = torch.empty((2 * F, Tp // 32), dtype=torch.uint8, device=dev)
    n_tiles_max = triton.cdiv(F, 32)
    t_out = probs.shape[0] if src_rows is not None else Tp
    if src_rows is not None:
        assert src_rows.shape == (Tp,) and src_rows.dtype == torch.int32 and src_rows.is_contiguous()
    dpp = torch.zeros((n_tiles_max, t_out), dtype=torch.float32, device=dev)
    dy = torch.empty((Tp, 2 * F), dtype=torch.bfloat16, device=dev) if store_dy else y
    seed = torch.randint(0, 2**31 - 1, (1,)).item()
    offset = torch.randint(0, 2**30 - 1, (1,)).item()
    grid = lambda M: (triton.cdiv(Tp, M["BLOCK_M"]), triton.cdiv(F, M["BLOCK_N"]))
    _fc2_dgrad_swiglu_bwd_kernel[grid](
        dq, ds, w2q, w2s, expert_indices, y, probs.contiguous().float(),
        src_rows if src_rows is not None else expert_indices, h_t,
        gq, gs, gqt, gst, dpp, dy,
        w2q.stride(0), w2q.stride(1), w2s.stride(0), w2s.stride(1), w2s.stride(2),
        seed, offset, Tp, triton.next_power_of_2(Tp), t_out,
        F=F, D=D, USE_SR=use_sr, SCALE_SELECTION=blockscale_selection, STORE_DY=store_dy,
        ROW_MAP=src_rows is not None,
    )
    n_tiles = triton.cdiv(F, _fc2_dgrad_swiglu_bwd_kernel.best_config.kwargs["BLOCK_N"])
    out = (gq, gs, gqt.t(), gst.t(), dpp[:n_tiles].sum(0))
    return out + ((dy,) if store_dy else ())
