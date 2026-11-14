"""Fused MXFP4 quantization of one bf16 activation or gradient tensor into both operands it is needed as.

For X [T, C] (T tokens, C features) and the 32x32 Hadamard H of the step, one kernel reads every tile of X
once and writes
    A = Q(X) along C           (forward GEMM operand for X = x, dgrad operand for X = dy)
    B = Q(H·X) along tokens    (wgrad operand; H applied to every block of 32 consecutive tokens)
which ALTO otherwise computes with three passes over X (quantize along C; bf16 matmul with H writing H·X;
quantize H·X along tokens).

Numerics: the scale rule, element rounding (round-to-nearest or stochastic) and packing are ALTO's own
Triton functions (`_calculate_scales`, `_pack_fp4`), so A is bit-identical to `convert_to_mxfp4(X, axis=-1)`.
For B, H·X is computed with `tl.dot` on bf16 inputs with fp32 accumulation and rounded to bf16, as the
separate bf16 matmul does; only the order of the 32 additions can differ, which changes a small fraction of
the rotated bf16 values by one bf16 unit in the last place before quantization.

Layouts match `convert_to_mxfp4`: A [T, C/2] uint8 with scales [T, C/32]; B returned as a [T/2, C] view of a
[C, T/2] buffer with scales a [T/32, C] view of a [C, T/32] buffer (what convert_to_mxfp4(axis=0) returns).
"""
import torch
import triton
import triton.language as tl

from .mxfp_quantization import _calculate_scales, _pack_fp4, is_cdna4


def _configs():
    shapes = [(64, 64, 4), (64, 128, 4), (128, 64, 4), (128, 128, 8), (32, 128, 4), (128, 32, 4)]
    return [triton.Config({"BLOCK_T": t, "BLOCK_C": c}, num_warps=w, num_stages=1) for t, c, w in shapes]


@triton.autotune(configs=_configs(), key=["T_BUCKET", "C_BUCKET", "USE_SR", "SCALE_SELECTION"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def _dual_quant_kernel(
    x_ptr, h_ptr,
    qa_ptr, sa_ptr, qb_ptr, sb_ptr,
    stride_xt, stride_xc,
    philox_seed, philox_offset,
    T, C, T_BUCKET, C_BUCKET,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    USE_SR: tl.constexpr, USE_ASM: tl.constexpr, SCALE_SELECTION: tl.constexpr,
):
    QB: tl.constexpr = 32
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_t = offs_t < T
    mask_c = offs_c < C
    x = tl.load(x_ptr + offs_t[:, None] * stride_xt + offs_c[None, :] * stride_xc,
                mask=mask_t[:, None] & mask_c[None, :], other=0.0)

    # independent Philox counter ranges for every tile and for the two outputs
    tile_elems = BLOCK_T * BLOCK_C // 2
    tile_id = pid_t * tl.num_programs(1) + pid_c
    n_tiles = tl.num_programs(0) * tl.num_programs(1)
    off_a = philox_offset + tile_id * tile_elems
    off_b = philox_offset + (n_tiles + tile_id) * tile_elems

    # ---- A: quantize along C (rows = tokens, blocks of 32 features)
    sa = _calculate_scales(x, BLOCK_M=BLOCK_T, BLOCK_N=BLOCK_C, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                           USE_DYNAMIC_CLIP=False, SCALE_SELECTION=SCALE_SELECTION)
    qa = _pack_fp4(x, sa, philox_seed, off_a, BLOCK_M=BLOCK_T, BLOCK_N=BLOCK_C, QUANT_BLOCK_SIZE=QB,
                   IS_2D_BLOCK=False, USE_SR=USE_SR, USE_ASM=USE_ASM)
    offs_c2 = pid_c * (BLOCK_C // 2) + tl.arange(0, BLOCK_C // 2)
    offs_cs = pid_c * (BLOCK_C // QB) + tl.arange(0, BLOCK_C // QB)
    tl.store(qa_ptr + offs_t[:, None] * (C // 2) + offs_c2[None, :], qa,
             mask=mask_t[:, None] & (offs_c2 < C // 2)[None, :])
    tl.store(sa_ptr + offs_t[:, None] * (C // QB) + offs_cs[None, :], sa,
             mask=mask_t[:, None] & (offs_cs < C // QB)[None, :])

    # ---- B: rotate every 32-token block with H (bf16 inputs, fp32 accumulation, bf16 result), then
    # quantize along tokens (rows = features, blocks of 32 tokens)
    hr = tl.arange(0, QB)
    ht = tl.load(h_ptr + hr[None, :] * QB + hr[:, None])                 # Hᵀ: ht[j, i] = H[i, j]
    xt = tl.trans(x)                                                      # [BLOCK_C, BLOCK_T]
    xg = tl.reshape(xt, (BLOCK_C * BLOCK_T // QB, QB))                    # rows (c, token group), cols j
    r = tl.dot(xg, ht, out_dtype=tl.float32)                              # r[(c,g), i] = (H x_g)[i, c]
    r = tl.reshape(r.to(x.dtype), (BLOCK_C, BLOCK_T))
    sb = _calculate_scales(r, BLOCK_M=BLOCK_C, BLOCK_N=BLOCK_T, QUANT_BLOCK_SIZE=QB, IS_2D_BLOCK=False,
                           USE_DYNAMIC_CLIP=False, SCALE_SELECTION=SCALE_SELECTION)
    qb = _pack_fp4(r, sb, philox_seed, off_b, BLOCK_M=BLOCK_C, BLOCK_N=BLOCK_T, QUANT_BLOCK_SIZE=QB,
                   IS_2D_BLOCK=False, USE_SR=USE_SR, USE_ASM=USE_ASM)
    offs_t2 = pid_t * (BLOCK_T // 2) + tl.arange(0, BLOCK_T // 2)
    offs_ts = pid_t * (BLOCK_T // QB) + tl.arange(0, BLOCK_T // QB)
    tl.store(qb_ptr + offs_c[:, None] * (T // 2) + offs_t2[None, :], qb,
             mask=mask_c[:, None] & (offs_t2 < T // 2)[None, :])
    tl.store(sb_ptr + offs_c[:, None] * (T // QB) + offs_ts[None, :], sb,
             mask=mask_c[:, None] & (offs_ts < T // QB)[None, :])


def dual_quantize_supported(x: torch.Tensor) -> bool:
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.shape[0] % 32 == 0 and x.shape[1] % 32 == 0
            and x.is_cuda and is_cdna4())


def dual_quantize(x: torch.Tensor, hadamard: torch.Tensor, use_sr: bool = False,
                  blockscale_selection: str = "default"):
    """x [T, C] bf16, hadamard [32, 32] (the matrix applied as H @ x_block). Returns
    (qa [T, C/2], sa [T, C/32], qb [T/2, C] view, sb [T/32, C] view)."""
    assert dual_quantize_supported(x), (x.shape, x.dtype)
    T, C = x.shape
    h = hadamard.to(device=x.device, dtype=x.dtype).contiguous()
    assert h.shape == (32, 32)
    qa = torch.empty((T, C // 2), dtype=torch.uint8, device=x.device)
    sa = torch.empty((T, C // 32), dtype=torch.uint8, device=x.device)
    qb = torch.empty((C, T // 2), dtype=torch.uint8, device=x.device)
    sb = torch.empty((C, T // 32), dtype=torch.uint8, device=x.device)
    seed = torch.randint(0, 2**31 - 1, (1,)).item()
    offset = torch.randint(0, 2**30 - 1, (1,)).item()
    grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(C, META["BLOCK_C"]))
    _dual_quant_kernel[grid](
        x, h, qa, sa, qb, sb, x.stride(0), x.stride(1), seed, offset,
        T, C, triton.next_power_of_2(T), triton.next_power_of_2(C),
        USE_SR=use_sr, USE_ASM=True, SCALE_SELECTION=blockscale_selection,
    )
    return qa, sa, qb.t(), sb.t()
